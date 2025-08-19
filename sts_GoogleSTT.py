import os
import threading
import sounddevice as sd
import numpy as np
import time
from dotenv import load_dotenv
from openai import OpenAI
from elevenlabs.client import ElevenLabs
import pyaudio
from google.cloud import speech

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 200
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Updated voice ID for ElevenLabs (Mine), Since the previous one could have only be accessed through a subscribed account and my 11labs account ain't one.
# ELEVENLABS_VOICE_ID = "nbOs83cg1fbwnhG6tlRB"  # Original(Previous) voice ID
SYSTEM_PROMPT = "You are a helpful voice assistant. Be concise and conversational."

# --- STATE MANAGEMENT ---
is_speaking_event = threading.Event()
interruption_event = threading.Event()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Set up Google STT client (needs GOOGLE_APPLICATION_CREDENTIALS env var set)
google_stt_client = speech.SpeechClient()

class GoogleSTT:
    def __init__(self, on_transcript_callback):
        self.on_transcript_callback = on_transcript_callback
        self.final_transcript_buffer = []
        self.last_final_time = time.time()
        threading.Thread(target=self.monitor_user_input, daemon=True).start()

    def process_responses(self, responses):
        """Process Google streaming STT responses"""
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            transcript = result.alternatives[0].transcript.strip()

            if result.is_final:
                self.final_transcript_buffer.append(transcript)
                self.last_final_time = time.time()
                if is_speaking_event.is_set():
                    print("🎤 User interrupted the bot.")
                    interruption_event.set()

    def monitor_user_input(self):
        while True:
            time.sleep(0.4)
            now = time.time()
            if self.final_transcript_buffer and (now - self.last_final_time) > 1.8:
                full_sentence = " ".join(self.final_transcript_buffer).strip()
                self.final_transcript_buffer = []
                if full_sentence:
                    print(f"🗣️ YOU SAID: {full_sentence}")
                    self.on_transcript_callback(full_sentence)

def get_llm_reply(conversation, result_dict):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=conversation[-10:],
            temperature=0.6,
            max_tokens=260,
        )
        result_dict["reply"] = response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ LLM ERROR:", e)
        result_dict["reply"] = ""

def speak_with_interruption(text, first_chunk=None, audio_stream=None):
    is_speaking_event.set()
    interruption_event.clear()
    print("🔊 Bot is speaking...")

    p = pyaudio.PyAudio()
    stream_out = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)

    try:
        # If first_chunk and audio_stream are provided, use them
        if first_chunk is not None and audio_stream is not None:
            stream_out.write(first_chunk)
            for chunk in audio_stream:
                if interruption_event.is_set():
                    print("🛑 TTS playback stopped due to interruption.")
                    break
                if chunk:
                    stream_out.write(chunk)
        else:
            # Fallback to original streaming if not provided
            audio_stream = elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id="eleven_turbo_v2",
                output_format="pcm_24000"
            )
            for chunk in audio_stream:
                if interruption_event.is_set():
                    print("🛑 TTS playback stopped due to interruption.")
                    break
                if chunk:
                    stream_out.write(chunk)

        print("🔊 TTS playback complete.")

    except Exception as e:
        print(f"❌ TTS streaming error: {e}")
    finally:
        stream_out.stop_stream()
        stream_out.close()
        p.terminate()
        is_speaking_event.clear()
        interruption_event.clear()
        print("🎤 Bot finished speaking.")

def main():
    print("🎤 Initializing Real-Time VoiceBot ...")
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    stop_event = threading.Event()
    llm_thread = None

    def on_transcript(sentence):
        nonlocal llm_thread
        if llm_thread and llm_thread.is_alive():
            return

        stt_start = time.time()
        conversation.append({"role": "user", "content": sentence})
        print("🤖 THINKING...")

        llm_result = {}
        llm_thread = threading.Thread(target=get_llm_reply, args=(conversation, llm_result))
        llm_thread.start()

        def run_tts_after_llm():
            llm_thread.join()
            stt_end = time.time()
            stt_time = stt_end - stt_start
            print(f"Time Taken for STT+LLM : {stt_time:.2f} seconds")

            reply = llm_result.get("reply", "")
            if not reply:
                print("⚠️ No reply from LLM.")
                return

            if interruption_event.is_set():
                print("⚠️ LLM response discarded due to interruption.")
                interruption_event.clear()
                return

            print(f"\n🤖 BOT: {reply}\n")
            conversation.append({"role": "assistant", "content": reply})

            def tts_with_timing(text):
                tts_start = time.time()
                try:
                    audio_stream = elevenlabs_client.text_to_speech.stream(
                        text=text,
                        voice_id=ELEVENLABS_VOICE_ID,
                        model_id="eleven_turbo_v2",
                        output_format="pcm_24000"
                    )
                    # Measure time to get the first chunk (initiation)
                    first_chunk = next(audio_stream)
                    tts_init_time = time.time() - tts_start
                    print(f"Time Taken to initiate TTS : {tts_init_time:.2f} seconds")
                    # Play the first chunk
                    speak_with_interruption(text, first_chunk, audio_stream)
                except Exception as e:
                    print(f"❌ TTS streaming error: {e}")

            tts_thread = threading.Thread(target=tts_with_timing, args=(reply,), daemon=True)
            tts_thread.start()

        threading.Thread(target=run_tts_after_llm, daemon=True).start()

    stt = GoogleSTT(on_transcript)

    print("🟢 VoiceBot is running... Speak into your mic")

    def audio_input_stream():
        def request_generator():
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="int16",
                blocksize=FRAMES_PER_BUFFER,
            ) as stream_in:
                while not stop_event.is_set():
                    audio_data, _ = stream_in.read(FRAMES_PER_BUFFER)
                    yield speech.StreamingRecognizeRequest(
                        audio_content=audio_data.tobytes()
                    )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        try:
            responses = google_stt_client.streaming_recognize(
                streaming_config, request_generator()
            )
            stt.process_responses(responses)
        except Exception as e:
            print("❌ Google STT Error:", e)

    mic_thread = threading.Thread(target=audio_input_stream)
    mic_thread.start()

    try:
        input("🔘 Press Enter to stop...\n")
    except KeyboardInterrupt:
        print("🛑 Manual Interrupt.")
    finally:
        stop_event.set()
        mic_thread.join()
        print("👋 VoiceBot shut down.")

if __name__ == "__main__":
    main()
