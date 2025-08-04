import os
import threading
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from openai import OpenAI
from elevenlabs import play, stream
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# Audio input config
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 200
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# Voice & system prompt
ELEVENLABS_VOICE_ID = "nbOs83cg1fbwnhG6tlRB"  # Use your own voice ID here
SYSTEM_PROMPT = "You are a helpful voice assistant."

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
dg_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

############################
# Deepgram STT Wrapper
############################

class DeepgramSTT:
    def __init__(self, on_transcript_callback):
        self.dg = dg_client
        self.connection = self.dg.listen.websocket.v("1")
        self.on_transcript_callback = on_transcript_callback
        self.setup_events()

    def setup_events(self):
        self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
        self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
        self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)

    def on_open(self, *args):
        print("🟢 DEEPGRAM CONNECTION OPENED.")

    def on_close(self, *args):
        print("🔴 DEEPGRAM CONNECTION CLOSED.")

    def on_transcript(self, _, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        is_final = getattr(result, "is_final", False)
        if sentence and is_final:
            print(f"🗣️ YOU SAID: {sentence}")
            self.on_transcript_callback(sentence)

    def start_connection(self):
        options = LiveOptions(
            model="nova-3",
            language="en-US",
            punctuate=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
            vad_events=True,
            endpointing=1500
        )
        if not self.connection.start(options):
            print("❌ FAILED TO START DEEPGRAM.")
            return False
        return True

    def send_audio_data(self, data):
        self.connection.send(data)

    def finish(self):
        self.connection.finish()
        print("✅ STT STREAM CLOSED.")

#########################
# LLM + Streaming TTS
#########################

def get_llm_reply(conversation, result_dict):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=conversation[-10:],  # Keep short context
            temperature=0.6,
            max_tokens=260,
        )
        result_dict["reply"] = response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ LLM ERROR:", e)
        result_dict["reply"] = ""

def speak_streaming(text):
    try:
        audio_stream = elevenlabs_client.text_to_speech.stream(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        print("🔊 Streaming TTS playback started...")
        stream(audio_stream)  # Play streamed audio chunk by chunk
        print("🔊 TTS playback complete.")
    except Exception as e:
        print("❌ TTS streaming error:", e)

############################
# Main VoiceBot Logic
############################

def main():
    print("🎤 Initializing Real-Time VoiceBot...")

    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    stop_event = threading.Event()

    # Transcript callback, now pipelined with background threads
    def on_transcript(sentence):
        conversation.append({"role": "user", "content": sentence})
        print("🤖 THINKING...")

        # Threaded LLM inference
        llm_result = {}
        llm_thread = threading.Thread(target=get_llm_reply, args=(conversation, llm_result))
        llm_thread.start()

        def run_tts_after_llm():
            llm_thread.join()
            reply = llm_result.get("reply", "")
            if not reply:
                print("⚠️ No reply from LLM.")
                return

            print(f"\n🤖 BOT: {reply}\n")
            conversation.append({"role": "assistant", "content": reply})

            # Now stream TTS audio in a thread
            tts_thread = threading.Thread(target=speak_streaming, args=(reply,), daemon=True)
            tts_thread.start()

        threading.Thread(target=run_tts_after_llm, daemon=True).start()

    # Start Deepgram STT
    dg_stt = DeepgramSTT(on_transcript)
    if not dg_stt.start_connection():
        return

    print("🟢 VoiceBot is running... Speak into your mic")

    def audio_input_stream():
        def mic_callback(indata, frames, time_info, status):
            if status:
                print("⚠️ AUDIO STATUS:", status)
            audio_data = (indata * 32767).astype(np.int16).tobytes()
            dg_stt.send_audio_data(audio_data)

        try:
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                callback=mic_callback,
                blocksize=FRAMES_PER_BUFFER,
                dtype="float32"
            ):
                stop_event.wait()
        except Exception as e:
            print("❌ Audio Error:", e)
        finally:
            dg_stt.finish()

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
