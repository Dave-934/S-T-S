# import os
# import threading
# import sounddevice as sd
# import numpy as np
# import time
# from dotenv import load_dotenv
# from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
# from openai import OpenAI
# from elevenlabs import play, stream
# from elevenlabs.client import ElevenLabs

# # Load environment variables
# load_dotenv()

# # Audio input config
# SAMPLE_RATE = 16000
# CHANNELS = 1
# CHUNK_DURATION_MS = 200
# FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# # Voice & system prompt
# ELEVENLABS_VOICE_ID = "nbOs83cg1fbwnhG6tlRB"  # Use your own voice ID here
# SYSTEM_PROMPT = "You are a helpful voice assistant."

# # Initialize clients
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
# dg_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

# ############################
# # Deepgram STT Wrapper
# ############################

# class DeepgramSTT:
#     def __init__(self, on_transcript_callback):
#         self.dg = dg_client
#         self.connection = self.dg.listen.websocket.v("1")
#         self.on_transcript_callback = on_transcript_callback
#         self.setup_events()

#     def setup_events(self):
#         self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
#         self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
#         self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)

#     def on_open(self, *args):
#         print("🟢 DEEPGRAM CONNECTION OPENED.")

#     def on_close(self, *args):
#         print("🔴 DEEPGRAM CONNECTION CLOSED.")

#     def on_transcript(self, _, result, **kwargs):
#         sentence = result.channel.alternatives[0].transcript
#         is_final = getattr(result, "is_final", False)
#         if sentence and is_final:
#             print(f"🗣️ YOU SAID: {sentence}")
#             self.on_transcript_callback(sentence)

#     def start_connection(self):
#         options = LiveOptions(
#             model="nova-3",
#             language="en-US",
#             punctuate=True,
#             encoding="linear16",
#             sample_rate=SAMPLE_RATE,
#             channels=1,
#             vad_events=True,
#             endpointing=1500
#         )
#         if not self.connection.start(options):
#             print("❌ FAILED TO START DEEPGRAM.")
#             return False
#         return True

#     def send_audio_data(self, data):
#         self.connection.send(data)

#     def finish(self):
#         self.connection.finish()
#         print("✅ STT STREAM CLOSED.")

# #########################
# # LLM + Streaming TTS
# #########################

# def get_llm_reply(conversation, result_dict):
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4.1-nano",
#             messages=conversation[-10:],  # Keep short context
#             temperature=0.6,
#             max_tokens=260,
#         )
#         result_dict["reply"] = response.choices[0].message.content.strip()
#     except Exception as e:
#         print("❌ LLM ERROR:", e)
#         result_dict["reply"] = ""

# def speak_streaming(text):
#     try:
#         audio_stream = elevenlabs_client.text_to_speech.stream(
#             text=text,
#             voice_id=ELEVENLABS_VOICE_ID,
#             model_id="eleven_multilingual_v2",
#             output_format="mp3_44100_128"
#         )
#         print("🔊 Streaming TTS playback started...")
#         stream(audio_stream)  # Play streamed audio chunk by chunk
#         print("🔊 TTS playback complete.")
#     except Exception as e:
#         print("❌ TTS streaming error:", e)

# ############################
# # Main VoiceBot Logic
# ############################

# def main():
#     print("🎤 Initializing Real-Time VoiceBot...")

#     conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
#     stop_event = threading.Event()

#     # Transcript callback, now pipelined with background threads
#     def on_transcript(sentence):
#         stt_start = time.time()
#         conversation.append({"role": "user", "content": sentence})
#         print("🤖 THINKING...")

#         # Threaded LLM inference
#         llm_result = {}
#         llm_thread = threading.Thread(target=get_llm_reply, args=(conversation, llm_result))
#         llm_thread.start()

#         def run_tts_after_llm():
#             llm_thread.join()
#             stt_end = time.time()
#             stt_time = stt_end - stt_start
#             print(f"Time Taken for STT+LLM : {stt_time:.2f} seconds")

#             reply = llm_result.get("reply", "")
#             if not reply:
#                 print("⚠️ No reply from LLM.")
#                 return

#             print(f"\n🤖 BOT: {reply}\n")
#             conversation.append({"role": "assistant", "content": reply})

#             def tts_with_timing(text):
#                 tts_start = time.time()
#                 speak_streaming(text)
#                 tts_end = time.time()
#                 tts_time = tts_end - tts_start
#                 print(f"Time Taken for TTS : {tts_time:.2f} seconds")

#             # Now stream TTS audio in a thread
#             tts_thread = threading.Thread(target=tts_with_timing, args=(reply,), daemon=True)
#             tts_thread.start()

#         threading.Thread(target=run_tts_after_llm, daemon=True).start()

#     # Start Deepgram STT
#     dg_stt = DeepgramSTT(on_transcript)
#     if not dg_stt.start_connection():
#         return

#     print("🟢 VoiceBot is running... Speak into your mic")

#     def audio_input_stream():
#         def mic_callback(indata, frames, time_info, status):
#             if status:
#                 print("⚠️ AUDIO STATUS:", status)
#             audio_data = (indata * 32767).astype(np.int16).tobytes()
#             dg_stt.send_audio_data(audio_data)

#         try:
#             with sd.InputStream(
#                 channels=CHANNELS,
#                 samplerate=SAMPLE_RATE,
#                 callback=mic_callback,
#                 blocksize=FRAMES_PER_BUFFER,
#                 dtype="float32"
#             ):
#                 stop_event.wait()
#         except Exception as e:
#             print("❌ Audio Error:", e)
#         finally:
#             dg_stt.finish()

#     mic_thread = threading.Thread(target=audio_input_stream)
#     mic_thread.start()

#     try:
#         input("🔘 Press Enter to stop...\n")
#     except KeyboardInterrupt:
#         print("🛑 Manual Interrupt.")
#     finally:
#         stop_event.set()
#         mic_thread.join()
#         print("👋 VoiceBot shut down.")

# if __name__ == "__main__":
#     main()


































































































# import os
# import threading
# import sounddevice as sd
# import numpy as np
# import time
# from dotenv import load_dotenv
# from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
# from openai import OpenAI
# from elevenlabs.client import ElevenLabs
# import pyaudio

# # Load environment variables
# load_dotenv()

# # --- CONFIGURATION ---
# SAMPLE_RATE = 16000
# CHANNELS = 1
# ELEVENLABS_VOICE_ID = "nbOs83cg1fbwnhG6tlRB"
# SYSTEM_PROMPT = "You are a helpful voice assistant."

# # --- STATE MANAGEMENT ---
# is_speaking_event = threading.Event()
# interruption_event = threading.Event()
# response_lock = threading.Lock() # NEW: Lock to prevent response race conditions

# # Initialize clients
# openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
# dg_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

# ############################
# # Deepgram STT Wrapper
# ############################
# class DeepgramSTT:
#     def __init__(self, on_transcript_callback):
#         self.dg = dg_client
#         self.connection = self.dg.listen.websocket.v("1")
#         self.on_transcript_callback = on_transcript_callback
#         self.setup_events()

#     def setup_events(self):
#         self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
#         self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
#         self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
#         self.connection.on(LiveTranscriptionEvents.Error, self.on_error)

#     def on_open(self, *args):
#         print("🟢 DEEPGRAM CONNECTION OPENED.")

#     def on_close(self, *args):
#         print("🔴 DEEPGRAM CONNECTION CLOSED.")
        
#     def on_error(self, _, error, **kwargs):
#         print(f"❌ DEEPGRAM ERROR: {error}")

#     def on_transcript(self, _, result, **kwargs):
#         sentence = result.channel.alternatives[0].transcript
#         is_final = result.is_final

#         if sentence and is_final:
#             # If the user speaks while the bot is talking, set the interruption event
#             if is_speaking_event.is_set():
#                 print("🎤 User interrupted the bot.")
#                 interruption_event.set()
#                 return # Stop processing this transcript, just interrupt

#             self.on_transcript_callback(sentence)

#     def start_connection(self):
#         options = LiveOptions(
#             model="nova-3",
#             language="en-US",
#             punctuate=True,
#             encoding="linear16",
#             sample_rate=SAMPLE_RATE,
#             channels=1,
#             endpointing=1000,
#             vad_events=True,
#         )
#         try:
#             if not self.connection.start(options):
#                 print("❌ FAILED TO START DEEPGRAM.")
#                 return False
#             return True
#         except Exception as e:
#             print(f"❌ Could not start Deepgram connection: {e}")
#             return False

#     def send_audio_data(self, data):
#         try:
#             self.connection.send(data)
#         except Exception as e:
#             print(f"❌ Error sending audio data: {e}")

#     def finish(self):
#         self.connection.finish()
#         print("✅ STT STREAM CLOSED.")

# #########################
# # LLM + Streaming TTS with Interruption
# #########################
# def get_llm_reply(conversation, result_dict):
#     try:
#         # Add current time to system context for time-sensitive queries
#         current_time_prompt = f"Current date and time is {time.strftime('%A, %B %d, %Y at %I:%M %p IST')}."
#         full_conversation = [{"role": "system", "content": f"{SYSTEM_PROMPT} {current_time_prompt}"}] + conversation[-10:]

#         response = openai_client.chat.completions.create(
#             model="gpt-4.1-nano",
#             messages=full_conversation,
#             temperature=0.7,
#             max_tokens=260,
#         )
#         result_dict["reply"] = response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"❌ LLM ERROR: {e}")
#         result_dict["reply"] = ""

# def speak_with_interruption(text):
#     is_speaking_event.set()
#     interruption_event.clear()
#     print("🔊 Bot is speaking...")

#     p = pyaudio.PyAudio()
#     stream_out = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

#     try:
#         audio_stream = elevenlabs_client.text_to_speech.stream(
#             text=text,
#             voice_id=ELEVENLABS_VOICE_ID,
#             model_id="eleven_turbo_v2",
#             output_format="pcm_24000"
#         )

#         for chunk in audio_stream:
#             if interruption_event.is_set():
#                 print("🛑 TTS playback stopped due to interruption.")
#                 break
#             if chunk:
#                 stream_out.write(chunk)
        
#     except Exception as e:
#         print(f"❌ TTS streaming error: {e}")
#     finally:
#         stream_out.stop_stream()
#         stream_out.close()
#         p.terminate()
#         is_speaking_event.clear()
#         # Do not clear interruption_event here, let the main loop handle it
#         print("🎤 Bot finished speaking.")

# ############################
# # Main VoiceBot Logic
# ############################
# def main():
#     print("🎤 Initializing Real-Time VoiceBot...")
#     conversation = []
#     stop_event = threading.Event()

#     def on_transcript(sentence):
#         # MODIFIED: Use a non-blocking lock to prevent race conditions
#         if not response_lock.acquire(blocking=False):
#             print("--- Bot is busy, ignoring transcript ---")
#             return

#         stt_start = time.time()
#         conversation.append({"role": "user", "content": sentence})
#         print(f"🗣️ YOU SAID: {sentence}")
#         print("🤖 THINKING...")

#         llm_result = {}
#         llm_thread = threading.Thread(target=get_llm_reply, args=(conversation, llm_result))
#         llm_thread.start()

#         def run_tts_after_llm():
#             try:
#                 llm_thread.join()
#                 stt_end = time.time()
#                 print(f"Time Taken for STT+LLM : {stt_end - stt_start:.2f} seconds")

#                 reply = llm_result.get("reply", "")
#                 if not reply:
#                     print("⚠️ No reply from LLM.")
#                     return
                
#                 if interruption_event.is_set():
#                     print("⚠️ LLM response discarded due to interruption.")
#                     interruption_event.clear()
#                     return

#                 print(f"\n🤖 BOT: {reply}\n")
#                 conversation.append({"role": "assistant", "content": reply})

#                 tts_start = time.time()
#                 speak_with_interruption(reply)
#                 tts_end = time.time()
#                 print(f"Time Taken for TTS : {tts_end - tts_start:.2f} seconds")
#             finally:
#                 # MODIFIED: Always release the lock when the response cycle is complete
#                 response_lock.release()

#         threading.Thread(target=run_tts_after_llm, daemon=True).start()

#     dg_stt = DeepgramSTT(on_transcript)
#     if not dg_stt.start_connection():
#         return

#     print("🟢 VoiceBot is running... Speak into your mic")

#     def audio_input_stream():
#         CHUNK_SIZE = int(SAMPLE_RATE * 200 / 1000) # 200ms chunks
#         mic_stream = sd.InputStream(
#             channels=CHANNELS,
#             samplerate=SAMPLE_RATE,
#             blocksize=CHUNK_SIZE,
#             dtype="float32"
#         )
#         mic_stream.start()
#         print("🎤 Microphone stream started.")
#         while not stop_event.is_set():
#             indata, overflowed = mic_stream.read(CHUNK_SIZE)
#             if overflowed:
#                 print("⚠️ AUDIO OVERFLOW")
#             audio_data = (indata * 32767).astype(np.int16).tobytes()
#             dg_stt.send_audio_data(audio_data)
#         mic_stream.stop()
#         mic_stream.close()
#         print("🎤 Microphone stream stopped.")
#         dg_stt.finish()

#     mic_thread = threading.Thread(target=audio_input_stream)
#     mic_thread.start()

#     try:
#         input("🔘 Press Enter to stop...\n")
#     except KeyboardInterrupt:
#         print("🛑 Manual Interrupt.")
#     finally:
#         stop_event.set()
#         mic_thread.join()
#         print("👋 VoiceBot shut down.")

# if __name__ == "__main__":
#     main()
























































import os
import threading
import sounddevice as sd
import numpy as np
import time
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from openai import OpenAI
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import pyaudio

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION_MS = 200
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
ELEVENLABS_VOICE_ID = "nbOs83cg1fbwnhG6tlRB"
SYSTEM_PROMPT = "You are a helpful voice assistant. Be concise and conversational."

# --- STATE MANAGEMENT ---
is_speaking_event = threading.Event()
interruption_event = threading.Event()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
dg_client = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))

class DeepgramSTT:
    def __init__(self, on_transcript_callback):
        self.dg = dg_client
        self.connection = self.dg.listen.websocket.v("1")
        self.on_transcript_callback = on_transcript_callback
        self.final_transcript_buffer = []
        self.last_final_time = time.time()
        self.setup_events()
        threading.Thread(target=self.monitor_user_input, daemon=True).start()

    def setup_events(self):
        self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
        self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
        self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
        self.connection.on(LiveTranscriptionEvents.Error, self.on_error)

    def on_open(self, *args):
        print("🟢 DEEPGRAM CONNECTION OPENED.")

    def on_close(self, *args):
        print("🔴 DEEPGRAM CONNECTION CLOSED.")

    def on_error(self, _, error, **kwargs):
        print(f"❌ DEEPGRAM ERROR: {error}")

    def on_transcript(self, _, result, **kwargs):
        sentence = result.channel.alternatives[0].transcript
        is_final = result.is_final
        now = time.time()

        if sentence and is_final:
            self.final_transcript_buffer.append(sentence)
            self.last_final_time = now
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

    def start_connection(self):
        options = LiveOptions(
            model="nova-3",
            language="en-US",
            punctuate=True,
            encoding="linear16",
            sample_rate=SAMPLE_RATE,
            channels=1,
            endpointing=1000,
            vad_events=True,
        )
        try:
            if not self.connection.start(options):
                print("❌ FAILED TO START DEEPGRAM.")
                return False
            return True
        except Exception as e:
            print(f"❌ Could not start Deepgram connection: {e}")
            return False

    def send_audio_data(self, data):
        try:
            self.connection.send(data)
        except Exception as e:
            print(f"❌ Error sending audio data: {e}")

    def finish(self):
        self.connection.finish()
        print("✅ STT STREAM CLOSED.")

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

def speak_with_interruption(text):
    is_speaking_event.set()
    interruption_event.clear()
    print("🔊 Bot is speaking...")

    p = pyaudio.PyAudio()
    stream_out = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)

    try:
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
    print("🎤 Initializing Real-Time VoiceBot...")
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
                speak_with_interruption(text)
                tts_end = time.time()
                tts_time = tts_end - tts_start
                print(f"Time Taken for TTS : {tts_time:.2f} seconds")

            tts_thread = threading.Thread(target=tts_with_timing, args=(reply,), daemon=True)
            tts_thread.start()

        threading.Thread(target=run_tts_after_llm, daemon=True).start()

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
