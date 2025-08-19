# Voice_Bot: Real-Time Speech-to-Speech (STS) Conversational AI

This project is an **in-development real-time Speech-to-Speech (STS) VoiceBot** that listens to your voice, transcribes it using Google Cloud STT, generates a natural-sounding response using OpenAI's GPT models, and streams the reply aloud via ElevenLabs' text-to-speech. The pipeline was reworked from Deepgram-based recognition to use Google STT for improved flexibility and wider platform compatibility.

***

## Features

- **Real-time microphone input**
- **Streaming speech-to-text with Google Cloud STT**
- **Conversational AI using OpenAI's GPT-4 models**
- **Streaming text-to-speech with ElevenLabs** 
- **Multi-threaded pipeline for low-latency, interruption-aware interactions**
- **User interruption support:** Bot response playback can be interrupted live via user speech
- **Automatic environmental variable loading** for API keys and credentials (.env support)
- **Observable latency reporting** at each main stage for transparency and tuning

***

## Requirements

- **Python 3.8+**
- **Google Cloud account** with a Speech-to-Text service enabled
    - Service account JSON credentials (set `$GOOGLE_APPLICATION_CREDENTIALS` or specify in `.env`)
- **OpenAI API key** (for GPT models)
- **ElevenLabs API key** (for TTS streaming)
- **ffmpeg** (`ffplay` optional, used in some configurations for audio output)
- **Windows, Mac, or Linux** (tested on Windows, cross-platform with minor tweaks)

***

## Python Dependencies

_Typical Requirements:_

- `sounddevice` (real-time microphone input)
- `numpy`
- `python-dotenv`
- `openai`
- `elevenlabs`
- `pyaudio` (low-level audio stream playback)
- `google-cloud-speech` (Google STT)

***

## Setup

1. **Clone this repository and enter the project folder.**
2. **Set up your environment variables:**
    - Create a `.env` file in the project root with the following keys:

```
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
GOOGLE_APPLICATION_CREDENTIALS=absolute_path_to_your_google_json_credentials_file
```

    - Make sure your Google credentials JSON file is accessible.
3. **Ensure ffmpeg is available:** (for other playback, optional in current version)
4. **Run your bot:**

```
python sts_GoogleSTT.py
```


***

## Usage

- **Speak into your microphone.**
- The bot transcribes your speech, generates a response, and reads it aloud with ElevenLabs.
- **Interrupt the bot’s speech** by speaking over it, your new speech input will preempt playback, demonstrating live interruption support.
- **Press Enter** at any time in the terminal to stop the bot.

***

## Notes

- **Feedback Loop:** Use headphones to avoid the bot picking up its own speech.
- **Latency:** Although multithreaded for responsiveness, latency depends on your network, device, and API response speeds.
- **Interruptions:** Speak during bot output to interrupt and trigger a new turn immediately.
- **Environment:** Tested on Windows (see code for audio backends); should work cross-platform with minimal changes.
- **Development Status:** This project is a work-in-progress—expect code changes, feature tweaks, and possible breaking changes as improvements roll out.

***

## Troubleshooting

- **Google Cloud STT errors:** Check your credentials path in `.env` and ensure Google Cloud project/service access.
- **No audio output:** Make sure `pyaudio` is installed, your audio output device is configured, and you’re not running via SSH/headless.
- **Bot repeats itself:** Use headphones. Double-check microphone/speaker settings.
- **Slow responses:** Network/API latency may vary; check your internet speed and API quota usage.

***

