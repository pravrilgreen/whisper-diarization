
# **Voice Transcription and Speaker Diarization**

## Description

This project leverages **Whisper**, **Pyannote**, and **Silero** technologies for converting speech to text and performing speaker diarization from audio streams provided via WebSocket. The goal is to transcribe speech in multiple languages and identify different speakers in the audio.

- **Whisper** for speech-to-text in languages like English, Vietnamese, and Japanese.
- **Pyannote** for speaker diarization and identifying multiple speakers in the audio.
- **Silero** for detecting speech activity in WebRTC-based audio streams.

## Requirements

- Python 3.8 or higher
- Libraries required:
  - `torch`
  - `numpy`
  - `sounddevice`
  - `pyannote.audio`
  - `websockets`
  - `webrtcvad`
  - `faster_whisper`
  - Additional dependencies (see installation section)

## Installation

### 1. **Clone the Repository**
Clone the repository to your local machine:

```bash
git clone https://github.com/pravrilgreen/whisper-diarization.git
cd whisper-diarization
```

### 2. **Set Up a Virtual Environment**

Create and activate a virtual environment:

- **On Windows**:
It might be because PowerShell's execution policy is blocking the script. You may need **administrator privileges** to change the policy and run the script.
```bash
python -m venv venv
.\venv\Scripts\activate
```

- **On macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. **Install Dependencies**

Install all the required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. **Set Up Environment Variables**

To securely manage the Hugging Face token (`HF_TOKEN`), set it as an environment variable.

- **On Windows (PowerShell)**:

```powershell
$env:HF_TOKEN="your-huggingface-token"
```

- **On macOS/Linux**:

```bash
export HF_TOKEN="your-huggingface-token"
```

Replace `your-huggingface-token` with your actual token from [Hugging Face](https://huggingface.co/).

### 5. **Run the Project**

Once everything is set up, run the project:

```bash
python server.py
```

---

## Features

- **Speech-to-Text**: Uses **Whisper** to transcribe audio in English, Vietnamese, and Japanese.
- **Speaker Diarization**: Uses **Pyannote** and **Silero** to identify multiple speakers in the audio.
- **Language Switching**: Supports language switching dynamically between English, Vietnamese, and Japanese.
- **Audio and Result Storage**: Automatically saves transcribed audio and diarized results.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## References

- [Whisper by OpenAI](https://huggingface.co/docs/transformers/model_doc/whisper)
- [Pyannote](https://github.com/pyannote/pyannote-audio)
- [Silero VAD](https://github.com/snakers4/silero-vad)
