# Voice Cloning with CSM-1B and FastAPI

This repository contains tools to clone voices using the Sesame CSM-1B model with FastAPI integration for selective voice cloning based on emotion analysis.

## Features

- **Selective Voice Cloning**: Uses BERT-based emotion analysis to clone only high-emotion segments
- **Filler Word Detection**: Automatically detects and clones filler words for natural speech
- **Hybrid TTS Strategy**: Uses Google TTS for low-emotion segments to speed up processing
- **GPU Acceleration**: Utilizes CUDA for fast inference
- **Docker Support**: Easy deployment with Docker and docker-compose
- **API Integration**: FastAPI-based REST API for easy integration with your applications

## Architecture

The system works as follows:
1. User sends a script to the GPU server
2. Text is analyzed for emotion and filler words
3. High-emotion segments and fillers are voice-cloned using CSM-1B
4. Low-emotion segments are quickly synthesized with Google TTS
5. All segments are combined and returned as a single audio file

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU
- Hugging Face account with access to the CSM-1B model
- Hugging Face API token
- Docker and docker-compose (for containerized deployment)

## Installation

### Docker Installation (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/your-username/csm-voice-cloning.git
cd csm-voice-cloning
```

2. Set your Hugging Face token:
```bash
echo "HF_TOKEN=your_hugging_face_token" > .env
```

3. Build and start the Docker container:
```bash
docker-compose up -d
```

### Manual Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/csm-voice-cloning.git
cd csm-voice-cloning
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Hugging Face token:
```bash
export HF_TOKEN="your_hugging_face_token"
```

4. Start the FastAPI server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Usage

The API provides a `/clone_voice` endpoint that accepts a JSON payload with the following structure:

```json
{
  "segments": [
    {
      "text": "This is a text segment to be synthesized.",
      "speaker_id": 999,
      "language": "en"
    }
  ],
  "context_audio": "base64_encoded_audio_file",
  "context_text": "Transcription of the context audio",
  "emotion_threshold": 0.7,
  "filler_detection": true
}
```

### Client Example

A client example is provided in `client_example.py`:

```bash
python client_example.py --text "Hello, this is a test of the voice cloning system." --context_audio sample.mp3
```

For a script file with multiple segments:

```bash
python client_example.py --script_file script.txt --context_audio sample.mp3 --emotion_threshold 0.6
```

## Accepting the Model on Hugging Face

Before using the model, you need to accept the terms on Hugging Face:

1. Visit the [Sesame CSM-1B model page](https://huggingface.co/sesame/csm-1b)
2. Click on "Access repository" and accept the terms
3. Make sure you're logged in with the same account that your HF_TOKEN belongs to

## Troubleshooting

- **CUDA out of memory**: Try reducing the batch size or use a GPU with more memory
- **Model download issues**: Ensure you've accepted the model terms on Hugging Face and your token is correct
- **API connection errors**: Verify that the server is running and accessible

## License

This project uses the Sesame CSM-1B model, which is subject to its own license terms. Please refer to the [model page](https://huggingface.co/sesame/csm-1b) for details. 