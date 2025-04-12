#!/bin/bash

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "You can set it with: export HF_TOKEN=your_hugging_face_token"
fi

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected."
    nvidia-smi
else
    echo "Warning: NVIDIA GPU not detected. Voice cloning will not work without GPU."
fi

# Create output directory if it doesn't exist
mkdir -p output

# Start the FastAPI server
echo "Starting FastAPI voice cloning server..."
uvicorn app:app --host 0.0.0.0 --port 8000 