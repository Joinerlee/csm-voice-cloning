import requests
import base64
import json
from pathlib import Path
import io
import argparse
import os

def encode_audio(file_path):
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def save_base64_audio(base64_str, output_path):
    """Save base64 encoded audio to a file."""
    audio_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as output_file:
        output_file.write(audio_data)

def main():
    parser = argparse.ArgumentParser(description="Client for Voice Cloning API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--context_audio", help="Path to context audio file for voice cloning")
    parser.add_argument("--context_text", default="", help="Transcription of the context audio")
    parser.add_argument("--output", default="output.wav", help="Output audio file path")
    parser.add_argument("--emotion_threshold", type=float, default=0.7, help="Emotion threshold for using CSM")
    parser.add_argument("--speaker_id", type=int, default=999, help="Speaker ID")
    parser.add_argument("--language", default="en", help="Language code (en, ko, etc.)")
    parser.add_argument("--filler_detection", type=bool, default=True, help="Enable filler word detection")
    
    # Add sample script or text file input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text to synthesize")
    group.add_argument("--script_file", help="Path to script file (one segment per line)")
    
    args = parser.parse_args()
    
    # Check health of the API
    try:
        health_response = requests.get(f"{args.url}/health")
        health_data = health_response.json()
        print(f"API Health: {health_data['status']}")
        print(f"GPU Available: {health_data['gpu_available']}")
        print(f"CSM Model Loaded: {health_data['csm_loaded']}")
    except Exception as e:
        print(f"Error checking API health: {e}")
        return
    
    # Prepare segments
    segments = []
    
    if args.text:
        # Single text segment
        segments.append({
            "text": args.text,
            "speaker_id": args.speaker_id,
            "language": args.language
        })
    else:
        # Multiple segments from file
        with open(args.script_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    segments.append({
                        "text": line,
                        "speaker_id": args.speaker_id,
                        "language": args.language
                    })
    
    # Prepare request data
    request_data = {
        "segments": segments,
        "emotion_threshold": args.emotion_threshold,
        "filler_detection": args.filler_detection
    }
    
    # Add context audio if provided
    if args.context_audio:
        if os.path.exists(args.context_audio):
            request_data["context_audio"] = encode_audio(args.context_audio)
            request_data["context_text"] = args.context_text
        else:
            print(f"Context audio file not found: {args.context_audio}")
            return
    
    # Send request to API
    try:
        print(f"Sending request to {args.url}/clone_voice with {len(segments)} segments...")
        response = requests.post(
            f"{args.url}/clone_voice", 
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            response_data = response.json()
            
            # Print processing details
            print("\nProcessing details:")
            for key, value in response_data["processing_details"].items():
                print(f"  {key}: {value}")
            
            # Save audio output
            save_base64_audio(response_data["audio"], args.output)
            print(f"\nSaved output audio to: {args.output}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error sending request: {e}")

if __name__ == "__main__":
    main() 