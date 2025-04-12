from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import torch
import torchaudio
import numpy as np
import base64
import os
import io
import json
from pathlib import Path
import tempfile
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gtts import gTTS
from huggingface_hub import hf_hub_download

# Import CSM voice cloning modules
from generator import load_csm_1b, Segment
from filler_words import detect_fillers, split_text_with_fillers

# Set up environment variables
os.environ["HF_TOKEN"] = ""  # Set your Hugging Face token here

app = FastAPI(title="Voice Cloning API with Emotion Analysis", 
              description="API for cloning voices using CSM-1B model with emotion-based selective processing")

# Pydantic models for request and response
class ScriptSegment(BaseModel):
    text: str
    speaker_id: int = 999
    language: str = "en"
    
class CloneRequest(BaseModel):
    segments: List[ScriptSegment]
    context_audio: Optional[str] = None  # Base64 encoded audio file
    context_text: Optional[str] = ""
    emotion_threshold: float = 0.7  # Threshold for emotion score to use voice cloning
    filler_detection: bool = True  # Whether to detect and clone filler words
    
class CloneResponse(BaseModel):
    audio: str  # Base64 encoded audio
    processing_details: Dict[str, Union[str, float]]

# Initialize models
@app.on_event("startup")
async def startup_event():
    global emotion_model, emotion_tokenizer, csm_generator
    
    # Load emotion analysis model
    emotion_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emotion_model = emotion_model.to(device)
    
    # Load CSM model for voice cloning
    if device == "cuda":
        try:
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            csm_generator = load_csm_1b(model_path, device)
            print("CSM model loaded successfully!")
        except Exception as e:
            print(f"Error loading CSM model: {e}")
            csm_generator = None
    else:
        print("CUDA not available, voice cloning disabled")
        csm_generator = None

# Function to analyze emotion score (1-5 scale)
def analyze_emotion(text: str) -> float:
    device = emotion_model.device
    
    # Tokenize and prepare input
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        
    # Return normalized score (0-1 range from original 1-5 scale)
    weighted_score = 0
    for i, score in enumerate(scores[0]):
        weighted_score += (i + 1) * score.item()
    
    # Normalize to 0-1 range
    normalized_score = (weighted_score - 1) / 4
    return normalized_score

# Function to process audio with silence removal
def remove_silence(audio, threshold=0.01, min_silence_duration=0.2, sample_rate=24000):
    # Convert to numpy for easier processing
    audio_np = audio.cpu().numpy()
    
    # Calculate energy
    energy = np.abs(audio_np)
    
    # Find regions above threshold (speech)
    is_speech = energy > threshold
    
    # Convert min_silence_duration to samples
    min_silence_samples = int(min_silence_duration * sample_rate)
    
    # Find speech segments
    speech_segments = []
    in_speech = False
    speech_start = 0
    
    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            # Start of speech segment
            in_speech = True
            speech_start = i
        elif not is_speech[i] and in_speech:
            # Potential end of speech segment
            # Only end if silence is long enough
            silence_count = 0
            for j in range(i, min(len(is_speech), i + min_silence_samples)):
                if not is_speech[j]:
                    silence_count += 1
                else:
                    break
            
            if silence_count >= min_silence_samples:
                # End of speech segment
                in_speech = False
                speech_segments.append((speech_start, i))
    
    # Handle case where audio ends during speech
    if in_speech:
        speech_segments.append((speech_start, len(is_speech)))
    
    # Concatenate speech segments
    if not speech_segments:
        return audio  # Return original if no speech found
    
    # Add small buffer around segments
    buffer_samples = int(0.05 * sample_rate)  # 50ms buffer
    processed_segments = []
    
    for start, end in speech_segments:
        buffered_start = max(0, start - buffer_samples)
        buffered_end = min(len(audio_np), end + buffer_samples)
        processed_segments.append(audio_np[buffered_start:buffered_end])
    
    # Concatenate all segments
    processed_audio = np.concatenate(processed_segments)
    
    return torch.tensor(processed_audio, device=audio.device)

# Function to generate audio using GTTS
def generate_gtts_audio(text: str, lang: str = 'en') -> torch.Tensor:
    # Create gTTS object and save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(temp_file.name)
        temp_file.close()
        
        # Load the audio file and convert to torch tensor
        audio, sample_rate = torchaudio.load(temp_file.name)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample to 24kHz (CSM model's sample rate)
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=24000)
        
        return audio.squeeze(0)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

# Process a single segment with filler word handling
def process_segment(
    segment: ScriptSegment, 
    context_segment: Optional[Segment], 
    emotion_threshold: float,
    filler_detection: bool
) -> List[tuple]:
    """
    Process a single segment, handling filler words and emotion analysis.
    
    Returns:
        List[tuple]: List of (subsegment_id, audio) tuples
    """
    subsegment_results = []
    processing_details = {}
    
    # Split text if filler detection is enabled
    if filler_detection:
        subsegments = split_text_with_fillers(segment.text, segment.language)
    else:
        subsegments = [{"text": segment.text, "should_clone": False}]
    
    # Process each subsegment
    for i, subsegment in enumerate(subsegments):
        subsegment_id = f"subseg_{i}"
        text = subsegment["text"]
        
        # For filler words, always use voice cloning
        if subsegment.get("should_clone", False):
            processing_details[f"{subsegment_id}_method"] = "csm_filler"
            try:
                audio = csm_generator.generate(
                    text=text,
                    speaker=segment.speaker_id,
                    context=[context_segment] if context_segment else [],
                    max_audio_length_ms=5_000,  # Shorter for fillers
                    temperature=0.6,
                    topk=20,
                )
                subsegment_results.append((subsegment_id, audio))
            except Exception as e:
                # Fallback to GTTS if cloning fails
                processing_details[f"{subsegment_id}_error"] = str(e)
                processing_details[f"{subsegment_id}_method"] = "gtts_fallback"
                audio = generate_gtts_audio(text, segment.language)
                subsegment_results.append((subsegment_id, audio))
        else:
            # For non-fillers, use emotion analysis
            emotion_score = analyze_emotion(text)
            processing_details[f"{subsegment_id}_emotion_score"] = emotion_score
            
            if emotion_score >= emotion_threshold:
                # Use CSM voice cloning for high emotion
                processing_details[f"{subsegment_id}_method"] = "csm_emotion"
                try:
                    audio = csm_generator.generate(
                        text=text,
                        speaker=segment.speaker_id,
                        context=[context_segment] if context_segment else [],
                        max_audio_length_ms=15_000,
                        temperature=0.6,
                        topk=20,
                    )
                    subsegment_results.append((subsegment_id, audio))
                except Exception as e:
                    # Fallback to GTTS if cloning fails
                    processing_details[f"{subsegment_id}_error"] = str(e)
                    processing_details[f"{subsegment_id}_method"] = "gtts_fallback"
                    audio = generate_gtts_audio(text, segment.language)
                    subsegment_results.append((subsegment_id, audio))
            else:
                # Use GTTS for low emotion segments
                processing_details[f"{subsegment_id}_method"] = "gtts"
                audio = generate_gtts_audio(text, segment.language)
                subsegment_results.append((subsegment_id, audio))
    
    return subsegment_results, processing_details

@app.post("/clone_voice", response_model=CloneResponse)
async def clone_voice(request: CloneRequest, background_tasks: BackgroundTasks):
    if csm_generator is None:
        raise HTTPException(status_code=503, detail="Voice cloning service unavailable (GPU required)")
    
    # Process context audio if provided
    context_segment = None
    if request.context_audio:
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(request.context_audio)
            audio_io = io.BytesIO(audio_bytes)
            
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Load audio file
            context_audio, sr = torchaudio.load(temp_file_path)
            context_audio = context_audio.mean(dim=0)  # Convert to mono
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Resample if needed
            if sr != csm_generator.sample_rate:
                context_audio = torchaudio.functional.resample(
                    context_audio, orig_freq=sr, new_freq=csm_generator.sample_rate
                )
            
            # Normalize audio
            context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
            
            # Process with silence removal
            context_audio = remove_silence(
                context_audio, 
                threshold=0.015, 
                min_silence_duration=0.15, 
                sample_rate=csm_generator.sample_rate
            )
            
            # Create context segment
            context_segment = Segment(
                text=request.context_text,
                speaker=request.segments[0].speaker_id if request.segments else 999,
                audio=context_audio
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing context audio: {str(e)}")
    
    # Process each segment
    all_results = []
    all_processing_details = {}
    
    for i, segment in enumerate(request.segments):
        segment_id = f"segment_{i}"
        
        # Process the segment
        subsegment_results, processing_details = process_segment(
            segment, 
            context_segment, 
            request.emotion_threshold,
            request.filler_detection
        )
        
        # Add processing details with segment prefix
        for key, value in processing_details.items():
            all_processing_details[f"{segment_id}_{key}"] = value
        
        # Add segment results
        all_results.extend(subsegment_results)
    
    # Concatenate all audio segments in order
    combined_audio = torch.cat([audio for _, audio in all_results])
    
    # Save combined audio to a temporary file
    temp_output_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        temp_output_path = temp_output_file.name
        temp_output_file.close()
        
        # Save combined audio
        torchaudio.save(
            temp_output_path, 
            combined_audio.unsqueeze(0).cpu(), 
            csm_generator.sample_rate
        )
        
        # Read the file and encode to base64
        with open(temp_output_path, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
    finally:
        # Clean up temporary file
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
    
    # Return combined audio in base64 format along with processing details
    return CloneResponse(
        audio=encoded_audio,
        processing_details=all_processing_details
    )

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "gpu_available": torch.cuda.is_available(),
        "csm_loaded": csm_generator is not None
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 