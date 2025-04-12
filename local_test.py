"""
로컬 테스트를 위한 스크립트
"""
import os
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from generator import load_csm_1b, Segment
from filler_words import detect_fillers, split_text_with_fillers
from gtts import gTTS
import tempfile
import numpy as np
import re

# 감정 분석 모델 로드
def load_emotion_model():
    print("감정 분석 모델 로딩 중...")
    emotion_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
    
    # GPU 사용 가능한 경우
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, tokenizer, device

# 감정 점수 분석 함수
def analyze_emotion(text, model, tokenizer, device):
    print(f"감정 분석 중: {text}")
    
    # 토크나이징 및 입력 준비
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1)
        
    # 가중치 평균 계산 (1-5 스케일)
    weighted_score = 0
    for i, score in enumerate(scores[0]):
        weighted_score += (i + 1) * score.item()
    
    # 0-1 범위로 정규화
    normalized_score = (weighted_score - 1) / 4
    return normalized_score

# 무음 제거 함수
def remove_silence(audio, threshold=0.01, min_silence_duration=0.2, sample_rate=24000):
    # numpy로 변환
    audio_np = audio.cpu().numpy()
    
    # 에너지 계산
    energy = np.abs(audio_np)
    
    # 기준치 이상인 부분 찾기
    is_speech = energy > threshold
    
    # 무음 길이 샘플 수 계산
    min_silence_samples = int(min_silence_duration * sample_rate)
    
    # 음성 세그먼트 찾기
    speech_segments = []
    in_speech = False
    speech_start = 0
    
    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            # 음성 세그먼트 시작
            in_speech = True
            speech_start = i
        elif not is_speech[i] and in_speech:
            # 음성 세그먼트 끝 가능성
            silence_count = 0
            for j in range(i, min(len(is_speech), i + min_silence_samples)):
                if not is_speech[j]:
                    silence_count += 1
                else:
                    break
            
            if silence_count >= min_silence_samples:
                # 음성 세그먼트 끝
                in_speech = False
                speech_segments.append((speech_start, i))
    
    # 마지막 부분이 음성인 경우
    if in_speech:
        speech_segments.append((speech_start, len(is_speech)))
    
    # 음성 세그먼트가 없으면 원래 오디오 반환
    if not speech_segments:
        return audio
    
    # 세그먼트 주변에 버퍼 추가
    buffer_samples = int(0.05 * sample_rate)  # 50ms 버퍼
    processed_segments = []
    
    for start, end in speech_segments:
        buffered_start = max(0, start - buffer_samples)
        buffered_end = min(len(audio_np), end + buffer_samples)
        processed_segments.append(audio_np[buffered_start:buffered_end])
    
    # 모든 세그먼트 연결
    processed_audio = np.concatenate(processed_segments)
    
    return torch.tensor(processed_audio, device=audio.device)

# GTTS로 오디오 생성
def generate_gtts_audio(text, lang='en'):
    print(f"GTTS로 오디오 생성 중: {text}")
    
    # 빈 텍스트나 구두점만 있는 경우 처리
    if not text or text.strip() == "" or re.match(r'^[,.!?;:]+$', text.strip()):
        print(f"  경고: 빈 텍스트 또는 구두점만 있음, 기본 음성 생성")
        # 무음 또는 기본 텍스트로 대체
        text = "silence"  # 기본 텍스트로 대체
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(temp_file.name)
        temp_file.close()
        
        # 오디오 파일 로드 및 변환
        audio, sample_rate = torchaudio.load(temp_file.name)
        
        # 모노로 변환
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # 24kHz로 리샘플링
        if sample_rate != 24000:
            audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=24000)
        
        return audio.squeeze(0)
    finally:
        # 임시 파일 정리
        os.unlink(temp_file.name)

def main():
    # 환경 변수 설정
    os.environ["HF_TOKEN"] = ""  # 여기에 Hugging Face 토큰 설정
    
    # 모델 로드
    emotion_model, emotion_tokenizer, device = load_emotion_model()
    
    # CSM 모델 로드
    print("CSM 모델 로드 중...")
    if device == "cuda":
        try:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
            csm_generator = load_csm_1b(model_path, device)
            print("CSM 모델 로드 성공!")
        except Exception as e:
            print(f"CSM 모델 로드 오류: {e}")
            csm_generator = None
    else:
        print("CUDA를 사용할 수 없습니다. 음성 클로닝 기능 비활성화.")
        csm_generator = None
    
    if csm_generator is None:
        print("CSM 모델 로드에 실패했습니다. GPU가 필요합니다.")
        return
    
    # 컨텍스트 오디오 처리
    context_audio_path = input("컨텍스트 오디오 파일 경로 (GTTS만 사용시 비워두기): ").strip()
    context_segment = None
    
    if context_audio_path:
        try:
            print(f"컨텍스트 오디오 로드 중: {context_audio_path}")
            context_audio, sr = torchaudio.load(context_audio_path)
            context_audio = context_audio.mean(dim=0)  # 모노로 변환
            
            # 리샘플링
            if sr != csm_generator.sample_rate:
                context_audio = torchaudio.functional.resample(
                    context_audio, orig_freq=sr, new_freq=csm_generator.sample_rate
                )
            
            # 오디오 정규화
            context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
            
            # 무음 제거
            context_audio = remove_silence(
                context_audio, 
                threshold=0.015, 
                min_silence_duration=0.15, 
                sample_rate=csm_generator.sample_rate
            )
            
            # 컨텍스트 텍스트 (선택적)
            context_text = input("컨텍스트 오디오 텍스트 (선택사항): ").strip()
            
            # 컨텍스트 세그먼트 생성
            context_segment = Segment(
                text=context_text,
                speaker=999,
                audio=context_audio
            )
            print(f"컨텍스트 오디오 길이: {len(context_audio) / csm_generator.sample_rate:.2f}초")
        except Exception as e:
            print(f"컨텍스트 오디오 처리 중 오류: {e}")
            context_segment = None
    
    # 감정 임계값 설정
    try:
        emotion_threshold = float(input("감정 임계값 (0.0~1.0, 기본값 0.7): ") or "0.7")
    except ValueError:
        emotion_threshold = 0.7
    
    # 필러 감지 여부
    filler_detection = input("필러 감지 사용? (y/n, 기본값 y): ").lower() != 'n'
    
    # 텍스트 입력 방법 선택
    input_method = input("입력 방법 선택 (1: 텍스트 직접 입력, 2: 파일에서 입력): ")
    
    segments = []
    
    if input_method == "1":
        print("텍스트를 입력하세요 (빈 줄을 입력하여 종료):")
        while True:
            line = input().strip()
            if not line:
                break
            segments.append(line)
    elif input_method == "2":
        script_file = input("스크립트 파일 경로: ")
        try:
            with open(script_file, "r", encoding="utf-8") as f:
                segments = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"파일 읽기 오류: {e}")
            return
    else:
        print("잘못된 입력 방법입니다.")
        return
    
    if not segments:
        print("텍스트가 없습니다.")
        return
    
    print(f"{len(segments)}개 세그먼트 처리 시작...")
    
    # 출력 디렉토리 생성
    os.makedirs("output", exist_ok=True)
    
    # 각 세그먼트 처리
    all_results = []
    
    for i, text in enumerate(segments):
        print(f"\n세그먼트 {i+1}/{len(segments)}: {text}")
        
        # 필러 감지 및 분할
        if filler_detection:
            subsegments = split_text_with_fillers(text)
        else:
            subsegments = [{"text": text, "should_clone": False}]
        
        # 각 서브세그먼트 처리
        for j, subsegment in enumerate(subsegments):
            subtext = subsegment["text"]
            print(f"  서브세그먼트 {j+1}/{len(subsegments)}: {subtext}")
            
            # 필러는 항상 클로닝
            if subsegment.get("should_clone", False):
                print(f"  필러 감지됨: {subtext}, CSM 사용")
                try:
                    audio = csm_generator.generate(
                        text=subtext,
                        speaker=999,
                        context=[context_segment] if context_segment else [],
                        max_audio_length_ms=5_000,
                        temperature=0.6,
                        topk=20,
                    )
                    all_results.append(audio)
                except Exception as e:
                    print(f"  CSM 생성 오류, GTTS로 대체: {e}")
                    audio = generate_gtts_audio(subtext)
                    all_results.append(audio)
            else:
                # 감정 분석
                emotion_score = analyze_emotion(subtext, emotion_model, emotion_tokenizer, device)
                print(f"  감정 점수: {emotion_score:.4f} (임계값: {emotion_threshold})")
                
                if emotion_score >= emotion_threshold:
                    # 높은 감정 세그먼트는 CSM으로 클로닝
                    print(f"  높은 감정 점수, CSM 사용")
                    try:
                        audio = csm_generator.generate(
                            text=subtext,
                            speaker=999,
                            context=[context_segment] if context_segment else [],
                            max_audio_length_ms=15_000,
                            temperature=0.6,
                            topk=20,
                        )
                        all_results.append(audio)
                    except Exception as e:
                        print(f"  CSM 생성 오류, GTTS로 대체: {e}")
                        audio = generate_gtts_audio(subtext)
                        all_results.append(audio)
                else:
                    # 낮은 감정 세그먼트는 GTTS 사용
                    print(f"  낮은 감정 점수, GTTS 사용")
                    audio = generate_gtts_audio(subtext)
                    all_results.append(audio)
    
    # 모든 오디오 세그먼트 결합
    print("\n모든 세그먼트 결합 중...")
    combined_audio = torch.cat(all_results)
    
    # 결합된 오디오 저장
    output_path = "output/combined_output.wav"
    torchaudio.save(
        output_path, 
        combined_audio.unsqueeze(0).cpu(), 
        csm_generator.sample_rate
    )
    
    print(f"\n완료! 결과 저장됨: {output_path}")
    print(f"총 오디오 길이: {len(combined_audio) / csm_generator.sample_rate:.2f}초")

if __name__ == "__main__":
    main() 