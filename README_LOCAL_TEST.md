# CSM-1B를 이용한 감정 기반 음성 클로닝 로컬 테스트 가이드

이 문서는 CSM-1B 모델을 사용한 감정 기반 음성 클로닝 시스템의 로컬 테스트 방법을 설명합니다.

## 주요 기능

- **감정 분석 기반 선택적 음성 클로닝**: BERT 기반 감정 분석을 통해 감정 수치가 높은 부분만 음성 클로닝
- **필러 단어 감지**: 필러 단어(um, uh, well 등)를 자동으로 감지하여 클로닝
- **하이브리드 TTS 방식**: 감정이 낮은 부분은 Google TTS를 사용하여 처리 속도 향상
- **GPU 가속**: CUDA를 활용한 빠른 처리

## 준비 사항

- Python 3.10 이상
- CUDA 호환 GPU
- Hugging Face 계정 및 CSM-1B 모델 접근 권한
- Hugging Face API 토큰

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install torch==2.4.0 torchaudio==2.4.0 tokenizers==0.21.0 transformers==4.49.0 huggingface_hub==0.28.1 moshi==0.2.2 torchtune==0.4.0 torchao==0.9.0 numpy gtts
pip install silentcipher @ git+https://github.com/SesameAILabs/silentcipher@master
```

2. Hugging Face에서 CSM-1B 모델 사용 조건 동의:
   - [Sesame CSM-1B 모델 페이지](https://huggingface.co/sesame/csm-1b) 방문
   - "Access repository" 클릭 후 약관 동의
   - HF_TOKEN은 같은 계정의 토큰을 사용해야 함

## 로컬 테스트 방법

1. `local_test.py` 파일에서 Hugging Face 토큰 설정:
```python
os.environ["HF_TOKEN"] = "여기에_토큰_입력"  
```

2. 테스트 스크립트 실행:
```bash
python local_test.py
```

3. 대화형 설정 진행:
   - 컨텍스트 오디오 파일 경로: 목소리 클로닝에 사용할 오디오 파일 경로 입력 (mp3 또는 wav)
   - 컨텍스트 오디오 텍스트: 오디오 파일의 내용을 텍스트로 입력 (선택 사항)
   - 감정 임계값: 클로닝을 적용할 감정 점수 기준값 (0.0~1.0, 기본값 0.7)
   - 필러 감지 사용 여부: 필러 단어를 자동으로 감지할지 여부
   - 입력 방법: 텍스트 직접 입력 또는 파일에서 가져오기

4. 결과 확인:
   - 처리된 오디오 파일은 `output/combined_output.wav`에 저장됨
   - 각 세그먼트의 감정 점수와 처리 방법이 콘솔에 출력됨

## 사용 예시

```
컨텍스트 오디오 파일 경로 (GTTS만 사용시 비워두기): sample.mp3
컨텍스트 오디오 텍스트 (선택사항): 이것은 샘플 음성입니다
감정 임계값 (0.0~1.0, 기본값 0.7): 0.65
필러 감지 사용? (y/n, 기본값 y): y
입력 방법 선택 (1: 텍스트 직접 입력, 2: 파일에서 입력): 2
스크립트 파일 경로: example_script.txt
```

## 테스트 스크립트 작성 팁

* 감정이 높은 부분에는 !, ? 등의 문장 부호를 사용하고 강한 감정 표현을 포함하세요.
* 필러 단어(um, uh, well 등)를 포함하면 자동으로 감지되어 클로닝됩니다.
* 각 줄은 별도의 세그먼트로 처리됩니다.
* 짧은 문장으로 나누는 것이 처리 품질과 속도 면에서 좋습니다.

## 문제 해결

- **GPU 메모리 부족**: 세그먼트를 더 짧게 분할하거나 배치 크기 줄이기
- **모델 다운로드 문제**: Hugging Face에서 모델 약관에 동의했는지 확인하고 토큰이 올바른지 확인
- **처리 오류**: 텍스트가 너무 길면 짧게 나누어 처리 