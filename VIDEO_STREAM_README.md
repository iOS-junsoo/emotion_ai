# 비디오 스트림 감정 분석 파이프라인

음성이 있는 비디오를 입력받아 실시간으로 감정을 분석하는 멀티모달 감정 인식 시스템입니다.

## 📋 개요

### 파이프라인 흐름

```
비디오 입력
    ↓
┌─────────────────────────────────────────┐
│  1. 표정 파트 (1초 간격)                │
│     - 프레임 캡처                       │
│     - 얼굴 감정 추출                    │
│     - 표정 버퍼에 저장                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. 음성 파트 (짧은 휴지기마다)        │
│     - VAD로 휴지기 탐지                 │
│     - 음성 감정 추출 → 음성 버퍼       │
│     - STT로 텍스트 변환 → 텍스트 누적  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. 발화 종료 (긴 휴지기 감지)         │
│     - 표정 버퍼 평균 계산               │
│     - 음성 버퍼 평균 계산               │
│     - 전체 텍스트로 텍스트 감정 추출   │
│     - 3개 감정 통합                     │
└─────────────────────────────────────────┘
    ↓
턴 결과 반환
```

## 🚀 설치

### 1. Python 패키지 설치

```bash
cd counseling-ai
pip install -r requirements.txt
```

### 2. FFmpeg 설치 (비디오/오디오 처리용)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- https://ffmpeg.org/download.html 에서 다운로드

### 3. 추가 모델 다운로드

처음 실행 시 다음 모델들이 자동으로 다운로드됩니다:
- **Whisper** (STT): `base` 모델 (~150MB)
- **DeepFace** (얼굴 감정): VGG-Face
- **SpeechBrain** (음성 감정): wav2vec2-IEMOCAP
- **KLUE-BERT** (텍스트 감정): M1NJ1/klue-bert-emotion
- **Silero VAD**: 음성 활동 감지

## 📖 사용법

### 기본 사용

```bash
python scripts/demo_video_stream.py --video <비디오_파일_경로>
```

### 예시

```bash
# 기본 설정으로 실행
python scripts/demo_video_stream.py --video test_video/sample.mp4

# 표정 캡처 간격 2초로 설정
python scripts/demo_video_stream.py --video sample.mp4 --face-interval 2.0

# 짧은 휴지기 500ms, 긴 휴지기 2000ms로 설정
python scripts/demo_video_stream.py --video sample.mp4 \
    --short-pause 500 \
    --long-pause 2000

# Whisper 모델을 small로 설정 (더 정확하지만 느림)
python scripts/demo_video_stream.py --video sample.mp4 \
    --whisper-model small

# 영어 비디오 처리
python scripts/demo_video_stream.py --video english_video.mp4 \
    --language en
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--video` | 입력 비디오 파일 경로 (필수) | - |
| `--face-interval` | 표정 캡처 간격 (초) | 1.0 |
| `--short-pause` | 짧은 휴지기 임계값 (ms) | 300 |
| `--long-pause` | 긴 휴지기/발화 종료 임계값 (ms) | 1500 |
| `--whisper-model` | Whisper 모델 크기 (tiny, base, small, medium, large) | base |
| `--language` | 언어 코드 (ko, en 등) | ko |

## 📊 출력 형식

### 턴별 결과

각 발화(턴)가 종료될 때마다 다음 정보가 출력됩니다:

```
================================================================================
📊 턴 결과 요약
================================================================================
발화 시간: 5.23초
표정 샘플: 5개
음성 청크: 3개

📝 전사된 텍스트:
  오늘 정말 힘든 하루였어요

😊 표정 감정:
  sad        65.3% ██████████████████████████████
  neutral    25.1% ████████████
  angry      9.6%  ████

🎤 음성 감정:
  sad        72.4% ██████████████████████████████
  fear       18.2% ████████
  neutral    9.4%  ████

✍️  텍스트 감정:
  sad        68.9% ██████████████████████████████
  angry      20.1% █████████
  neutral    11.0% █████

🎯 통합 감정:
  sad        68.9% ██████████████████████████████
  neutral    15.2% ███████
  angry      9.9%  ████
================================================================================
```

### 최종 요약

비디오 처리가 완료되면 전체 요약이 출력됩니다:

```
================================================================================
✅ 처리 완료!
================================================================================
총 턴 수: 3
총 발화 시간: 15.67초

📝 전체 전사 텍스트:
  오늘 정말 힘든 하루였어요 상사한테 혼났거든요 내일은 좀 나아질까요

🎯 전체 평균 감정:
  sad        62.3% ██████████████████████████████
  neutral    20.4% ██████████
  angry      12.1% ██████
================================================================================
```

## 🏗️ 아키텍처

### 주요 컴포넌트

```
ai_core/
├── emotion/
│   ├── face_deepface.py      # 얼굴 감정 분석
│   ├── audio_emotion.py      # 음성 감정 분석
│   ├── text_emotion.py       # 텍스트 감정 분석
│   └── audio_vad.py          # 음성 활동 감지 (짧은/긴 휴지기)
├── stt/
│   └── whisper_stt.py        # Speech-to-Text
└── service/
    └── video_stream_processor.py  # 비디오 스트림 처리 오케스트레이터
```

### VideoStreamProcessor 클래스

핵심 메서드:

- `process_frame()`: 1초마다 프레임에서 얼굴 감정 추출
- `process_audio_chunk()`: 오디오 청크마다 VAD로 휴지기 탐지
- `extract_audio_emotion()`: 짧은 휴지기 시 음성 감정 추출
- `transcribe_audio()`: 짧은 휴지기 시 STT로 텍스트 변환
- `finalize_turn()`: 긴 휴지기 시 3개 감정 통합 및 결과 반환

## ⚙️ 파라미터 튜닝 가이드

### 표정 캡처 간격 (`--face-interval`)

- **1.0초 (기본값)**: 일반적인 상황에 적합
- **0.5초**: 표정 변화가 빠른 경우
- **2.0초**: 계산 리소스 절약

### 휴지기 임계값

#### 짧은 휴지기 (`--short-pause`)

- **300ms (기본값)**: 자연스러운 문장 내 휴지
- **500ms**: 더 긴 문장 단위로 분석
- **200ms**: 세밀한 분석 (청크 많아짐)

#### 긴 휴지기 (`--long-pause`)

- **1500ms (기본값)**: 일반적인 발화 종료
- **2000ms**: 여유로운 발화 스타일
- **1000ms**: 빠른 대화

### Whisper 모델 크기

| 모델 | 크기 | 속도 | 정확도 | 권장 용도 |
|------|------|------|--------|-----------|
| tiny | ~40MB | 매우 빠름 | 낮음 | 테스트 |
| base | ~150MB | 빠름 | 보통 | 일반 사용 (기본값) |
| small | ~500MB | 보통 | 높음 | 높은 정확도 필요 시 |
| medium | ~1.5GB | 느림 | 매우 높음 | 전문적 용도 |
| large | ~3GB | 매우 느림 | 최고 | 최고 품질 |

## 🔧 프로그래밍 API

Python 코드에서 직접 사용:

```python
from ai_core.service.video_stream_processor import VideoStreamProcessor

# 프로세서 생성
processor = VideoStreamProcessor(
    face_capture_interval=1.0,
    short_pause_ms=300,
    long_pause_ms=1500,
    whisper_model="base",
    language="ko"
)

# 모델 워밍업
processor.warmup()

# 콜백 함수 정의
def on_turn_complete(result):
    print(f"텍스트: {result.transcribed_text}")
    print(f"감정: {result.aggregated_emotion}")

# 비디오 처리
results = processor.process_video_stream(
    "path/to/video.mp4",
    callback=on_turn_complete
)

# 결과 사용
for i, result in enumerate(results):
    print(f"턴 {i+1}: {result.transcribed_text}")
    print(f"  통합 감정: {result.aggregated_emotion}")
```

## 📝 제한사항

1. **실시간 웹캠 미지원**: 현재는 비디오 파일만 지원 (실시간 스트림은 추후 추가 예정)
2. **단일 화자**: 여러 사람이 동시에 말하는 경우 정확도 저하
3. **얼굴 감지**: 얼굴이 안 보이면 표정 감정은 neutral로 처리
4. **GPU 권장**: CPU만으로도 동작하지만 GPU 사용 시 훨씬 빠름

## 🐛 문제 해결

### FFmpeg 오류

```bash
# FFmpeg 설치 확인
ffmpeg -version

# 없으면 설치
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### CUDA/GPU 오류

DeepFace가 GPU 사용 시 오류가 나면 환경변수 설정:

```bash
export CUDA_VISIBLE_DEVICES=''  # CPU만 사용
python scripts/demo_video_stream.py --video sample.mp4
```

### 메모리 부족

- Whisper 모델을 `tiny`나 `base`로 줄이기
- 표정 캡처 간격을 늘리기 (예: `--face-interval 2.0`)

## 📚 관련 문서

- [전체 프로젝트 README](./README.md)
- [How to use](./How%20to%20use.md)
- [Repository 설명](./explain_repo.md)
