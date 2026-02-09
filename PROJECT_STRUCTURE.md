# 🤖 AI 감정 상담 시스템 - 프로젝트 구조

> **업데이트**: 2025-01-28  
> **버전**: 1.0

---

## 📖 목차

- [프로젝트 개요](#프로젝트-개요)
- [디렉토리 구조](#디렉토리-구조)
- [핵심 모듈 설명](#핵심-모듈-설명)
- [실행 스크립트](#실행-스크립트)
- [테스트 데이터](#테스트-데이터)
- [문서 파일](#문서-파일)
- [빠른 시작](#빠른-시작)

---

## 🎯 프로젝트 개요

멀티모달 감정 분석(얼굴, 음성, 텍스트)을 활용한 AI 기반 감정 상담 시스템입니다.

### **주요 기능**
- 🎥 실시간 비디오 기반 감정 분석
- 🗣️ 음성 인식 및 감정 추출
- 💬 감정 기반 대화형 상담
- 🎯 감정별 맞춤 응답 생성

---

## 📂 디렉토리 구조

```
counseling-ai/
├── ai_core/              # 핵심 AI 모델 및 서비스 로직
├── scripts/              # 실행 가능한 데모 및 테스트 스크립트
├── test_video/           # 테스트용 비디오 파일
├── test_audio/           # 테스트용 오디오 파일
├── test_image/           # 테스트용 이미지 파일
├── training/             # 모델 파인튜닝 스크립트
├── contracts/            # API 계약 정의 (gRPC, OpenAPI)
├── docker/               # Docker 컨테이너 설정
├── data/                 # 샘플 데이터
├── models/               # 다운로드된 모델 심볼릭 링크
└── *.md                  # 프로젝트 문서들
```

---

## 🧠 핵심 모듈 설명

### **ai_core/** - 핵심 AI 로직

#### **ai_core/mock_models.py** - Mock 모델 (GPU 없이 테스트)
| 클래스 | 설명 | 용도 |
|------|------|------|
| `MockLLMClient` | Mock LLM 클라이언트 | GPU 없이 API 구조 테스트 |
| `MockFaceEmotionEstimator` | Mock 얼굴 감정 분석 | 랜덤 감정 반환 |
| `MockAudioEmotionEstimator` | Mock 음성 감정 분석 | 랜덤 감정 반환 |
| `MockTextEmotionEstimator` | Mock 텍스트 감정 분석 | 키워드 기반 감정 |
| `MockWhisperSTT` | Mock STT | 미리 정의된 더미 텍스트 |

**활성화**: 환경 변수 `MOCK_MODE=true` 설정  
**장점**: GPU 불필요, 즉시 실행, 실제 API와 동일한 인터페이스

---

#### **ai_core/emotion/** - 감정 분석 모델
| 파일 | 설명 | 사용 모델 |
|------|------|-----------|
| `face_deepface.py` | 얼굴 표정 기반 감정 분석 | DeepFace (VGG-Face) |
| `audio_emotion.py` | 음성 톤/억양 기반 감정 분석 | SpeechBrain Wav2Vec2-IEMOCAP |
| `text_emotion.py` | 텍스트 기반 감정 분석 | KLUE-BERT (M1NJ1/klue-bert-emotion) |
| `audio_vad.py` | 음성 활동 감지 (Voice Activity Detection) | Silero VAD |
| `smoothing.py` | 감정 평활화 유틸리티 | - |

**출력**: 7가지 감정 확률 분포 (neutral, happy, sad, angry, fear, surprise, disgust)

---

#### **ai_core/llm/** - 대화형 언어 모델
| 파일 | 설명 | 상태 |
|------|------|------|
| `transformers_client.py` | Hugging Face Transformers LLM 클라이언트 | ✅ 현재 사용 |
| `vllm_client.py` | vLLM 고성능 추론 서버 클라이언트 | 🟡 선택적 |
| `qwen_client.py` | Qwen 전용 클라이언트 | 🟡 레거시 |
| `conversation_history.py` | 대화 히스토리 관리 | ✅ 필수 |

**기본 모델**: Qwen/Qwen2-7B-Instruct (70억 파라미터)

---

#### **ai_core/stt/** - 음성-텍스트 변환
| 파일 | 설명 | 사용 모델 |
|------|------|-----------|
| `whisper_stt.py` | 음성을 텍스트로 변환 | OpenAI Whisper (small) |

**지원 모델 크기**: tiny, base, small, medium, large

---

#### **ai_core/service/** - 서비스 오케스트레이션
| 파일 | 설명 |
|------|------|
| `orchestrator.py` | 메인 상담 로직 오케스트레이터 (감정 분석 → LLM 응답) |
| `video_stream_processor.py` | 실시간 비디오 스트림 처리 및 멀티턴 대화 |

**역할**: 모든 AI 모델을 통합하여 상담 워크플로우 관리

---

#### **ai_core/router/** - 응답 톤 라우팅
| 파일 | 설명 |
|------|------|
| `tone_router.py` | 감정별 응답 톤 및 전략 결정 |

**기능**: 사용자 감정에 따라 공감적, 격려적, 차분한 등의 톤 선택

---

#### **ai_core/io/** - 입출력 관리
| 파일 | 설명 | 상태 |
|------|------|------|
| `schemas.py` | Pydantic 데이터 스키마 정의 | ✅ 필수 |
| `redis_store.py` | Redis 기반 세션 저장소 | ❌ 현재 미사용 |
| `kafka_pubsub.py` | Kafka 이벤트 스트리밍 | ❌ 현재 미사용 |

---

#### **ai_core/configs/** - 설정 파일
```yaml
llm.yaml       # LLM 모델 설정
paths.yaml     # 파일 경로 설정
stream.yaml    # 스트림 처리 설정
```

---

## 🚀 실행 스크립트

### **scripts/** - 데모 및 테스트 스크립트

#### ✅ **메인 데모 (중요)**
| 파일 | 설명 | 용도 |
|------|------|------|
| `demo_full_counseling_workflow.py` | 🔥 **전체 상담 워크플로우** | 프로덕션 데모 |
| `simple_counseling_demo.py` | 간단한 텍스트 기반 상담 | 빠른 테스트 |

**실행 예시**:
```bash
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3
```

---

#### 🟡 **비디오 처리 데모**
| 파일 | 설명 |
|------|------|
| `demo_video_multiturn_integrated.py` | 멀티턴 비디오 상담 (VAD 기반) |
| `demo_video_single_turn.py` | 단일 비디오를 한 턴으로 처리 |
| `demo_video_stream.py` | 실시간 비디오 스트림 처리 |

---

#### 🔧 **단위 테스트**
| 파일 | 테스트 대상 |
|------|------------|
| `test_face_emotion.py` | 얼굴 감정 분석 단독 테스트 |
| `test_audio_emotion.py` | 음성 감정 분석 단독 테스트 |
| `test_text_emotion.py` | 텍스트 감정 분석 단독 테스트 |

**실행**:
```bash
python scripts/test_face_emotion.py --image test_image/happy.png
```

---

#### ⚙️ **유틸리티**
| 파일 | 설명 |
|------|------|
| `demo_llm_call.py` | LLM 단독 호출 테스트 |
| `demo_emotion_stream.py` | 감정 스트림 처리 데모 |
| `demo_counseling_start.py` | 상담 시작 인사말 생성 테스트 |
| `create_test_video.py` | 테스트용 비디오 생성 |
| `analyze_label_patterns.py` | 감정 라벨 패턴 분석 |

---

#### 🖥️ **서버 실행**
| 파일 | 설명 |
|------|------|
| `run_vllm.sh` | vLLM 서버 실행 스크립트 |
| `run_vllm_with_lora.sh` | vLLM + LoRA 어댑터 실행 |

---

## 🎬 테스트 데이터

### **test_video/** - 비디오 파일
```
sample.mp4       # 메인 테스트 비디오 (13MB, ~10초)
```

### **test_audio/** - 오디오 파일
```
angry.mp3        # 분노 감정 샘플
sad.mp3          # 슬픔 감정 샘플
```

### **test_image/** - 이미지 파일
```
happy.png        # 행복 표정 샘플
sad.png          # 슬픔 표정 샘플
```

---

## 🎓 모델 학습 (training/)

| 파일 | 설명 |
|------|------|
| `emotion_style_lora.py` | 감정별 스타일 LoRA 파인튜닝 |
| `sft_lora.py` | Supervised Fine-Tuning with LoRA |
| `prepare_cactus.py` | CACTUS 데이터셋 준비 |
| `merge_adapter.py` | LoRA 어댑터 병합 |

**현재 상태**: 프롬프트 기반 감정 조절로 파인튜닝 불필요

---

## 📄 문서 파일

### **주요 가이드**
| 파일 | 내용 |
|------|------|
| `README.md` | 프로젝트 메인 설명 |
| `PROJECT_STRUCTURE.md` | 🔥 **이 문서** - 전체 구조 설명 |
| `QUICKSTART_STEP2_3.md` | 빠른 시작 가이드 |

### **기능별 가이드**
| 파일 | 내용 |
|------|------|
| `MULTITURN_EMOTION_README.md` | 멀티턴 감정 대화 설명 |
| `VIDEO_MULTITURN_GUIDE.md` | 비디오 멀티턴 처리 가이드 |
| `VIDEO_STREAM_README.md` | 비디오 스트림 처리 가이드 |
| `VLLM_ALTERNATIVE_GUIDE.md` | vLLM 대신 Transformers 사용법 |

### **개발 참고**
| 파일 | 내용 |
|------|------|
| `CURRENT_PIPELINE_STATUS.md` | 현재 파이프라인 상태 |
| `IMPLEMENTATION_SUMMARY.md` | 구현 요약 |
| `explain_repo.md` | 저장소 상세 설명 |
| `How to use.md` | 사용법 |

---

## 🐳 배포 관련

### **docker/**
```
Dockerfile       # Docker 컨테이너 이미지 정의
```

### **contracts/**
```
emotion.proto    # gRPC 프로토콜 정의
openapi.yaml     # REST API OpenAPI 스펙
```

### **main.py**
FastAPI 서버 엔트리포인트 (개발 중)

---

## ⚙️ 설정 파일

### **requirements.txt**
Python 패키지 의존성 목록

**주요 패키지**:
- `torch==2.3.1` - PyTorch
- `transformers==4.45.2` - Hugging Face Transformers
- `speechbrain==1.0.0` - 음성 감정 인식
- `openai-whisper` - STT
- `opencv-python` - 비디오 처리
- `deepface` - 얼굴 감정 인식

---

## 🚀 빠른 시작

### **방법 1: Mock 모드 (GPU 없이 테스트)** 🎭 ⭐ 권장

#### **1. 환경 설정**
```bash
# Conda 환경 생성
conda create -n emotion_ai python=3.10
conda activate emotion_ai

# 최소 패키지만 설치
pip install numpy pydantic opencv-python soundfile
```

#### **2. Mock 모드로 실행**
```bash
# Mock 모드 활성화
export MOCK_MODE=true  # Linux/macOS
# 또는
set MOCK_MODE=true     # Windows

# 데모 실행
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0 \
    --auto
```

**특징**:
- ✅ GPU 불필요
- ✅ 즉시 실행 (수초 내)
- ✅ API 구조 완벽 테스트
- ✅ 백엔드 개발에 최적

---

### **방법 2: 실제 AI 모델 (GPU 환경)**

#### **1. 환경 설정**
```bash
# Conda 환경 생성
conda create -n emotion_ai python=3.10
conda activate emotion_ai

# 전체 패키지 설치
pip install -r requirements.txt
```

#### **2. 메인 데모 실행**
```bash
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0
```

#### **3. 옵션**
```bash
--whisper-model small          # Whisper 모델 크기 (tiny/base/small/medium/large)
--load-in-8bit                 # 8-bit 양자화 (메모리 절약)
--model-name Qwen/Qwen2-1.5B-Instruct  # 더 작은 LLM 사용
--auto                         # 자동 모드 (사용자 입력 없이)
```

---

## 📊 시스템 요구사항

### **Mock 모드 (백엔드 개발/테스트)** 🎭 ⭐ 권장
- **GPU**: ❌ 불필요
- **RAM**: 4GB 이상
- **저장공간**: 500MB
- **Python**: 3.10
- **환경 변수**: `MOCK_MODE=true`

**장점**: GPU 없이 API 구조 완벽 테스트, 즉시 실행

---

### **실제 AI 모델 (프로덕션)** 🖥️
- **GPU**: ✅ NVIDIA CUDA 지원 (L40S, RTX 3090 등)
- **GPU 메모리**: 16GB 이상 (8-bit: 10GB)
- **RAM**: 32GB 이상
- **저장공간**: 50GB 이상 (모델 캐시)
- **Python**: 3.10
- **CUDA**: 11.8 이상
- **FFmpeg**: ✅ 필수 (비디오/오디오 처리)

---

## 🎯 백엔드 테스트용 최소 파일

FastAPI 통합 테스트에 필요한 최소 구성:

```
✅ ai_core/ (전체 폴더)
✅ scripts/demo_full_counseling_workflow.py
✅ test_video/sample.mp4
✅ requirements.txt
✅ README.md
✅ PROJECT_STRUCTURE.md (이 파일)
```

---

## 📝 참고사항

### **모델 다운로드**
- 모든 모델은 첫 실행 시 자동으로 다운로드됩니다
- 캐시 위치:
  - Hugging Face: `~/.cache/huggingface/`
  - Whisper: `~/.cache/whisper/`
  - DeepFace: `~/.deepface/weights/`
  - SpeechBrain: `models/speechbrain_emotion/`

### **GPU 메모리 최적화**
```bash
# 8-bit 양자화 사용
--load-in-8bit

# 작은 모델 사용
--model-name Qwen/Qwen2-1.5B-Instruct

# 작은 Whisper 모델
--whisper-model tiny
```

---

## 🔗 관련 문서

- [빠른 시작 가이드](QUICKSTART_STEP2_3.md)
- [멀티턴 감정 가이드](MULTITURN_EMOTION_README.md)
- [비디오 처리 가이드](VIDEO_MULTITURN_GUIDE.md)
- [모델 및 라이브러리 목록](MODELS_AND_LIBRARIES.md) - 예정

---

## 📧 문의

프로젝트 관련 문의나 이슈는 GitHub Issues를 통해 남겨주세요.

---

**마지막 업데이트**: 2025-01-28  
**버전**: 1.0
