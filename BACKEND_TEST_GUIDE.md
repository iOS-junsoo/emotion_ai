# 🚀 백엔드 개발자용 테스트 가이드

> **파일**: emotion_ai_backend_test.zip (3.4MB)  
> **포함 파일**: 37개 (AI 모듈, 실행 스크립트, 테스트 데이터, 문서)

---

## 📦 포함된 파일

```
emotion_ai_backend_test.zip
├── ai_core/                    # 모든 AI 모델 및 서비스 로직
│   ├── emotion/                # 얼굴/음성/텍스트 감정 분석
│   ├── llm/                    # LLM 클라이언트 (Transformers, vLLM)
│   ├── stt/                    # Whisper STT
│   ├── service/                # 오케스트레이터
│   ├── router/                 # 톤 라우팅
│   └── io/                     # 데이터 스키마
│
├── scripts/
│   └── demo_full_counseling_workflow.py  # 메인 데모
│
├── test_video/
│   └── sample.mp4              # 테스트용 비디오 (13MB)
│
├── requirements.txt            # Python 패키지 목록
├── README.md                   # 프로젝트 소개
└── PROJECT_STRUCTURE.md        # 전체 구조 설명
```

---

## 🎯 빠른 시작

### **🎭 GPU 없는 환경 (백엔드 개발자 권장)** ⭐⭐⭐⭐⭐

#### **1단계: 압축 해제**
```bash
unzip emotion_ai_backend_test.zip -d emotion_ai_test
cd emotion_ai_test
```

#### **2단계: 환경 설정**
```bash
# Conda 환경 생성
conda create -n emotion_ai python=3.10
conda activate emotion_ai

# 최소 패키지만 설치 (Mock 모드용)
pip install numpy pydantic opencv-python soundfile
```

#### **3단계: Mock 모드로 실행** 🎭
```bash
# GPU 불필요! API 구조 테스트용
export MOCK_MODE=true

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
- ✅ 실제 모델 다운로드 불필요

---

### **🖥️ GPU 있는 환경 (실제 AI 모델 사용)**

#### **1-2단계: 동일**

#### **3단계: 패키지 전체 설치**
```bash
# 모든 AI 패키지 설치 (첫 실행 시 10-15분 소요)
pip install -r requirements.txt
```

#### **4단계: 실제 모델로 실행**
```bash
# MOCK_MODE 설정 없이 실행
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0
```

---

## ⚙️ 실행 옵션

### **작은 모델로 빠른 테스트**
```bash
# Qwen2-1.5B 사용 (권장)
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0 \
    --model-name Qwen/Qwen2-1.5B-Instruct
```

### **메모리 절약 모드**
```bash
# 8-bit 양자화
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0 \
    --load-in-8bit
```

### **자동 테스트 모드**
```bash
# 사용자 입력 없이 자동 실행
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0 \
    --auto
```

---

## 📊 시스템 요구사항

### **Mock 모드 (백엔드 개발용)** 🎭 ⭐ 권장
- **GPU**: ❌ 불필요
- **RAM**: 4GB 이상
- **저장공간**: 500MB
- **Python**: 3.10+
- **FFmpeg**: ❌ 불필요 (Mock 모드에서는)

### **실제 AI 모델 (프로덕션)** 🖥️
- **GPU**: ✅ NVIDIA CUDA 지원 (최소 10GB VRAM)
- **RAM**: 16GB 이상 (32GB 권장)
- **저장공간**: 50GB 이상 (모델 자동 다운로드)
- **Python**: 3.10
- **CUDA**: 11.8 이상
- **FFmpeg**: ✅ 필수 (비디오 처리용)

---

## 🤖 사용되는 AI 모델

### **핵심 모델 (자동 다운로드)**
1. **Qwen2-7B-Instruct** (7B) - LLM
2. **DeepFace VGG-Face** (138M) - 얼굴 감정
3. **SpeechBrain Wav2Vec2** (95M) - 음성 감정
4. **KLUE-BERT** (110M) - 텍스트 감정
5. **Whisper Small** (244M) - STT

**첫 실행 시**: 모델 자동 다운로드 (~20-30GB, 10-15분)  
**이후 실행**: 캐시에서 즉시 로드

---

## 📝 실행 결과

### **예상 출력**
```
================================================================================
💬 AI 감정 상담 시스템
================================================================================

😊 현재 당신의 감정은 어떤가요?
  1. 슬픔 😢
  2. 분노 😠
  ...

⚙️  시스템 초기화 중...
✅ 초기화 완료!

💬 상담 시작
⏳ 상담사가 인사말을 준비하는 중...

🤖 상담사
"많이 힘드셨겠어요..."

📹 사용자 응답 분석 중...
✅ 분석 완료!
   😊 얼굴 감정: sad (45.3%)
   🎤 음성 감정: angry (32.1%)
   ✍️  텍스트 감정: sad (78.9%)
   🎯 최종 감정: sad
   📝 전사된 내용: "..."

📊 상담 요약
💭 감정 변화: angry → sad
📝 상담 내용: ...
💡 추천 행동:
   • 감정 일기 작성
   • 가벼운 운동
   ...
```

---

## 🎭 Mock 모드 vs 실제 모델

| 항목 | Mock 모드 🎭 | 실제 모델 🖥️ |
|------|--------------|-------------|
| **GPU 필요** | ❌ 불필요 | ✅ 필수 (10GB+) |
| **설치 크기** | ~100MB | ~30GB |
| **실행 속도** | 즉시 (1초 이내) | 첫 실행: 2-3분, 이후: 10-20초 |
| **응답 품질** | 단순 더미 | 실제 AI 품질 |
| **용도** | API 구조 테스트 | 실제 서비스, 품질 검증 |

**백엔드 개발 단계별 전략**:
```
1. Mock 모드로 API 개발 (빠른 반복)
2. API 완성 후 실제 모델 연동
3. GPU 서버에서 통합 테스트
```

---

## 🔧 트러블슈팅

### **문제: GPU 없음**
```bash
# 해결: Mock 모드 사용
export MOCK_MODE=true

# 그대로 실행
python scripts/demo_full_counseling_workflow.py --video test_video/sample.mp4 --auto
```

### **문제: GPU 메모리 부족**
```bash
# 8-bit 양자화 사용
--load-in-8bit

# 더 작은 모델 사용
--model-name Qwen/Qwen2-1.5B-Instruct
```

### **문제: FFmpeg 없음**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

---

## 📞 API 통합 힌트

### **Mock 모드 활성화**
```python
import os

# Mock 모드 켜기
os.environ['MOCK_MODE'] = 'true'

# 이제 실제 모델 대신 Mock 모델 사용
from ai_core.service.orchestrator import TurnOrchestrator

# 평소와 동일하게 사용
orchestrator = TurnOrchestrator(
    llm_backend="transformers",
    gpu_id=0
)
```

### **주요 클래스**
```python
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput, TurnOutput

# 초기화
orchestrator = TurnOrchestrator(
    llm_backend="transformers",
    gpu_id=0
)

# 세션 시작
result = orchestrator.start_session(
    session_id="session_001",
    user_initial_text="요즘 너무 힘들어요",
    user_emotion="sad"
)

# 대화 턴 처리
turn_input = TurnInput(utterance="...")
output = orchestrator.run(turn_input, session_id="session_001")
```

### **Mock 모드 동작 방식**

Mock 모드는 환경 변수 `MOCK_MODE=true`로 활성화되며, 다음과 같이 동작합니다:

1. **모델 교체**: 실제 AI 모델 대신 더미 모델 사용
   - `MockLLMClient`: 키워드 기반 간단한 응답 생성
   - `MockFaceEmotionEstimator`: 랜덤 얼굴 감정
   - `MockAudioEmotionEstimator`: 랜덤 음성 감정
   - `MockTextEmotionEstimator`: 키워드 기반 텍스트 감정
   - `MockWhisperSTT`: 미리 정의된 더미 전사 텍스트

2. **API 구조 유지**: 입출력 인터페이스는 실제 모델과 100% 동일
   - `TurnInput`, `TurnOutput` 스키마 동일
   - `orchestrator.run()`, `start_session()` 호출 방식 동일
   - 응답 형식 동일

3. **빠른 실행**: GPU 연산 없이 즉시 응답
   - 모델 로딩: 0초
   - 추론 시간: 0.1-0.5초

**개발 워크플로우 예시**:
```bash
# 1. Mock 모드로 API 개발
export MOCK_MODE=true
python your_backend_server.py

# 2. API 완성 후 실제 모델 테스트
unset MOCK_MODE  # 또는 export MOCK_MODE=false
python your_backend_server.py
```

---

## 📧 문의

- 프로젝트 전체 구조: `PROJECT_STRUCTURE.md` 참고
- 기술 문의: GitHub Issues 또는 이메일

---

**준비 완료!** 이 ZIP 파일을 백엔드 개발자에게 전달하세요. 🎉
