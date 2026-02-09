# 비디오 멀티턴 상담 시스템 사용 가이드

## 🎯 **문제 해결**

### **문제 1: LLM 응답 실패**
❌ **이전**: vLLM 서버 필요 (localhost:8000)
✅ **해결**: Transformers 백엔드 지원

### **문제 2: 텍스트 전사 실패**
❌ **이전**: Whisper가 한국어를 제대로 인식하지 못함
✅ **해결**: 언어 설정 명시화, 더 큰 모델 옵션 제공

---

## 🚀 **권장 사용법**

### **기본 실행 (Transformers + 작은 Whisper)**

```bash
cd /home/junsu/Desktop/emotion_ai/counseling-ai

python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3
```

**특징**:
- LLM: Transformers (로컬 GPU 3)
- Whisper: base 모델
- 언어: 한국어 (ko)

---

### **고품질 전사 (더 큰 Whisper 모델)**

```bash
python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3 \
    --whisper-model small
```

**Whisper 모델 크기**:
- `tiny`: 가장 빠르지만 정확도 낮음
- `base`: 기본값, 빠르고 적당한 정확도
- `small`: 느리지만 더 정확 (권장)
- `medium`: 매우 느리지만 높은 정확도
- `large`: 가장 느리지만 최고 정확도

---

### **메모리 절약 모드 (8비트 양자화)**

```bash
python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3 \
    --load-in-8bit
```

**효과**: LLM 메모리 사용량 약 50% 절감

---

### **빠른 테스트 모드**

```bash
python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3 \
    --fast
```

**특징**:
- 표정 캡처: 2초마다 (기본 1초)
- Whisper: tiny 모델
- 빠른 처리, 낮은 품질

---

## 🎛️ **전체 옵션**

### **비디오 관련**
- `--video`: 비디오 파일 경로 (필수)
- `--gpu`: GPU 번호 (기본: 3)
- `--fast`: 빠른 처리 모드
- `--whisper-model`: Whisper 모델 크기 (tiny/base/small/medium/large)
- `--language`: 전사 언어 (기본: ko)

### **감정 가중치**
- `--face-weight`: 표정 가중치 (기본: 0.45)
- `--audio-weight`: 음성 가중치 (기본: 0.35)
- `--text-weight`: 텍스트 가중치 (기본: 0.20)

### **대화 관련**
- `--session-id`: 세션 ID (기본: video_session_001)
- `--disable-multiturn`: 멀티턴 비활성화

### **LLM 백엔드**
- `--backend`: LLM 백엔드 (vllm/transformers, 기본: transformers)
- `--load-in-8bit`: 8비트 양자화 (메모리 절약)
- `--model-name`: LLM 모델 이름 (기본: Qwen/Qwen2-7B-Instruct)

---

## 📊 **출력 예시**

```
================================================================================
🎤 턴 1: 발화 감지 (시간: 3.40초)
--------------------------------------------------------------------------------
📝 전사 텍스트:
   요즘 너무 힘들어요. 스트레스가 심해서 잠도 못 자고 있어요.

😊 통합 감정 (neutral 제외):
   주요: sad (65.3%)
   전체: sad: 65.3%, anxious: 25.2%, fear: 9.5%

🤖 상담사 응답 생성 중...
💬 대화 맥락: 1턴 누적
🎯 응답 전략: gentle / 공감적 경청

💬 상담사:
   많이 힘드셨겠어요. 스트레스로 잠도 못 자신다니 정말 고되시겠네요.
   천천히 이야기 나눠봐요. 어떤 일로 이렇게 힘드신 건가요?

================================================================================
🎤 턴 2: 발화 감지 (시간: 4.12초)
--------------------------------------------------------------------------------
...
```

---

## ⚠️ **문제 해결**

### **1. 전사 텍스트가 이상하거나 비어있음**

**원인**:
- 음성이 너무 작거나 배경 소음이 많음
- Whisper 모델이 너무 작음
- 언어 설정이 잘못됨

**해결**:
```bash
# 더 큰 Whisper 모델 사용
python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3 \
    --whisper-model small  # 또는 medium
```

### **2. LLM 응답 실패**

**원인**: vLLM 백엔드를 사용하려고 했지만 서버가 없음

**해결**:
```bash
# Transformers 백엔드 사용
--backend transformers
```

### **3. GPU 메모리 부족**

**해결**:
```bash
# 8비트 양자화 사용
--load-in-8bit

# 또는 더 작은 모델
--model-name Qwen/Qwen2-1.5B-Instruct
```

---

## 🎬 **테스트 워크플로우**

### **1단계: 비디오만 확인 (LLM 없이)**

```bash
python scripts/demo_video_stream.py \
    --video test_video/sample.mp4 \
    --gpu 3
```

**확인사항**:
- 몇 개의 턴이 감지되는가?
- 전사 텍스트가 정확한가?
- 감정 분석이 합리적인가?

### **2단계: 전체 파이프라인 테스트**

```bash
python scripts/demo_video_multiturn_integrated.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 3 \
    --whisper-model small  # 더 정확한 전사
```

**확인사항**:
- LLM 응답이 생성되는가?
- 응답이 감정에 적합한가?
- 멀티턴 맥락이 유지되는가?

---

## 💡 **권장 설정**

### **개발/테스트**
```bash
--backend transformers \
--gpu 3 \
--whisper-model base \
--fast
```

### **프로덕션/데모**
```bash
--backend transformers \
--gpu 3 \
--whisper-model small \
--load-in-8bit
```

---

## 📝 **현재 비디오 정보**

`test_video/sample.mp4`에서 감지된 내용:
- 총 6개 턴 감지됨
- 일부 턴에서 전사 실패 (빈 텍스트 또는 이상한 문자)

**해결 방법**:
1. 더 나은 품질의 비디오 사용
2. 더 큰 Whisper 모델 사용 (`--whisper-model small`)
3. 오디오 전처리 (노이즈 제거 등)
