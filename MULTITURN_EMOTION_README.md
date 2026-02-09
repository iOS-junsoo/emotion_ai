# 멀티턴 감정 기반 대화 시스템 가이드

## 📋 목차

1. [시스템 개요](#시스템-개요)
2. [현재 사용 LLM 모델](#현재-사용-llm-모델)
3. [멀티턴 대화 기능](#멀티턴-대화-기능)
4. [감정 기반 LoRA 파인튜닝](#감정-기반-lora-파인튜닝)
5. [전체 워크플로우](#전체-워크플로우)
6. [실행 방법](#실행-방법)

---

## 🎯 시스템 개요

이 프로젝트는 **멀티모달 감정 인식**과 **감정 기반 응답 생성**을 결합한 상담 보조 시스템입니다.

### 주요 기능

- ✅ **멀티모달 감정 추출**: 얼굴(DeepFace), 음성(SpeechBrain), 텍스트(KLUE-BERT)
- ✅ **멀티턴 대화 지원**: 이전 대화 및 감정 변화 추적
- ✅ **감정 기반 LoRA 파인튜닝**: 감정에 맞는 응답 스타일 학습
- ✅ **실시간 비디오 스트림 처리**: GPU 가속 지원

---

## 🤖 현재 사용 LLM 모델

### **Qwen2-7B-Instruct** (또는 Qwen2.5-14B-Instruct)

```yaml
# ai_core/configs/llm.yaml
model:
  name: "Qwen2-7B-Instruct"
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

#### 모델 특징

| 항목 | 설명 |
|------|------|
| **제조사** | Alibaba Cloud |
| **파라미터** | 7B / 14B |
| **컨텍스트 길이** | 32K 토큰 |
| **언어** | 다국어 (한국어 포함) |
| **라이선스** | Apache 2.0 |
| **특화 분야** | 대화, 지시 수행, 추론 |

#### 멀티턴 대화 지원

✅ **Qwen2는 멀티턴 대화를 기본적으로 지원**합니다.
- 대화 히스토리를 `messages` 배열로 전달
- 이전 턴의 맥락을 이해하고 일관성 있는 응답 생성
- 감정 변화 추적 가능

---

## 💬 멀티턴 대화 기능

### 1. **대화 히스토리 관리**

```python
# ai_core/llm/conversation_history.py
from ai_core.llm.conversation_history import ConversationHistory

history = ConversationHistory(session_id="user_123")

# 턴 추가
history.add_turn(
    user_utterance="요즘 너무 힘들어요.",
    assistant_response="정말 힘드시겠어요. 어떤 부분이 가장 힘드신가요?",
    emotion={"sad": 0.8, "neutral": 0.2},
    tone="empathetic",
    strategy="active_listening"
)

# 감정 변화 추이
trajectory = history.get_emotion_trajectory(n=5)
print(trajectory)  # ['sad', 'sad', 'anxious', 'neutral', 'happy']
```

### 2. **멀티턴 Orchestrator 사용**

```python
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput

orchestrator = TurnOrchestrator(enable_multiturn=True)

# 세션 ID로 대화 연속성 유지
session_id = "user_123"

# 첫 번째 턴
ti1 = TurnInput(utterance="요즘 너무 힘들어요.")
output1 = orchestrator.run(ti1, session_id=session_id)
print(output1.response)

# 두 번째 턴 (이전 대화 기억)
ti2 = TurnInput(utterance="조금 나아진 것 같아요.")
output2 = orchestrator.run(ti2, session_id=session_id)
print(output2.response)  # 이전 "힘들다"는 맥락을 기억하고 응답
```

### 3. **대화 히스토리 포맷**

LLM에 전달되는 메시지 형식:

```json
{
  "messages": [
    {"role": "system", "content": "당신은 공감적인 상담사입니다..."},
    {"role": "user", "content": "[감정: sad] 요즘 너무 힘들어요."},
    {"role": "assistant", "content": "정말 힘드시겠어요..."},
    {"role": "user", "content": "[감정: neutral] 조금 나아진 것 같아요."},
    ...
  ]
}
```

---

## 🎓 감정 기반 LoRA 파인튜닝

### LoRA란?

**Low-Rank Adaptation (LoRA)**는 대규모 언어 모델을 효율적으로 파인튜닝하는 기법입니다.

#### 장점

- ✅ **적은 파라미터**: 전체 모델의 0.1~1%만 학습
- ✅ **빠른 학습**: GPU 메모리 절약, 학습 시간 단축
- ✅ **유연성**: 여러 LoRA 어댑터 동시 사용 가능
- ✅ **배포 용이**: 베이스 모델 + 어댑터 조합

### 학습 데이터 준비

```jsonl
# data/emotion_conversations.jsonl
{"emotion": "sad", "user": "요즘 너무 힘들어요.", "assistant": "정말 힘드시겠어요. 천천히 이야기 나눠볼까요?"}
{"emotion": "angry", "user": "정말 화가 나요!", "assistant": "화가 나시는 게 당연해요. 그 감정을 충분히 느끼셔도 괜찮아요."}
{"emotion": "happy", "user": "오늘 정말 좋은 일이 있었어요!", "assistant": "와, 정말 좋으시겠어요! 어떤 일이 있으셨는지 들려주세요."}
...
```

### LoRA 학습 실행

```bash
conda activate emotion_ai
cd /home/junsu/Desktop/emotion_ai/counseling-ai

# GPU 3번 사용하여 학습
python training/emotion_style_lora.py
```

또는 Python에서:

```python
from training.emotion_style_lora import train_emotion_lora

train_emotion_lora(
    base_model_name="Qwen/Qwen2-7B-Instruct",
    data_path="./data/emotion_conversations.jsonl",
    output_dir="./models/qwen2-emotion-lora",
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    gradient_accumulation_steps=8,
    gpu_id=3  # GPU 3번 사용
)
```

### 학습 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `r` | 8 | LoRA rank (4-16 권장) |
| `lora_alpha` | 16 | Scaling factor (보통 r의 2배) |
| `learning_rate` | 2e-4 | 학습률 |
| `num_epochs` | 3 | 에폭 수 |
| `batch_size` | 2-4 | 배치 크기 |
| `max_length` | 512 | 최대 시퀀스 길이 |

### LoRA 어댑터 병합 (선택사항)

배포를 위해 LoRA를 베이스 모델에 병합:

```python
from training.emotion_style_lora import merge_lora_to_base

merge_lora_to_base(
    base_model_name="Qwen/Qwen2-7B-Instruct",
    lora_adapter_path="./models/qwen2-emotion-lora",
    output_merged_path="./models/qwen2-emotion-merged"
)
```

---

## 🔄 전체 워크플로우

### 1️⃣ **초기 설정 (1회)**

```bash
# 1. 환경 설정
conda activate emotion_ai
cd /home/junsu/Desktop/emotion_ai/counseling-ai

# 2. vLLM 서버 시작 (베이스 모델)
bash scripts/run_vllm.sh  # 또는 run_vllm_with_lora.sh
```

### 2️⃣ **멀티턴 대화 실행**

```python
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput

# Orchestrator 초기화 (멀티턴 활성화)
orch = TurnOrchestrator(enable_multiturn=True)

# 세션 시작
session_id = "user_001"

# 대화 진행
turns = [
    "요즘 너무 힘들어요.",
    "아무것도 하기 싫어요.",
    "조금 나아진 것 같아요.",
]

for utterance in turns:
    ti = TurnInput(utterance=utterance)
    output = orch.run(ti, session_id=session_id)
    
    print(f"사용자: {utterance}")
    print(f"감정: {output.emotion.aggregated}")
    print(f"상담사: {output.response}")
    print(f"턴 수: {output.meta['turn_count']}")
    print()
```

### 3️⃣ **비디오 스트림 + 멀티턴 통합**

```python
from ai_core.service.video_stream_processor import VideoStreamProcessor
from ai_core.service.orchestrator import TurnOrchestrator

# 비디오 스트림 프로세서
video_processor = VideoStreamProcessor(gpu_id=3, verbose=False)

# 멀티턴 대화 orchestrator
orch = TurnOrchestrator(enable_multiturn=True)
session_id = "video_session_001"

def on_turn_complete(turn_result):
    """각 발화 종료 시 LLM 응답 생성"""
    ti = TurnInput(utterance=turn_result.transcribed_text)
    
    # 멀티턴 응답 생성
    output = orch.run(ti, session_id=session_id)
    
    print(f"🎤 사용자: {turn_result.transcribed_text}")
    print(f"😊 감정: {turn_result.aggregated_emotion_no_neutral}")
    print(f"🤖 상담사: {output.response}")
    print()

# 비디오 처리
video_processor.process_video_stream(
    video_path="test_video/sample.mp4",
    callback=on_turn_complete
)
```

### 4️⃣ **LoRA 학습 및 적용**

```bash
# 1. 학습 데이터 준비 (data/emotion_conversations.jsonl)

# 2. LoRA 학습
python training/emotion_style_lora.py

# 3. LoRA 적용 vLLM 서버 재시작
bash scripts/run_vllm_with_lora.sh

# 4. 테스트
python scripts/demo_llm_call.py
```

---

## 🚀 실행 방법

### 방법 1: 단일 턴 테스트

```bash
conda activate emotion_ai
cd /home/junsu/Desktop/emotion_ai/counseling-ai

python scripts/demo_llm_call.py
```

### 방법 2: 멀티턴 대화 테스트

```python
# scripts/demo_multiturn_conversation.py (새로 만들 파일)
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput

orch = TurnOrchestrator(enable_multiturn=True)
session_id = "test_session"

conversations = [
    "요즘 너무 힘들어요. 스트레스가 많아요.",
    "일도 잘 안 풀리고, 사람들과의 관계도 힘들어요.",
    "그래도 오늘은 좀 나은 것 같아요.",
    "조언 감사합니다. 힘이 됐어요."
]

for utterance in conversations:
    ti = TurnInput(utterance=utterance)
    output = orch.run(ti, session_id=session_id)
    
    print("=" * 80)
    print(f"👤 사용자: {utterance}")
    print(f"😊 감정: {output.emotion.aggregated}")
    print(f"🎯 톤: {output.meta['tone']}, 전략: {output.meta['strategy']}")
    print(f"🤖 상담사: {output.response}")
    print(f"📊 누적 턴: {output.meta['turn_count']}")
    print()
```

### 방법 3: 비디오 스트림 + 멀티턴

```bash
conda activate emotion_ai
cd /home/junsu/Desktop/emotion_ai/counseling-ai

# 실시간 처리 모드 (GPU 3번)
python scripts/demo_video_stream.py \
    --video test_video/sample.mp4 \
    --gpu 3 \
    --fast \
    --face-weight 0.45 \
    --audio-weight 0.35 \
    --text-weight 0.20
```

---

## 📊 성능 비교

### LoRA vs Full Fine-tuning

| 항목 | Full Fine-tuning | LoRA |
|------|-----------------|------|
| 학습 파라미터 | 7B 전체 | ~50M (0.7%) |
| GPU 메모리 | ~80GB | ~24GB |
| 학습 시간 | 수일 | 수시간 |
| 디스크 용량 | ~14GB | ~200MB |
| 어댑터 교체 | 불가 | 즉시 가능 |

### 멀티턴 vs 단일 턴

| 항목 | 단일 턴 | 멀티턴 |
|------|---------|--------|
| 대화 일관성 | ❌ | ✅ |
| 감정 변화 추적 | ❌ | ✅ |
| 맥락 이해 | ❌ | ✅ |
| 메모리 사용 | 낮음 | 중간 |
| 구현 복잡도 | 낮음 | 중간 |

---

## 🔧 고급 설정

### 1. **감정 가중치 조정**

```python
processor = VideoStreamProcessor(
    face_weight=0.50,   # 표정 50%
    audio_weight=0.30,  # 음성 30%
    text_weight=0.20    # 텍스트 20%
)
```

### 2. **대화 히스토리 길이 조정**

```python
history = ConversationHistory(
    session_id="user_123",
    max_history_turns=10  # 최대 10턴 보관
)
```

### 3. **여러 LoRA 어댑터 사용**

```bash
# vLLM 서버에 여러 어댑터 등록
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-7B-Instruct \
    --enable-lora \
    --lora-modules \
        emotion-sad=./models/lora-sad \
        emotion-angry=./models/lora-angry \
        emotion-happy=./models/lora-happy \
    --max-lora-rank 16
```

```python
# API 호출 시 어댑터 선택
response = client.chat_multiturn(
    model="emotion-sad",  # sad 감정에 특화된 어댑터
    messages=[...]
)
```

---

## 🐛 트러블슈팅

### Q1: "멀티턴 대화가 이전 대화를 기억하지 못해요"

**A1**: `session_id`를 동일하게 유지했는지 확인하세요.

```python
# ❌ 잘못된 사용
output1 = orch.run(ti1, session_id="user1")
output2 = orch.run(ti2, session_id="user2")  # 다른 세션!

# ✅ 올바른 사용
session_id = "user1"
output1 = orch.run(ti1, session_id=session_id)
output2 = orch.run(ti2, session_id=session_id)
```

### Q2: "LoRA 학습 시 OOM (Out of Memory) 에러"

**A2**: 배치 크기와 그래디언트 누적 스텝 조정:

```python
train_emotion_lora(
    batch_size=1,  # 배치 크기 감소
    gradient_accumulation_steps=16,  # 누적 스텝 증가
    max_length=256  # 시퀀스 길이 감소
)
```

### Q3: "vLLM 서버가 LoRA 어댑터를 로드하지 못해요"

**A3**: LoRA 어댑터 경로와 베이스 모델이 일치하는지 확인:

```bash
# 어댑터 구조 확인
ls -la ./models/qwen2-emotion-lora/
# adapter_config.json, adapter_model.safetensors 파일 존재 확인
```

---

## 📚 참고 자료

- [Qwen2 모델 문서](https://github.com/QwenLM/Qwen2)
- [LoRA 논문](https://arxiv.org/abs/2106.09685)
- [PEFT 라이브러리](https://github.com/huggingface/peft)
- [vLLM 문서](https://docs.vllm.ai/)

---

## 📝 요약

1. ✅ **현재 모델**: Qwen2-7B-Instruct (멀티턴 대화 기본 지원)
2. ✅ **멀티턴 구현**: `ConversationHistory` + `TurnOrchestrator`
3. ✅ **감정 추적**: 턴별 감정 저장 및 변화 추이 분석
4. ✅ **LoRA 파인튜닝**: 감정별 응답 스타일 학습 (파라미터 0.7%, GPU 메모리 ~24GB)
5. ✅ **실시간 처리**: GPU 가속 비디오 스트림 + 멀티턴 대화 통합

이제 **감정을 추적하며 일관성 있는 멀티턴 대화**와 **감정별 맞춤 응답**이 가능합니다! 🎉
