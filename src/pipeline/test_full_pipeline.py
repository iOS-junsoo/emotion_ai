"""
로컬 통합 테스트: GPT-4o-mini 플랜 + 감정별 LoRA 상담 시뮬레이션
================================================================
1. GPT-4o-mini → 맞춤형 5-Step 플랜 생성
2. 로컬 Qwen 2.5 3B + CBT merge + 한국어 merge + 감정별 LoRA → 상담

실행: python test_full_pipeline.py
위치: src/pipeline/test_full_pipeline.py
"""

import os
import sys
import time
import torch

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

# ============================================================
# 설정
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CBT_EN_PATH = os.path.join(PROJECT_DIR, "models", "base", "cbt-counselor")
CBT_KR_PATH = os.path.join(PROJECT_DIR, "models", "base", "cbt-counselor-korean")
LORA_DIR = os.path.join(PROJECT_DIR, "models", "lora")

# ⚠️ 여기에 API 키 입력
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]


# ============================================================
# 1. GPT-4o-mini 플랜 생성
# ============================================================

def generate_plan():
    """GPT-4o-mini로 동적 플랜 생성"""
    from dynamic_plan import DynamicPlanGenerator

    print("=" * 60)
    print("1. GPT-4o-mini 플랜 생성")
    print("=" * 60)

    generator = DynamicPlanGenerator(api_key=OPENAI_API_KEY)
    plan = generator.generate(
        topic="직장 스트레스",
        emotion="angry",
        detail="상사가 제 성과를 자기 것처럼 가져가고, 매일 야근하는데 인정도 안 해줘요.",
    )

    print(f"\n  동적 플랜: {plan['is_dynamic']}")
    analysis = plan.get("analysis", {})
    print(f"  핵심 문제: {analysis.get('core_problem', 'N/A')}")
    print(f"  인지 패턴: {analysis.get('cognitive_pattern', 'N/A')}")
    print(f"\n  5-Step:")
    for step in plan["steps"]:
        status = "●" if step["status"] == "active" else "○"
        print(f"    {status} Step {step['step']}: {step['name']} — {step.get('goal', '')[:50]}")

    return plan


# ============================================================
# 2. 로컬 모델 로드 (CBT merge + 한국어 merge + 감정별 LoRA)
# ============================================================

def load_model():
    """로컬 모델 로드"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n{'=' * 60}")
    print("2. 로컬 모델 로드")
    print("=" * 60)

    start = time.time()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 베이스 모델
    print("  [1/4] 베이스 모델 로드...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )

    # 영어 CBT merge
    print("  [2/4] 영어 CBT LoRA merge...")
    if os.path.exists(CBT_EN_PATH):
        model = PeftModel.from_pretrained(base_model, CBT_EN_PATH)
        model = model.merge_and_unload()
        print("    ✅ merge 완료")
    else:
        print(f"    ⚠️ 경로 없음: {CBT_EN_PATH}, 스킵")
        model = base_model

    # 한국어 CBT merge
    print("  [3/4] 한국어 CBT LoRA merge...")
    if os.path.exists(CBT_KR_PATH):
        model = PeftModel.from_pretrained(model, CBT_KR_PATH)
        model = model.merge_and_unload()
        print("    ✅ merge 완료")
    else:
        print(f"    ⚠️ 경로 없음: {CBT_KR_PATH}, 스킵")

    # 감정별 LoRA multi-adapter
    print("  [4/4] 감정별 LoRA 로드...")
    loaded = []
    first = True
    for emo in EMOTIONS:
        path = os.path.join(LORA_DIR, emo)
        if not os.path.exists(path):
            print(f"    [{emo}] ❌ 스킵")
            continue
        if first:
            model = PeftModel.from_pretrained(model, path, adapter_name=emo)
            first = False
        else:
            model.load_adapter(path, adapter_name=emo)
        loaded.append(emo)
        print(f"    [{emo}] ✅")

    model.set_adapter("neutral")
    model.eval()

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1024**3

    print(f"\n  로드 완료! 어댑터: {loaded}")
    print(f"  소요: {elapsed:.1f}초 | GPU: {vram:.2f} GB")

    return model, tokenizer, loaded


# ============================================================
# 3. 상담 시뮬레이션
# ============================================================

def run_simulation(plan, model, tokenizer, loaded_adapters):
    """동적 플랜 + 감정별 LoRA 상담 시뮬레이션"""
    from dynamic_plan import build_dynamic_system_prompt

    print(f"\n{'=' * 60}")
    print("3. 상담 시뮬레이션 (GPT-4o-mini 플랜 + 감정별 LoRA)")
    print(f"   주제: {plan['topic']} | 초기 감정: {plan['initial_emotion']}")
    print("=" * 60)

    # 내담자 발화
    USER_TURNS = {
        1: [
            "상사가 제 성과를 자기 것처럼 가져가서 정말 화가 나요. 매일 야근하는데 인정도 안 해주고...",
            "네, 정말 힘들어요. 요즘은 출근하기가 싫을 정도예요.",
        ],
        2: [
            "지난주에 제가 3주 동안 준비한 보고서를 팀장이 자기 이름으로 제출했어요.",
            "그때 '나는 여기서 아무 의미도 없구나'라는 생각이 들었어요.",
        ],
        3: [
            "음... 동료들은 제가 잘한다고 해주긴 해요. 근데 팀장이 인정 안 해주니까 의미가 없는 것 같아요.",
            "그러고 보니 팀장 한 사람의 평가만으로 제 능력 전체를 판단하고 있었네요.",
        ],
        4: [
            "이번 주에 퇴근 시간을 지켜보려고요. 그리고 동료한테 솔직하게 이야기해볼까 해요.",
        ],
        5: [
            "오늘 이야기하면서 많이 정리가 된 것 같아요. 감사합니다.",
        ],
    }

    STEP_EMOTIONS = {1: "angry", 2: "sad", 3: "sad", 4: "neutral", 5: "neutral"}

    current_emotion = "neutral"
    history = []

    def switch_and_generate(emotion, messages, max_new_tokens=200):
        nonlocal current_emotion
        if emotion != current_emotion and emotion in loaded_adapters:
            model.set_adapter(emotion)
            current_emotion = emotion

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, do_sample=True, repetition_penalty=1.2,
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    for step_num in range(1, 6):
        step_data = plan["steps"][step_num - 1]
        emotion = STEP_EMOTIONS[step_num]

        print(f"\n{'━' * 60}")
        print(f"📌 Step {step_num}: {step_data['name']} [LoRA: {emotion}]")
        print(f"   목표: {step_data.get('goal', '')}")
        qs = step_data.get("key_questions", [])
        if qs:
            print(f"   핵심 질문: {qs}")
        print(f"{'━' * 60}")

        # 시스템 프롬프트 생성
        sys_prompt = build_dynamic_system_prompt(step_data, plan["topic"], emotion)

        # 스텝 시작 발화
        context = [{"role": "system", "content": sys_prompt}]
        if history:
            context.extend(history[-4:])
        starter = "안녕하세요, 상담 시작할게요." if step_num == 1 else "네, 계속 이야기할게요."
        context.append({"role": "user", "content": starter})

        opening = switch_and_generate(emotion, context, max_new_tokens=150)
        print(f"\n  🩺 상담사 [{emotion}]: {opening}")
        history.append({"role": "assistant", "content": opening})

        # 턴별 대화
        for user_text in USER_TURNS.get(step_num, []):
            step_data["current_turns"] += 1
            print(f"\n  🙋 내담자: {user_text}")
            history.append({"role": "user", "content": user_text})

            sys_prompt = build_dynamic_system_prompt(step_data, plan["topic"], emotion)
            context = [{"role": "system", "content": sys_prompt}]
            context.extend(history[-6:])

            response = switch_and_generate(emotion, context, max_new_tokens=200)
            print(f"  🩺 상담사 [{emotion}]: {response}")
            history.append({"role": "assistant", "content": response})

    print(f"\n{'━' * 60}")
    print(f"✅ 완료! 총 대화: {len(history)}개")
    print(f"{'━' * 60}")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    # 1. 플랜 생성 (GPU 불필요)
    plan = generate_plan()

    if not plan.get("is_dynamic"):
        print("\n⚠️ 동적 플랜 생성 실패. 폴백 플랜으로 진행합니다.")

    # 2. 모델 로드
    model, tokenizer, loaded_adapters = load_model()

    # 3. 상담 시뮬레이션
    run_simulation(plan, model, tokenizer, loaded_adapters)
