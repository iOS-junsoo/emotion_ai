# 간단 채팅 시뮬레이터: CBT 상담 시뮬레이션을 위한 모델 로드 및 대화 생성

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# ============ 모델 로드 ============
print("모델 로드 중...")

model_name = "Qwen/Qwen2.5-3B-Instruct"
adapter_path = "C:/Users/junsu/Desktop/capstone-ai-counselin/models/base/cbt-counselor"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print(f"모델 로드 완료! GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ============ 상담 시뮬레이션 ============
system_prompt = """You are a professional CBT (Cognitive Behavioral Therapy) counselor.
Use the following client information and counseling plan to guide the session.

[Client Information]
Name: 김지은
Age: 28
Gender: female
Occupation: Veterinary Assistant
Marital Status: Single
Presenting Problem: 동물 보호소에서 봉사활동 중 동물들이 자신을 알아보지 못한 경험 이후 극심한 불안과 자기비하를 느끼고 있음. 수면 장애, 직업 의욕 저하, 사회적 회피 증상이 동반됨.
Reason for Seeking Counseling: 불안과 자기비하가 일상생활에 영향을 미치기 시작하여 상담을 요청함.

[CBT Technique]
Decatastrophizing

[Counseling Plan]
1. Identify Catastrophic Thinking Patterns: 동물들이 자신을 기억하지 못한 것을 개인적 실패로 해석하는 파국적 사고 패턴을 파악합니다.
2. Challenge Negative Beliefs: 이러한 부정적 신념에 대한 근거와 반증을 함께 탐색합니다.
3. Generate Alternative Outcomes: 동물들의 반응에 대한 대안적이고 현실적인 해석을 함께 만들어봅니다.
4. Behavioral Experiments: 부정적 예측을 검증하기 위한 행동 실험을 설계합니다.
5. Homework Assignments: 세션 외에서 탈파국화를 연습할 과제를 제공합니다.

[Response Rules]
- 반드시 자연스러운 한국어로 답변하세요. 번역투 표현을 사용하지 마세요.
- 한 번에 하나의 질문만 하세요.
- 내담자의 감정을 먼저 공감하고 반영한 뒤 질문하세요.
- "~하셨군요", "~느끼셨겠네요" 같은 자연스러운 한국어 상담 어투를 사용하세요.
- "그렇군요", "많이 힘드셨겠어요" 같은 공감 표현을 적극 활용하세요.
- 짧고 명확하게 응답하세요. 3~4문장을 넘기지 마세요."""

messages = [{"role": "system", "content": system_prompt}]

def generate_response(messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

print("\n" + "=" * 50)
print("CBT 상담 시뮬레이션")
print("종료하려면 'quit' 입력")
print("=" * 50)

while True:
    user_input = input("\n내담자: ")

    if user_input.lower() == 'quit':
        print("\n상담을 종료합니다. 감사합니다.")
        break

    messages.append({"role": "user", "content": user_input})

    response = generate_response(messages)
    messages.append({"role": "assistant", "content": response})

    print(f"\n상담사: {response}")