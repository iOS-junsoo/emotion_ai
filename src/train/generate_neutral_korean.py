"""
neutral 감정 한국어 상담 데이터 생성
====================================
GPT-4o-mini로 neutral 감정 한국어 상담 대화 400개 생성
기존 영어 데이터를 대체하여 neutral LoRA 재학습에 사용

실행: python generate_neutral_korean.py
위치: src/train/generate_neutral_korean.py
"""

import openai
import json
import os
import time
import random

# API 키
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 상담 주제 (다양하게)
TOPICS = [
    "직장 스트레스", "대인관계 고민", "진로 고민", "자존감",
    "불면증", "가족 갈등", "학업 스트레스", "건강 걱정",
    "이직 고민", "연애 고민", "외로움", "번아웃",
    "시간 관리", "결정 장애", "미래 불안", "일상 권태",
]

# neutral 상황 시나리오 (다양한 neutral 유형)
NEUTRAL_SCENARIOS = [
    "감정을 잘 모르겠다고 하는 내담자",
    "괜찮다고 하지만 실제로는 복잡한 감정을 느끼는 내담자",
    "감정을 억누르고 담담하게 이야기하는 내담자",
    "처음 상담이라 긴장되지만 표현하지 않는 내담자",
    "상황을 객관적으로 설명하려고 하는 내담자",
    "감정보다 해결책을 원하는 내담자",
    "피곤하고 무기력한 상태의 내담자",
    "변화를 원하지만 아직 구체적이지 않은 내담자",
]


def generate_conversation(topic, scenario, idx):
    """neutral 한국어 상담 대화 생성"""

    prompt = f"""당신은 CBT(인지행동치료) 기반 심리 상담 데이터를 생성하는 전문가입니다.

아래 조건에 맞는 한국어 상담사-내담자 대화를 생성해주세요.

[조건]
- 내담자의 현재 감정: 평온/무감정 (neutral)
- 내담자 상황: {scenario}
- 상담 주제: {topic}
- 상담사의 어투 특징:
  - 중립적이고 탐색적인 어투
  - 감정을 강요하지 않고 부드럽게 탐색
  - "어떤 생각이 드세요?", "좀 더 이야기해 주실 수 있나요?" 같은 개방형 질문
  - 내담자가 감정을 찾도록 자연스럽게 안내
  - 판단하지 않고 있는 그대로 수용
- 대화는 4~6턴 (상담사-내담자 왕복)
- 반드시 자연스러운 한국어 구어체 사용
- "~하셨군요", "~느끼셨겠네요" 같은 한국어 상담 어투 사용
- 절대 영어, 중국어, 일본어를 섞지 마세요
- CBT 기법(감정 인식, 사고 탐색 등)을 자연스럽게 적용

[출력 형식 - JSON만 출력]
{{
  "emotion": "neutral",
  "topic": "{topic}",
  "messages": [
    {{"role": "system", "content": "당신은 CBT 기반 심리 상담사입니다. 내담자의 현재 감정: 평온/무감정 (neutral). 중립적이고 탐색적인 어투로 대화하세요. 반드시 한국어로만 응답하세요."}},
    {{"role": "user", "content": "내담자 발화"}},
    {{"role": "assistant", "content": "상담사 응답"}},
    ...더 많은 턴...
  ]
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "한국어로만 상담 데이터를 생성하세요. JSON 형식으로만 출력하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # JSON 추출
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
        return data

    except Exception as e:
        print(f"  에러 ({idx}): {e}")
        return None


# ============================================================
# 메인
# ============================================================

def main():
    print("=" * 60)
    print("neutral 한국어 상담 데이터 생성 (400개)")
    print("=" * 60)

    output_dir = "data/processed/emotion_lora"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "neutral_korean.json")

    # 이어하기 지원
    existing = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"  기존 데이터: {len(existing)}개, 이어서 생성")

    target = 400
    start_idx = len(existing)

    if start_idx >= target:
        print(f"  이미 {start_idx}개 생성 완료!")
        return

    print(f"  생성 시작: {start_idx}번째부터 → 목표 {target}개")
    print(f"  예상 비용: ~${(target - start_idx) * 0.0005:.2f}")
    print()

    new_data = list(existing)

    try:
        for i in range(start_idx, target):
            topic = random.choice(TOPICS)
            scenario = random.choice(NEUTRAL_SCENARIOS)

            result = generate_conversation(topic, scenario, i)

            if result:
                new_data.append(result)

            if (i + 1) % 10 == 0:
                print(f"  진행: {i+1}/{target} (생성: {len(new_data)}개)")

            # 50개마다 중간 저장
            if (i + 1) % 50 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, ensure_ascii=False, indent=2)
                print(f"  >> 중간 저장 ({len(new_data)}개)")

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n  중단됨! 진행분 저장 중...")

    # 최종 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"완료! {len(new_data)}개 생성")
    print(f"저장: {output_path}")
    print(f"{'=' * 60}")

    # 샘플 확인
    if new_data:
        sample = new_data[-1]
        print(f"\n[샘플] 주제: {sample.get('topic', 'N/A')}")
        for msg in sample.get('messages', [])[:4]:
            role_kr = {"system": "시스템", "user": "내담자", "assistant": "상담사"}
            print(f"  [{role_kr.get(msg['role'], msg['role'])}] {msg['content'][:80]}")


if __name__ == "__main__":
    main()
