import openai
import json
import os
import time
import random

# API 키 설정 - 여기에 네 키 입력
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

emotions = {
    'happy': '기쁨/행복',
    'sad': '슬픔/우울',
    'angry': '분노/화남',
    'surprise': '놀람/당황',
    'fear': '불안/두려움',
    'disgust': '혐오/불쾌',
    'neutral': '평온/무감정',
}

# 상담 주제 목록
topics = [
    "직장 스트레스",
    "불면증 케어",
    "이별과 상실",
    "자존감 회복",
    "대인관계",
    "진로 고민",
    "건강 문제",
    "기타",
]

def generate_conversation(emotion, topic, idx):
    """감정별 상담 대화 생성"""

    prompt = f"""당신은 CBT(인지행동치료) 기반 심리 상담 데이터를 생성하는 전문가입니다.

아래 조건에 맞는 상담사-내담자 대화를 생성해주세요.

[조건]
- 내담자의 현재 감정: {emotions[emotion]} ({emotion})
- 상담 주제: {topic}
- 상담사는 내담자의 감정({emotions[emotion]})을 인지하고 있습니다.
- 상담사의 어투는 내담자의 감정에 맞게 조정되어야 합니다:
  - 내담자가 슬프면: 부드럽고 따뜻한 어투, 천천히 공감
  - 내담자가 화나면: 차분하고 안정적인 어투, 감정을 수용
  - 내담자가 불안하면: 안심시키는 어투, 구체적 안내
  - 내담자가 행복하면: 밝고 격려하는 어투
  - 내담자가 놀랐으면: 안정시키며 상황 파악하는 어투
  - 내담자가 혐오감을 느끼면: 수용적이고 판단하지 않는 어투
  - 내담자가 평온하면: 중립적이고 탐색적인 어투
- 상담사는 내담자의 말을 감정 맥락에서 해석해야 합니다.
  예: "웃기네"를 화난 감정에서 말하면 "어이없다"로 이해
  예: "괜찮아요"를 슬픈 감정에서 말하면 "괜찮지 않다"로 이해
  예: "감사합니다"를 혐오 감정에서 말하면 "비꼬는 것"으로 이해
- 대화는 4~6턴 (상담사-내담자 왕복)
- 자연스러운 한국어 구어체 사용
- CBT 기법(인지 재구조화, 탈파국화 등)을 자연스럽게 적용

[출력 형식]
JSON 형식으로 출력해주세요:
{{
  "emotion": "{emotion}",
  "topic": "{topic}",
  "messages": [
    {{"role": "system", "content": "시스템 프롬프트"}},
    {{"role": "user", "content": "내담자 발화"}},
    {{"role": "assistant", "content": "상담사 응답"}},
    ...
  ]
}}

시스템 프롬프트에는 "내담자의 현재 감정: {emotions[emotion]}" 정보를 포함해주세요."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000,
        )

        content = response.choices[0].message.content
        # JSON 추출
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        data = json.loads(content.strip())
        return data

    except Exception as e:
        print(f"  에러 ({emotion}-{idx}): {e}")
        return None


# ============ 데이터 생성 ============
print("=" * 50)
print("감정별 상담 데이터 추가 생성 (300개씩)")
print("=" * 50)

output_dir = "data/processed/emotion_lora"
os.makedirs(output_dir, exist_ok=True)

per_emotion = 300  # 추가 300개
all_data = []

for emotion, emo_kr in emotions.items():
    # 기존 데이터 로드
    emotion_path = os.path.join(output_dir, f"{emotion}.json")
    existing = []
    if os.path.exists(emotion_path):
        with open(emotion_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"\n[{emotion} ({emo_kr})] 기존: {len(existing)}개, 추가 {per_emotion}개 생성 중...")
    else:
        print(f"\n[{emotion} ({emo_kr})] 새로 {per_emotion}개 생성 중...")

    new_data = []
    for i in range(per_emotion):
        topic = random.choice(topics)
        result = generate_conversation(emotion, topic, i)

        if result:
            new_data.append(result)

        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{per_emotion} (성공: {len(new_data)})")

        time.sleep(0.5)  # 레이트 리밋 방지

    # 기존 + 신규 합치기
    combined = existing + new_data
    with open(emotion_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    all_data.extend(combined)
    print(f"  [{emotion}] 완료: {len(combined)}개 (기존 {len(existing)} + 신규 {len(new_data)})")

# 전체 저장
all_path = os.path.join(output_dir, "all_emotions.json")
with open(all_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 50}")
print(f"전체 완료: {len(all_data)}개")
print(f"저장 경로: {output_dir}")
print(f"{'=' * 50}")