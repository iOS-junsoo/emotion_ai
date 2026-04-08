import openai
import json
import os
import time

# API 키 설정
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# CACTUS 데이터 로드
print("CACTUS 데이터 로드 중...")
with open('data/processed/cactus_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"전체 데이터: {len(train_data)}개")

# 저장 경로
output_dir = "data/processed/cactus_korean"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cactus_korean.json")

# 이미 번역된 데이터 로드 (이어하기 용)
translated = []
if os.path.exists(output_path):
    with open(output_path, 'r', encoding='utf-8') as f:
        translated = json.load(f)
    print(f"이미 번역된 데이터: {len(translated)}개, 이어서 진행")

start_idx = len(translated)

def translate_conversation(messages):
    """대화 메시지를 한국어로 번역"""
    # system 프롬프트와 대화를 합쳐서 번역 요청
    conversation_text = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            conversation_text += f"[SYSTEM]\n{content}\n\n"
        elif role == 'user':
            conversation_text += f"[CLIENT]\n{content}\n\n"
        elif role == 'assistant':
            conversation_text += f"[COUNSELOR]\n{content}\n\n"

    prompt = f"""아래 CBT 심리 상담 대화를 자연스러운 한국어로 번역해주세요.

[번역 규칙]
- 번역투를 절대 사용하지 마세요
- 한국어 상담사가 실제로 쓰는 자연스러운 구어체를 사용하세요
- "~하셨군요", "~느끼셨겠네요" 같은 한국어 상담 어투를 사용하세요
- 이름은 한국식으로 변경하세요 (예: Brooke → 지은)
- 문화적 맥락도 한국에 맞게 조정하세요
- [SYSTEM], [CLIENT], [COUNSELOR] 태그는 그대로 유지하세요

[원문]
{conversation_text}

[한국어 번역]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API 에러: {e}")
        return None


def parse_translated(translated_text, original_messages):
    """번역된 텍스트를 messages 형식으로 파싱"""
    new_messages = []
    current_role = None
    current_content = []

    for line in translated_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('[SYSTEM]'):
            if current_role:
                new_messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "system"
            current_content = []
        elif line.startswith('[CLIENT]'):
            if current_role:
                new_messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "user"
            current_content = []
        elif line.startswith('[COUNSELOR]'):
            if current_role:
                new_messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})
            current_role = "assistant"
            current_content = []
        else:
            current_content.append(line)

    if current_role:
        new_messages.append({"role": current_role, "content": '\n'.join(current_content).strip()})

    # 최소 3개 메시지 (system + user + assistant)
    if len(new_messages) >= 3:
        return {"messages": new_messages}
    else:
        return None


# ============ 번역 시작 ============
print(f"\n번역 시작! ({start_idx}번째부터)")
print(f"예상 비용: ~${len(train_data) * 0.0005:.1f}")
print("Ctrl+C로 중단해도 진행분은 저장됩니다.\n")

try:
    for i in range(start_idx, len(train_data)):
        item = train_data[i]

        result_text = translate_conversation(item['messages'])

        if result_text:
            parsed = parse_translated(result_text, item['messages'])
            if parsed:
                translated.append(parsed)

        # 진행 상황 출력
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{len(train_data)} (번역 완료: {len(translated)}개)")

        # 100개마다 중간 저장
        if (i + 1) % 100 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)
            print(f"  >> 중간 저장 완료 ({len(translated)}개)")

        time.sleep(0.3)  # 레이트 리밋 방지

except KeyboardInterrupt:
    print("\n\n중단됨! 진행분 저장 중...")

# 최종 저장
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(translated, f, ensure_ascii=False, indent=2)

print(f"\n최종 저장 완료: {len(translated)}개")
print(f"저장 경로: {output_path}")