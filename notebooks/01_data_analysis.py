from datasets import load_dataset
import json
import os
import re

# 데이터 로드
dataset = load_dataset("LangAGI-Lab/cactus")
train = dataset['train']

# ============ 1. 기초 통계 ============
print("=" * 50)
print("1. 기초 통계")
print("=" * 50)

# attitude 분포
from collections import Counter
attitudes = Counter(train['attitude'])
print(f"\nattitude 분포: {dict(attitudes)}")

# cbt_technique 종류
techniques = Counter(train['cbt_technique'])
print(f"\nCBT 기법 수: {len(techniques)}종")
print("상위 10개:")
for tech, count in techniques.most_common(10):
    print(f"  {tech}: {count}개")

# 대화 턴 수 분석
def count_turns(dialogue):
    counselor_turns = len(re.findall(r'Counselor:', dialogue))
    client_turns = len(re.findall(r'Client:', dialogue))
    return counselor_turns, client_turns

turn_counts = [count_turns(d) for d in train['dialogue']]
counselor_avg = sum(c for c, _ in turn_counts) / len(turn_counts)
client_avg = sum(c for _, c in turn_counts) / len(turn_counts)
total_avg = counselor_avg + client_avg

print(f"\n평균 턴 수: 총 {total_avg:.1f}턴 (상담사 {counselor_avg:.1f} + 내담자 {client_avg:.1f})")
print(f"최소 턴: {min(c+cl for c, cl in turn_counts)}")
print(f"최대 턴: {max(c+cl for c, cl in turn_counts)}")

# dialogue 길이 분포
lengths = [len(d) for d in train['dialogue']]
print(f"\n대화 길이: 평균 {sum(lengths)/len(lengths):.0f}자, 최소 {min(lengths)}자, 최대 {max(lengths)}자")

# ============ 2. Qwen Chat Format 변환 ============
print("\n" + "=" * 50)
print("2. Qwen Chat Format 변환")
print("=" * 50)

def parse_dialogue(dialogue):
    """Counselor/Client 턴을 파싱"""
    turns = []
    current_role = None
    current_text = []

    for line in dialogue.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('Counselor:'):
            if current_role:
                turns.append((current_role, ' '.join(current_text)))
            current_role = 'counselor'
            current_text = [line[len('Counselor:'):].strip()]
        elif line.startswith('Client:'):
            if current_role:
                turns.append((current_role, ' '.join(current_text)))
            current_role = 'client'
            current_text = [line[len('Client:'):].strip()]
        else:
            if current_text:
                current_text.append(line)

    if current_role:
        turns.append((current_role, ' '.join(current_text)))

    return turns

def convert_to_qwen_format(sample):
    """CACTUS 샘플을 Qwen chat format으로 변환"""
    system_prompt = (
        f"You are a professional CBT (Cognitive Behavioral Therapy) counselor. "
        f"Use the following client information and counseling plan to guide the session.\n\n"
        f"[Client Information]\n{sample['intake_form']}\n\n"
        f"[CBT Technique]\n{sample['cbt_technique']}\n\n"
        f"[Counseling Plan]\n{sample['cbt_plan']}"
    )

    turns = parse_dialogue(sample['dialogue'])
    messages = [{"role": "system", "content": system_prompt}]

    for role, text in turns:
        if role == 'client':
            messages.append({"role": "user", "content": text})
        elif role == 'counselor':
            messages.append({"role": "assistant", "content": text})

    return {"messages": messages}

# 변환 테스트
sample = train[0]
converted = convert_to_qwen_format(sample)
print(f"\n변환된 메시지 수: {len(converted['messages'])}개")
print(f"시스템 프롬프트 길이: {len(converted['messages'][0]['content'])}자")
print(f"\n첫 3개 메시지:")
for i, msg in enumerate(converted['messages'][:3]):
    role = msg['role']
    content = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
    print(f"  [{role}] {content}")

# ============ 3. 전체 데이터 변환 및 저장 ============
print("\n" + "=" * 50)
print("3. 전체 데이터 변환 및 저장")
print("=" * 50)

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

converted_data = []
skipped = 0

for i, sample in enumerate(train):
    try:
        converted = convert_to_qwen_format(sample)
        # 최소 3개 메시지 (system + user + assistant) 있는지 확인
        if len(converted['messages']) >= 3:
            converted_data.append(converted)
        else:
            skipped += 1
    except Exception as e:
        skipped += 1

# 저장
output_path = os.path.join(output_dir, "cactus_qwen_format.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print(f"변환 완료: {len(converted_data)}개")
print(f"스킵: {skipped}개")
print(f"저장 경로: {output_path}")

# ============ 4. 학습/검증 데이터 분할 ============
print("\n" + "=" * 50)
print("4. 학습/검증 데이터 분할")
print("=" * 50)

import random
random.seed(42)

# attitude와 cbt_technique 분포를 유지하면서 분할 (stratified split)
from collections import defaultdict

# attitude 기준 stratified split (90:10)
indices_by_attitude = defaultdict(list)
for i, sample in enumerate(train):
    indices_by_attitude[sample['attitude']].append(i)

train_indices = []
val_indices = []

for attitude, indices in indices_by_attitude.items():
    random.shuffle(indices)
    split_point = int(len(indices) * 0.9)
    train_indices.extend(indices[:split_point])
    val_indices.extend(indices[split_point:])

random.shuffle(train_indices)
random.shuffle(val_indices)

print(f"전체: {len(train)}개")
print(f"학습: {len(train_indices)}개 (90%)")
print(f"검증: {len(val_indices)}개 (10%)")

# 분할 후 attitude 분포 확인
train_attitudes = Counter(train[i]['attitude'] for i in train_indices)
val_attitudes = Counter(train[i]['attitude'] for i in val_indices)
print(f"\n학습 attitude 분포: {dict(train_attitudes)}")
print(f"검증 attitude 분포: {dict(val_attitudes)}")

# 분할된 데이터 저장
train_data = [converted_data[i] for i in train_indices]
val_data = [converted_data[i] for i in val_indices]

train_path = os.path.join(output_dir, "cactus_train.json")
val_path = os.path.join(output_dir, "cactus_val.json")

with open(train_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)
with open(val_path, 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

print(f"\n저장 완료:")
print(f"  학습: {train_path}")
print(f"  검증: {val_path}")