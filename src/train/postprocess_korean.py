"""
한국어 CACTUS 번역 데이터 후처리 스크립트
==========================================
번역된 cactus_korean.json을 학습 가능한 상태로 정제

후처리 항목:
1. 구조 검증 (messages 형식, role 순서)
2. 이름 완전 제거 ("안녕하세요, 지은" → "안녕하세요")
3. system 프롬프트 내 이름 필드 비우기 (이름: 지은 → 이름:)
4. 영어 잔류 검출
5. 빈 content / 깨진 데이터 필터링
6. 번역투 표현 검출 및 통계
7. 중복 제거
8. 학습/검증 분할
9. 저장

사용법:
    python postprocess_korean.py
    python postprocess_korean.py --input path/to/cactus_korean.json
    python postprocess_korean.py --dry-run  # 통계만 보기
"""

import json
import re
import os
import argparse
import random
from collections import Counter


# ============================================================
# 설정
# ============================================================

# 제거할 한국 이름 목록 (GPT-4o-mini가 번역 시 사용하는 이름들)
# ⚠️ 일반 단어와 겹치는 이름은 제외! (지원, 미래, 보라, 하늘 등)
#    "지원" → "사회적 지원", "지원 시스템" 등 일반 단어로 6,989회 오탐
#    "미래" → "미래 계획", "미래에 대한" 등 일반 단어로 1,955회 오탐
#    "보라" → "보라색" 등은 적지만 "살펴보라", "해보라" 등 동사형 오탐 가능
#    "하늘" → "하늘이 맑다" 등 일반 단어
#    "은지" → "~인지", "~은지" 등 어미와 충돌 가능성
KOREAN_NAMES = [
    "지은", "민수", "수진", "현우", "서연", "지호", "예진", "도현",
    "하은", "준서", "유진", "태현", "소연", "민지", "성준",
    "재현", "수빈", "동현", "선영", "영호", "정민",
    "세진", "다은", "시우", "지민", "소희", "윤서",
    "민호", "서준", "예림", "태민", "채원", "승현", "나연", "진우",
]

# 일반 단어와 겹치는 이름 → 호칭 패턴일 때만 제거 (이름+님/씨/아/야)
AMBIGUOUS_NAMES = ["지원", "미래", "보라", "하늘", "은지"]

# 영어 이름 (system 프롬프트에 남아있을 수 있음)
ENGLISH_NAMES = [
    "Brooke", "Alex", "Jordan", "Taylor", "Casey", "Morgan", "Riley",
    "Sam", "Jamie", "Chris", "Sarah", "Emily", "Michael", "David",
    "John", "Lisa", "Amy", "Robert", "Jessica", "Daniel", "Emma",
    "Olivia", "Sophia", "Liam", "Noah", "James", "William", "Benjamin",
    "Lucas", "Henry", "Ethan", "Mason", "Logan", "Jack", "Ryan",
]

# 번역투 검출 패턴 (치환 후에도 남아있는지 확인용)
TRANSLATIONESE_DETECT = [
    (r"당신[은의에을이가도]", "\"당신\" (번역투)"),
    (r"그것은\s", "\"그것은\" (번역투)"),
    (r"나는\s.*느낀다", "\"나는~느낀다\" (번역투)"),
    (r"우리가\s할\s수\s있는", "\"우리가 할 수 있는\" (번역투)"),
    (r"것이\s중요합니다", "\"것이 중요합니다\" (번역투)"),
]

# ============================================================
# 번역투 자동 치환 규칙
# ============================================================
# "당신"이 12,370회로 가장 큰 문제.
# 한국어 상담에서는 주어를 생략하거나 "~분" 등을 사용함.
# 규칙: 구체적 패턴 → 치환, 나머지 "당신" → 삭제(주어 생략)

TRANSLATIONESE_FIXES = [
    # ── "당신" 패턴별 치환 (구체적인 것 먼저) ──

    # 소유격: "당신의 감정" → "그 감정" / "느끼시는 감정"
    (re.compile(r"당신의\s*(이야기|말씀)"), r"하신 \1"),
    (re.compile(r"당신의\s*(감정|기분|마음|생각|느낌)"), r"그 \1"),
    (re.compile(r"당신의\s*(경험|상황|문제|고민|걱정)"), r"그 \1"),
    (re.compile(r"당신의\s*(노력|용기|결정|선택)"), r"그런 \1"),
    (re.compile(r"당신의\s*(삶|인생|일상|생활)"), r"일상"),
    (re.compile(r"당신의"), ""),  # 나머지 소유격 → 삭제

    # 주격: "당신이 느끼는" → "느끼시는"
    (re.compile(r"당신이\s*(느끼|생각하|겪|경험하)"), r"\1"),
    (re.compile(r"당신이\s*(할\s*수)"), r"\1"),
    (re.compile(r"당신이\s"), ""),  # 나머지 주격 → 삭제

    # 목적격: "당신을 지지하기" → "지지하기"
    (re.compile(r"당신을\s"), ""),

    # 여격/처격: "당신에게" → "" (삭제)
    (re.compile(r"당신에게\s"), ""),

    # 보조사: "당신도", "당신은"
    (re.compile(r"당신도\s"), ""),
    (re.compile(r"당신은\s"), ""),

    # 남은 "당신" 단독 → 삭제
    (re.compile(r"당신"), ""),

    # ── "그것은" 패턴 ──
    (re.compile(r"그것은\s"), ""),
    (re.compile(r"그것이\s"), ""),
    (re.compile(r"그것을\s"), "그걸 "),

    # ── 기타 번역투 ──
    (re.compile(r"것이\s중요합니다"), "게 중요해요"),
    (re.compile(r"것이\s필요합니다"), "게 필요해요"),
    (re.compile(r"것이\s도움이\s됩니다"), "게 도움이 돼요"),
    (re.compile(r"할\s수\s있습니다"), "할 수 있어요"),
    (re.compile(r"하는\s것입니다"), "하는 거예요"),
    (re.compile(r"있을\s수\s있습니다"), "있을 수 있어요"),
    (re.compile(r"없을\s수\s있습니다"), "없을 수 있어요"),
]


def fix_translationese(text):
    """번역투 자동 치환"""
    cleaned = text

    for pattern, replacement in TRANSLATIONESE_FIXES:
        cleaned = pattern.sub(replacement, cleaned)

    # 치환 후 정리: 연속 공백, 문장부호 앞 공백
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.!?,])", r"\1", cleaned)
    cleaned = cleaned.strip()

    return cleaned

# 영어 단어 잔류 검출 (system 프롬프트 제외, 대화 부분만)
ENGLISH_RESIDUE_PATTERN = re.compile(
    r"\b(?!LGBTQ|CBT|PTSD|ADHD|OCD|SNS|IT|AI|HR|CEO|MBA|PhD|"
    r"OK|ok|Oh|oh|Hmm|hmm|Yes|No|MBTI|"
    r"stakes|email|stress|trauma|burnout)"
    r"[A-Za-z]{4,}\b"
)


# ============================================================
# 이름 제거 함수
# ============================================================

def build_name_patterns():
    """이름 제거용 정규식 패턴 빌드"""
    patterns = []

    for name in KOREAN_NAMES:
        # "지은님", "지은씨", "지은아", "지은야" → 전체 제거
        patterns.append((re.compile(rf"{name}(님|씨|아|야)"), ""))

        # ", 지은." / ", 지은!" → "," (쉼표 유지)
        patterns.append((re.compile(rf",\s*{name}[.!?]?(?=\s|$)"), ","))

        # "안녕하세요, 지은" → "안녕하세요,"
        patterns.append((re.compile(rf",\s*{name}\b"), ","))

        # "지은에게" 등 조사 붙은 경우 → 제거
        patterns.append((re.compile(rf"{name}(에게|이가|이의|이는|이를|이와|이랑|한테|과|와|의|가|는|를|도)"), ""))

        # 단독 "지은" → 제거
        patterns.append((re.compile(rf"\b{name}\b"), ""))

    for name in ENGLISH_NAMES:
        patterns.append((re.compile(rf"\b{name}\b", re.IGNORECASE), ""))

    return patterns


NAME_PATTERNS = build_name_patterns()


def remove_names_from_text(text):
    """텍스트에서 이름 완전 제거"""
    cleaned = text

    for pattern, replacement in NAME_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # 정리
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"^\s*,\s*", "", cleaned)
    cleaned = re.sub(r"\s+([.!?,])", r"\1", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def remove_names_from_system(text):
    """system 프롬프트에서 이름 필드 비우기 + 본문 이름 제거"""
    cleaned = text

    # "이름: 지은" / "이름:\n지은" → "이름:"
    for name in KOREAN_NAMES:
        cleaned = re.sub(rf"(이름\s*:\s*)\n?{name}", r"\1", cleaned)

    for name in ENGLISH_NAMES:
        cleaned = re.sub(rf"(이름\s*:\s*)\n?{name}", r"\1", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"(Name\s*:\s*)\n?{name}", r"\1", cleaned, flags=re.IGNORECASE)

    # 본문 내 이름도 제거
    cleaned = remove_names_from_text(cleaned)

    return cleaned


# ============================================================
# 후처리 함수
# ============================================================

def validate_structure(item, idx):
    """구조 검증"""
    issues = []

    if "messages" not in item:
        return ["messages 키 없음"], True

    messages = item["messages"]

    if not isinstance(messages, list) or len(messages) < 3:
        return [f"메시지 수 부족: {len(messages) if isinstance(messages, list) else 'N/A'}"], True

    if messages[0]["role"] != "system":
        issues.append(f"첫 메시지가 system이 아님: {messages[0]['role']}")

    valid_roles = {"system", "user", "assistant"}
    for i, msg in enumerate(messages):
        if msg.get("role") not in valid_roles:
            issues.append(f"msg[{i}] 유효하지 않은 role: {msg.get('role')}")
            return issues, True

        if not msg.get("content") or not msg["content"].strip():
            issues.append(f"msg[{i}] 빈 content (role={msg['role']})")

    # user/assistant 교대 확인 (system 이후)
    non_system = [m for m in messages if m["role"] != "system"]
    for i in range(1, len(non_system)):
        if non_system[i]["role"] == non_system[i - 1]["role"]:
            issues.append(f"연속 같은 role: {non_system[i]['role']} (위치 {i})")

    return issues, False


def clean_content(text, is_system=False):
    """content 정제"""
    cleaned = text.strip()

    # 태그 잔류 제거
    cleaned = re.sub(r"\[SYSTEM\]|\[CLIENT\]|\[COUNSELOR\]", "", cleaned).strip()

    # 앞뒤 따옴표 제거 (GPT가 가끔 붙임)
    if cleaned.startswith('"') and cleaned.endswith('"') and cleaned.count('"') == 2:
        cleaned = cleaned[1:-1].strip()

    # 이름 제거
    if is_system:
        cleaned = remove_names_from_system(cleaned)
    else:
        cleaned = remove_names_from_text(cleaned)

    # 번역투 치환 (대화 부분만, system 프롬프트는 원본 유지)
    if not is_system:
        cleaned = fix_translationese(cleaned)

    # 최종 정리
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned


def detect_issues_in_conversation(messages):
    """대화 부분에서 품질 이슈 검출 (번역투 치환 후 잔류 확인)"""
    issues = []

    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            continue

        content = msg["content"]

        # 번역투 잔류 검출 (치환 후에도 남아있는지)
        for pattern, label in TRANSLATIONESE_DETECT:
            if re.search(pattern, content):
                issues.append(f"번역투 잔류: {label}")
                break

        # 영어 잔류
        eng_matches = ENGLISH_RESIDUE_PATTERN.findall(content)
        if eng_matches:
            issues.append(f"영어 잔류: {', '.join(eng_matches[:3])}")

    return issues


def get_content_hash(item):
    """중복 검출용 해시"""
    user_texts = [
        m["content"][:80] for m in item["messages"] if m["role"] == "user"
    ][:2]
    return "|".join(user_texts)


def postprocess_item(item, idx):
    """단일 아이템 후처리"""
    issues, fatal = validate_structure(item, idx)
    if fatal:
        return None, issues

    cleaned_messages = []
    for msg in item["messages"]:
        is_system = (msg["role"] == "system")
        cleaned_content = clean_content(msg["content"], is_system=is_system)

        if not cleaned_content:
            issues.append(f"정제 후 빈 content (role={msg['role']})")
            continue

        cleaned_messages.append({
            "role": msg["role"],
            "content": cleaned_content,
        })

    if len(cleaned_messages) < 3:
        issues.append("정제 후 메시지 수 부족")
        return None, issues

    conv_issues = detect_issues_in_conversation(cleaned_messages)
    issues.extend(conv_issues)

    return {"messages": cleaned_messages}, issues


# ============================================================
# 통계 출력
# ============================================================

def print_stats(raw_data, cleaned, dropped, all_issues, name_changes, translationese_fixes):
    """후처리 통계"""
    print(f"\n{'=' * 60}")
    print("후처리 결과")
    print(f"{'=' * 60}")
    print(f"  원본: {len(raw_data)}개")
    print(f"  정제 완료: {len(cleaned)}개")
    print(f"  제거됨: {len(dropped)}개")
    print(f"  이름 제거된 데이터: {name_changes}개")
    print(f"  번역투 치환된 데이터: {translationese_fixes}개")

    if dropped:
        print(f"\n  --- 제거 사유 ---")
        drop_reasons = Counter()
        for idx, issues in dropped:
            for issue in issues:
                if not issue.startswith("번역투") and not issue.startswith("영어 잔류"):
                    drop_reasons[issue] += 1
        for reason, count in drop_reasons.most_common():
            print(f"    {reason}: {count}개")

    issue_stats = Counter()
    for idx, issues in all_issues:
        for issue in issues:
            if issue.startswith("번역투:") or issue.startswith("영어 잔류:"):
                issue_stats[issue] += 1

    if issue_stats:
        print(f"\n  --- 품질 이슈 (참고용, 제거하지 않음) ---")
        for issue, count in issue_stats.most_common(15):
            print(f"    {issue}: {count}개")

    if cleaned:
        msg_counts = [len(item["messages"]) for item in cleaned]
        turn_counts = [
            sum(1 for m in item["messages"] if m["role"] == "user")
            for item in cleaned
        ]
        print(f"\n  --- 데이터 통계 ---")
        print(f"    메시지 수: 평균 {sum(msg_counts)/len(msg_counts):.1f}, "
              f"최소 {min(msg_counts)}, 최대 {max(msg_counts)}")
        print(f"    턴 수(user): 평균 {sum(turn_counts)/len(turn_counts):.1f}, "
              f"최소 {min(turn_counts)}, 최대 {max(turn_counts)}")


def print_sample(cleaned, raw_data):
    """정제 전후 샘플 비교"""
    if not cleaned or not raw_data:
        return

    print(f"\n{'=' * 60}")
    print("샘플 비교 (정제 전 → 후)")
    print(f"{'=' * 60}")

    sample_idx = 0
    for i, item in enumerate(raw_data):
        text = " ".join(m.get("content", "") for m in item.get("messages", []))
        if any(name in text for name in KOREAN_NAMES[:5]):
            sample_idx = i
            break

    if sample_idx < len(raw_data):
        raw_msgs = raw_data[sample_idx]["messages"]
        print(f"\n  [원본]")
        for j in range(1, min(4, len(raw_msgs))):
            print(f"    [{raw_msgs[j]['role']}] {raw_msgs[j]['content'][:100]}")

    if cleaned:
        clean_msgs = cleaned[min(sample_idx, len(cleaned) - 1)]["messages"]
        print(f"\n  [정제 후]")
        for j in range(1, min(4, len(clean_msgs))):
            print(f"    [{clean_msgs[j]['role']}] {clean_msgs[j]['content'][:100]}")


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="한국어 CACTUS 번역 데이터 후처리")
    parser.add_argument(
        "--input", type=str,
        default="data/processed/cactus_korean/cactus_korean.json",
        help="번역된 데이터 경로",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/processed/cactus_korean",
        help="출력 디렉토리",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="검증 데이터 비율")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--dry-run", action="store_true", help="통계만 출력 (파일 저장 안 함)")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("한국어 CACTUS 번역 데이터 후처리")
    print("=" * 60)

    if not os.path.exists(args.input):
        print(f"\n[에러] 파일을 찾을 수 없습니다: {args.input}")
        print("  --input 으로 경로를 지정해주세요.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"\n원본 데이터: {len(raw_data)}개")

    # ── 1단계: 사용된 이름 탐지 ──
    print(f"\n[1/4] 데이터에서 사용된 이름 탐지 중...")
    found_names = Counter()
    for item in raw_data:
        all_text = " ".join(m.get("content", "") for m in item.get("messages", []))
        for name in KOREAN_NAMES:
            count = all_text.count(name)
            if count > 0:
                found_names[name] += count

    if found_names:
        print(f"  발견된 이름 ({len(found_names)}종류):")
        for name, count in found_names.most_common(10):
            print(f"    \"{name}\": {count}회")
        if len(found_names) > 10:
            print(f"    ... 외 {len(found_names) - 10}개")
    else:
        print("  한국 이름 발견되지 않음")

    # ── 2단계: 후처리 ──
    print(f"\n[2/4] 후처리 진행 중...")
    cleaned = []
    dropped = []
    all_issues = []
    name_changes = 0
    translationese_fixes = 0
    seen_hashes = set()

    for idx, item in enumerate(raw_data):
        original_text = " ".join(m.get("content", "") for m in item.get("messages", []))

        result, issues = postprocess_item(item, idx)

        if issues:
            all_issues.append((idx, issues))

        if result is None:
            dropped.append((idx, issues))
            continue

        content_hash = get_content_hash(result)
        if content_hash in seen_hashes:
            dropped.append((idx, ["중복 데이터"]))
            continue
        seen_hashes.add(content_hash)

        cleaned_text = " ".join(m["content"] for m in result["messages"])
        if any(name in original_text for name in KOREAN_NAMES) and \
           not any(name in cleaned_text for name in KOREAN_NAMES):
            name_changes += 1

        # 번역투 치환 감지
        if "당신" in original_text and "당신" not in cleaned_text:
            translationese_fixes += 1
        elif "그것은" in original_text and "그것은" not in cleaned_text:
            translationese_fixes += 1

        cleaned.append(result)

        if (idx + 1) % 500 == 0:
            print(f"  진행: {idx + 1}/{len(raw_data)}")

    # ── 3단계: 통계 ──
    print(f"\n[3/4] 통계 분석...")
    print_stats(raw_data, cleaned, dropped, all_issues, name_changes, translationese_fixes)
    print_sample(cleaned, raw_data)

    if args.dry_run:
        print(f"\n[dry-run 모드] 파일 저장을 건너뜁니다.")
        return

    # ── 4단계: 저장 ──
    print(f"\n[4/4] 저장 중...")
    os.makedirs(args.output_dir, exist_ok=True)

    random.shuffle(cleaned)
    val_size = int(len(cleaned) * args.val_ratio)
    val_data = cleaned[:val_size]
    train_data = cleaned[val_size:]

    all_path = os.path.join(args.output_dir, "cactus_korean_clean.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    train_path = os.path.join(args.output_dir, "cactus_korean_train.json")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    val_path = os.path.join(args.output_dir, "cactus_korean_val.json")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    if dropped:
        log_path = os.path.join(args.output_dir, "dropped_log.json")
        log_entries = [{"index": idx, "issues": issues} for idx, issues in dropped]
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("저장 완료!")
    print(f"{'=' * 60}")
    print(f"  전체 (정제): {all_path} ({len(cleaned)}개)")
    print(f"  학습: {train_path} ({len(train_data)}개)")
    print(f"  검증: {val_path} ({len(val_data)}개)")
    if dropped:
        print(f"  제거 로그: {os.path.join(args.output_dir, 'dropped_log.json')} ({len(dropped)}개)")
    print(f"\n  → train_base.py에서 경로만 바꿔서 재학습하면 됩니다!")


if __name__ == "__main__":
    main()