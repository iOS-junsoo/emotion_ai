"""
GPU 통합 테스트: LoRA 스위칭 + 히스토리 요약 + LLM 응답
위치: src/pipeline/test_integration.py

테스트 시나리오:
  Step 1 (공감 형성) → 3턴 진행 → 스텝 전환 (LLM 요약 생성)
  Step 2 (문제 탐색) → 2턴 진행 → 감정 변화로 LoRA 전환
  → 이전 스텝 요약이 컨텍스트에 포함되는지 확인
"""

import os
import sys
import time

# 경로 설정 (src/pipeline/ 기준)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

from lora_switcher import LoRASwitcher
from history_manager import HistoryManager


def main():
    # ── 1. LoRA Switcher 초기화 ──
    print("=" * 60)
    print("GPU 통합 테스트: LoRA + HistoryManager")
    print("=" * 60)

    switcher = LoRASwitcher(
        base_model_name="Qwen/Qwen2.5-3B-Instruct",
        cbt_adapter_path=os.path.join(PROJECT_DIR, "models", "base", "cbt-counselor"),
        lora_dir=os.path.join(PROJECT_DIR, "models", "lora"),
    )
    switcher.load_all()

    # ── 2. HistoryManager 초기화 (LLM 연결) ──
    history = HistoryManager(max_recent_turns=3, lora_switcher=switcher)

    # ── 3. Step 1: 공감 형성 ──
    print("\n" + "=" * 60)
    print("Step 1: 공감 형성")
    print("=" * 60)

    switcher.switch("angry")  # 초기 감정: 분노

    # 턴 1: 상담사 첫 발화
    system_prompt = (
        "당신은 CBT 기반 심리 상담사입니다. "
        "내담자가 '직장 스트레스' 주제로 상담을 시작했습니다. "
        "내담자의 현재 감정: angry. "
        "먼저 내담자의 이야기를 충분히 경청하고 공감해주세요. "
        "짧고 자연스럽게 말하세요. 3~4문장을 넘기지 마세요."
    )
    messages = history.build_context_messages(system_prompt)
    # 첫 발화는 user 입력 없이 생성
    messages.append({"role": "user", "content": "안녕하세요. 상담 시작할게요."})
    opening = switcher.generate(messages, max_new_tokens=150)
    history.add_assistant_message(opening)
    print(f"\n[상담사] {opening}")

    # 턴 2: 사용자 발화 + 응답
    user_text = "상사가 제 성과를 자기 것처럼 가져가서 정말 화가 나요."
    history.add_user_message(user_text)
    print(f"\n[내담자] {user_text}")

    messages = history.build_context_messages(system_prompt)
    response = switcher.generate(messages, max_new_tokens=150)
    history.add_assistant_message(response)
    print(f"[상담사] {response}")

    # 턴 3
    user_text = "매일 야근하는데 인정도 안 해주고, 진짜 지쳤어요."
    history.add_user_message(user_text)
    print(f"\n[내담자] {user_text}")

    messages = history.build_context_messages(system_prompt)
    response = switcher.generate(messages, max_new_tokens=150)
    history.add_assistant_message(response)
    print(f"[상담사] {response}")

    # ── 4. 스텝 전환: Step 1 → Step 2 ──
    print("\n" + "=" * 60)
    print("스텝 전환: Step 1 → Step 2 (LLM 요약 생성)")
    print("=" * 60)

    start = time.time()
    history.on_step_transition(1, "공감 형성")
    elapsed = time.time() - start

    print(f"\n  요약 생성 시간: {elapsed:.1f}초")
    print(f"  Step 1 요약: {history.step_summaries[1]['summary']}")

    # ── 5. Step 2: 문제 탐색 (감정 변화: angry → sad) ──
    print("\n" + "=" * 60)
    print("Step 2: 문제 탐색 (감정: angry → sad)")
    print("=" * 60)

    # 감정 변화 → LoRA 전환
    switcher.switch("sad")

    system_prompt_2 = (
        "당신은 CBT 기반 심리 상담사입니다. "
        "내담자가 '직장 스트레스' 주제로 상담 중입니다. "
        "내담자의 현재 감정: sad. "
        "이제 내담자의 구체적인 상황, 감정, 생각을 탐색하세요. "
        "짧고 자연스럽게 말하세요. 3~4문장을 넘기지 마세요."
    )

    # Step 2 첫 발화 - 이전 스텝 요약이 포함되어야 함
    messages = history.build_context_messages(system_prompt_2)

    # 이전 스텝 요약이 시스템 프롬프트에 포함됐는지 확인
    has_summary = "이전 상담 요약" in messages[0]["content"]
    print(f"\n  이전 스텝 요약 포함 여부: {has_summary}")
    if has_summary:
        # 시스템 프롬프트에서 요약 부분만 출력
        content = messages[0]["content"]
        idx = content.find("[이전 상담 요약]")
        if idx >= 0:
            print(f"  {content[idx:idx+200]}...")

    # 첫 발화 생성
    messages.append({"role": "user", "content": "네, 계속 이야기할게요."})
    opening = switcher.generate(messages, max_new_tokens=150)
    history.add_assistant_message(opening)
    print(f"\n[상담사 - sad LoRA] {opening}")

    # 턴 2
    user_text = "솔직히... 제가 부족한 건가 싶기도 해요. 무능한 것 같아요."
    history.add_user_message(user_text)
    print(f"\n[내담자] {user_text}")

    messages = history.build_context_messages(system_prompt_2)
    response = switcher.generate(messages, max_new_tokens=150)
    history.add_assistant_message(response)
    print(f"[상담사 - sad LoRA] {response}")

    # ── 6. 최종 상태 ──
    print("\n" + "=" * 60)
    print("최종 상태")
    print("=" * 60)

    status = history.get_status()
    print(f"  완료 스텝: {status['completed_steps']}")
    print(f"  현재 스텝 턴: {status['current_step_turns']}")
    print(f"  전체 턴: {status['total_turns']}")

    print(f"\n  스텝별 요약:")
    for num, info in history.get_step_summaries().items():
        print(f"    Step {num} ({info['step_name']}): {info['summary'][:80]}...")

    print(f"\n  LoRA 상태: {switcher.get_status()}")
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
