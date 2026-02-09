import sys
from typing import List, Dict
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput
from ai_core.router.tone_router import route_tone

def format_history(history: List[Dict[str, str]], max_turns: int = 6) -> str:
    # 마지막 max_turns*2 메시지만 요약해 포함
    h = history[-max_turns*2:]
    lines = []
    for m in h:
        role = "사용자" if m["role"] == "user" else "상담사"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)

def main() -> None:
    orch = TurnOrchestrator()
    history: List[Dict[str, str]] = []

    print("멀티턴 상담 데모 시작 (종료: exit/quit).")
    while True:
        try:
            user_text = input("나(자신): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n상담을 종료합니다.")
            break
        if user_text.lower() in {"exit", "quit"}:
            print("상담을 종료합니다.")
            break
        if not user_text:
            continue

        # 1) 감정 집계
        ti = TurnInput(utterance=user_text)
        aggregated = orch.aggregate_emotion(ti)
        route = route_tone(aggregated)

        # 2) 히스토리 + 현재 감정/톤 반영 프롬프트 구성
        sys_prompt = "당신은 공감적이고 전문적인 상담 보조자입니다. 한국어로 답하세요."
        hist_block = format_history(history, max_turns=6)
        user_block = (
            f"대화 히스토리:\n{hist_block}\n\n"
            f"현재 사용자 발화: {user_text}\n"
            f"추정 감정 분포: {aggregated}\n"
            f"[tone={route['tone']}, strategy={route['strategy']}]에 맞춰 답변하세요."
        )

        # 3) LLM 호출
        reply = orch.llm.chat(
            system_prompt=sys_prompt,
            user_prompt=user_block,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

        print(f"상담사: {reply}\n")

        # 4) 히스토리 업데이트
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()