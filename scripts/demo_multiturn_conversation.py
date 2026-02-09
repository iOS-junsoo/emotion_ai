"""
멀티턴 감정 기반 대화 데모

이 스크립트는 다음을 시연합니다:
1. 멀티턴 대화에서 이전 대화 기억
2. 감정 변화 추적
3. 감정에 맞는 응답 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput
from typing import List


def print_separator(char="=", length=80):
    """구분선 출력"""
    print(char * length)


def print_turn_info(turn_num: int, utterance: str, output):
    """턴 정보 출력"""
    print_separator()
    print(f"📍 턴 {turn_num}")
    print_separator("-")
    print(f"👤 사용자: {utterance}")
    print()
    
    # 감정 분포
    emotion = output.emotion.aggregated
    top_emotion = max(emotion.items(), key=lambda x: x[1])
    print(f"😊 감정 분석:")
    print(f"   주요 감정: {top_emotion[0]} ({top_emotion[1]*100:.1f}%)")
    print(f"   전체 분포: {', '.join([f'{k}: {v*100:.1f}%' for k, v in sorted(emotion.items(), key=lambda x: -x[1])[:3]])}")
    print()
    
    # 메타 정보
    print(f"🎯 응답 전략:")
    print(f"   톤: {output.meta['tone']}")
    print(f"   전략: {output.meta['strategy']}")
    print(f"   누적 턴 수: {output.meta.get('turn_count', 1)}")
    print()
    
    # LLM 응답
    print(f"🤖 상담사: {output.response}")
    print()


def demo_multiturn_conversation(
    conversations: List[str],
    session_id: str = "demo_session",
    enable_multiturn: bool = True
):
    """
    멀티턴 대화 데모 실행
    
    Args:
        conversations: 대화 목록
        session_id: 세션 ID
        enable_multiturn: 멀티턴 활성화 여부
    """
    print_separator("=", 80)
    print("🎭 멀티턴 감정 기반 대화 데모")
    print_separator("=", 80)
    print(f"세션 ID: {session_id}")
    print(f"멀티턴 모드: {'✅ 활성화' if enable_multiturn else '❌ 비활성화'}")
    print(f"총 턴 수: {len(conversations)}")
    print()
    
    # Orchestrator 초기화
    orch = TurnOrchestrator(enable_multiturn=enable_multiturn)
    
    # 대화 진행
    for turn_num, utterance in enumerate(conversations, 1):
        ti = TurnInput(utterance=utterance)
        
        # 멀티턴 모드에 따라 session_id 전달
        if enable_multiturn:
            output = orch.run(ti, session_id=session_id)
        else:
            output = orch.run(ti)
        
        print_turn_info(turn_num, utterance, output)
    
    # 감정 변화 추이 출력 (멀티턴 모드일 때만)
    if enable_multiturn:
        history = orch.session_manager.get_or_create_session(session_id)
        emotion_trajectory = history.get_emotion_trajectory()
        
        print_separator()
        print("📈 감정 변화 추이")
        print_separator("-")
        print(" → ".join(emotion_trajectory))
        print()
    
    print_separator()
    print("✅ 대화 완료!")
    print_separator()


def main():
    """메인 함수"""
    
    # 시나리오 1: 슬픔 → 분노 → 수용 → 희망
    print("\n🎬 시나리오 1: 감정의 변화 추적\n")
    
    conversations_1 = [
        "요즘 너무 힘들어요. 매일 우울하고 아무것도 하기 싫어요.",
        "일도 잘 안 풀리고, 사람들도 저를 이해 못 해요. 화가 나요.",
        "그래도 당신이 이야기를 들어주니까 조금 나은 것 같아요.",
        "앞으로 어떻게 해야 할지 조금 보이는 것 같아요. 감사합니다."
    ]
    
    demo_multiturn_conversation(
        conversations=conversations_1,
        session_id="scenario_1",
        enable_multiturn=True
    )
    
    # 시나리오 2: 불안 → 안정
    print("\n" + "="*80)
    print("\n🎬 시나리오 2: 불안에서 안정으로\n")
    
    conversations_2 = [
        "내일 중요한 발표가 있는데 너무 불안해요. 잘할 수 있을까요?",
        "실수할까 봐 무서워요. 계속 나쁜 생각만 들어요.",
        "조언 감사합니다. 준비를 더 해봐야겠어요.",
        "많이 진정됐어요. 이제 자신감이 생기네요."
    ]
    
    demo_multiturn_conversation(
        conversations=conversations_2,
        session_id="scenario_2",
        enable_multiturn=True
    )
    
    # 시나리오 3: 멀티턴 비활성화 비교 (같은 대화)
    print("\n" + "="*80)
    print("\n🎬 시나리오 3: 단일 턴 모드 (비교용)\n")
    print("⚠️  이전 대화를 기억하지 못합니다\n")
    
    demo_multiturn_conversation(
        conversations=conversations_1[:2],  # 처음 2턴만
        session_id="scenario_3",
        enable_multiturn=False  # 멀티턴 비활성화
    )


if __name__ == "__main__":
    main()
