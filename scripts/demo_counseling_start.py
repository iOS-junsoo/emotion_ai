"""
상담 시작 기능 테스트 (Step 2, 3)

이 스크립트는 다음을 테스트합니다:
- Step 2: 초기 LLM 시작 문장 생성
- Step 3: 상담 세션 초기화 및 대화 구성
- 감정 기반 응답 스타일 프롬프팅
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def print_separator(char="=", length=80):
    """구분선 출력"""
    print(char * length)


def test_initial_greeting(llm_backend=None, gpu_id=3, load_in_8bit=False, model_name=None):
    """Step 2 테스트: 초기 인사말 생성"""
    print_separator()
    print("🧪 Step 2 테스트: 초기 인사말 생성")
    print_separator()
    print()
    
    orchestrator = TurnOrchestrator(
        enable_multiturn=True,
        llm_backend=llm_backend,
        gpu_id=gpu_id,
        load_in_8bit=load_in_8bit,
        model_name=model_name
    )
    
    test_cases = [
        {
            "text": "요즘 너무 힘들어요. 일도 잘 안 풀리고 스트레스가 심해요.",
            "emotion": "sad"
        },
        {
            "text": "정말 화가 나요! 왜 이런 일이 생기는지 모르겠어요.",
            "emotion": "angry"
        },
        {
            "text": "오늘 정말 좋은 일이 있었어요! 취업에 성공했어요.",
            "emotion": "happy"
        },
        {
            "text": "내일 중요한 발표가 있는데 너무 불안해요.",
            "emotion": "anxious"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"테스트 케이스 {i}:")
        print(f"  사용자 입력: {test_case['text']}")
        print(f"  감정: {test_case['emotion']}")
        print()
        
        try:
            greeting = orchestrator.generate_initial_greeting(
                user_initial_text=test_case['text'],
                user_emotion=test_case['emotion']
            )
            
            print(f"  상담사 인사말:")
            print(f"    {greeting}")
            print()
            print_separator("-")
            print()
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            print()
            print_separator("-")
            print()


def test_start_session(llm_backend=None, gpu_id=3, load_in_8bit=False, model_name=None):
    """Step 3 테스트: 세션 시작"""
    print_separator()
    print("🧪 Step 3 테스트: 세션 시작 및 초기 대화 구성")
    print_separator()
    print()
    
    orchestrator = TurnOrchestrator(
        enable_multiturn=True,
        llm_backend=llm_backend,
        gpu_id=gpu_id,
        load_in_8bit=load_in_8bit,
        model_name=model_name
    )
    
    # 세션 시작
    session_id = "test_session_001"
    user_text = "요즘 우울한 기분이 계속돼요. 뭘 해도 의미가 없는 것 같아요."
    user_emotion = "sad"
    
    print(f"📝 사용자 입력: {user_text}")
    print(f"😊 사용자 감정: {user_emotion}")
    print(f"🆔 세션 ID: {session_id}")
    print()
    
    try:
        result = orchestrator.start_session(
            session_id=session_id,
            user_initial_text=user_text,
            user_emotion=user_emotion
        )
        
        print("✅ 세션 시작 결과:")
        print(f"  메시지: {result['message']}")
        print(f"  감지된 감정: {result['detected_emotion']}")
        print(f"  응답 톤: {result['tone']}")
        print(f"  응답 전략: {result['strategy']}")
        print()
        print(f"💬 상담사 인사말:")
        print(f"  {result['greeting']}")
        print()
        
        # 세션 히스토리 확인
        history = orchestrator.session_manager.get_or_create_session(session_id)
        print(f"📚 대화 히스토리: {len(history.turns)}턴 저장됨")
        print()
        
        return session_id, orchestrator
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_multiturn_with_emotion_style(session_id: str, orchestrator: TurnOrchestrator):
    """감정 기반 응답 스타일 테스트"""
    print_separator()
    print("🧪 감정 기반 멀티턴 대화 테스트")
    print_separator()
    print()
    
    if not session_id or not orchestrator:
        print("⚠️  세션이 초기화되지 않았습니다.")
        return
    
    # 시나리오: sad → neutral → happy 감정 변화
    conversations = [
        ("네, 정말 힘들어요. 아침에 일어나는 것도 힘들고...", None),
        ("그래도 오늘 친구랑 이야기하니까 조금 나은 것 같아요.", None),
        ("감사합니다. 많이 도움이 됐어요. 조금씩 나아질 것 같아요.", None)
    ]
    
    for turn_num, (user_text, _) in enumerate(conversations, 2):
        print(f"턴 {turn_num}:")
        print(f"  👤 사용자: {user_text}")
        print()
        
        try:
            # 감정 추출 및 응답 생성
            ti = TurnInput(utterance=user_text)
            output = orchestrator.run(ti, session_id=session_id)
            
            print(f"  😊 감지된 감정: {max(output.emotion.aggregated.items(), key=lambda x: x[1])[0]}")
            print(f"  🎯 응답 톤: {output.meta['tone']}")
            print(f"  💬 상담사: {output.response}")
            print()
            print_separator("-")
            print()
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            print()
            print_separator("-")
            print()
    
    # 감정 변화 추이
    history = orchestrator.session_manager.get_or_create_session(session_id)
    trajectory = history.get_emotion_trajectory()
    
    print("📈 감정 변화 추이:")
    print(f"  {' → '.join(trajectory)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="상담 시작 기능 테스트")
    parser.add_argument(
        "--test",
        type=str,
        choices=["greeting", "session", "multiturn", "all"],
        default="all",
        help="실행할 테스트 (기본값: all)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default=None,
        help="LLM 백엔드 선택 (기본값: 자동)"
    )
    parser.add_argument("--gpu", type=int, default=3, help="GPU 번호 (transformers 모드)")
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="8비트 양자화 사용 (메모리 절약, transformers 모드)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="모델 이름 (기본: Qwen/Qwen2-7B-Instruct)"
    )
    
    args = parser.parse_args()
    
    print()
    print("="*80)
    print("🎬 상담 시작 기능 테스트 (Step 2, 3)")
    print("="*80)
    print()
    
    if args.backend == "vllm":
        print("⚠️  vLLM 모드: 서버가 실행 중이어야 합니다.")
        print("   실행 명령: bash scripts/run_vllm.sh")
    else:
        print(f"✅ Transformers 모드: GPU {args.gpu} 사용 (서버 불필요)")
        if args.load_in_8bit:
            print(f"   8비트 양자화: 활성화 (메모리 절약)")
        if args.model_name:
            print(f"   모델: {args.model_name}")
    print()
    
    session_id = None
    orchestrator = None
    
    try:
        if args.test in ["greeting", "all"]:
            test_initial_greeting(args.backend, args.gpu, args.load_in_8bit, args.model_name)
            print()
        
        if args.test in ["session", "multiturn", "all"]:
            session_id, orchestrator = test_start_session(args.backend, args.gpu, args.load_in_8bit, args.model_name)
            print()
        
        if args.test in ["multiturn", "all"]:
            if session_id and orchestrator:
                test_multiturn_with_emotion_style(session_id, orchestrator)
        
        print_separator()
        print("✅ 모든 테스트 완료!")
        print_separator()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
