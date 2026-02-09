"""
간단한 상담 데모 (Step 2, 3 포함)

사용자 친화적인 CLI 기반 상담 시뮬레이션
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def print_header():
    """헤더 출력"""
    print("\n" + "="*80)
    print("💬 AI 감정 상담 시스템")
    print("="*80)
    print()
    print("이 시스템은 당신의 감정을 이해하고 공감적으로 응답합니다.")
    print("종료하려면 '종료', 'exit', 'quit'를 입력하세요.")
    print()


def select_emotion():
    """초기 감정 선택"""
    print("현재 당신의 감정은 어떤가요?")
    print()
    emotions = {
        "1": ("sad", "슬픔 😢"),
        "2": ("angry", "분노 😠"),
        "3": ("happy", "기쁨 😊"),
        "4": ("anxious", "불안 😰"),
        "5": ("fear", "두려움 😨"),
        "6": ("neutral", "평온 😐"),
        "7": (None, "자동 감지 (텍스트 분석)")
    }
    
    for key, (_, label) in emotions.items():
        print(f"  {key}. {label}")
    print()
    
    while True:
        choice = input("선택 (1-7): ").strip()
        if choice in emotions:
            emotion, label = emotions[choice]
            print(f"\n✓ '{label}' 선택됨\n")
            return emotion
        print("올바른 번호를 입력해주세요.")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI 감정 상담 시스템")
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
    
    print_header()
    
    # 시스템 초기화
    print("⚙️  시스템 초기화 중...")
    if args.backend == "vllm":
        print("   vLLM 서버 모드 (서버가 실행 중이어야 합니다)")
    else:
        print(f"   Transformers 모드 (로컬 GPU {args.gpu} 사용)")
        if args.load_in_8bit:
            print(f"   8비트 양자화: 활성화 (메모리 절약)")
        if args.model_name:
            print(f"   모델: {args.model_name}")
    print()
    
    try:
        orchestrator = TurnOrchestrator(
            enable_multiturn=True,
            llm_backend=args.backend,
            gpu_id=args.gpu,
            load_in_8bit=args.load_in_8bit,
            model_name=args.model_name
        )
        print("✅ 시스템 초기화 완료!\n")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("\n해결 방법:")
        if args.backend == "vllm":
            print("  1. vLLM 서버를 실행하세요:")
            print("     bash scripts/run_vllm.sh")
            print("  2. 또는 Transformers 모드를 사용하세요:")
            print("     python scripts/simple_counseling_demo.py --backend transformers --gpu 3")
        else:
            print("  Transformers 설치 확인:")
            print("     pip install transformers torch")
        return 1
    
    # Step 1: 초기 감정 선택
    initial_emotion = select_emotion()
    
    # Step 2: 초기 상담 내용 입력
    print("="*80)
    print("무엇을 이야기하고 싶으신가요?")
    print("(여러 줄 입력 가능, 빈 줄로 Enter 시 완료)")
    print("="*80)
    print()
    
    initial_text_lines = []
    while True:
        line = input()
        if not line:
            break
        initial_text_lines.append(line)
    
    if not initial_text_lines:
        print("\n텍스트가 입력되지 않았습니다. 종료합니다.")
        return 0
    
    initial_text = " ".join(initial_text_lines)
    print()
    print(f"📝 입력하신 내용:")
    print(f"   {initial_text[:100]}{'...' if len(initial_text) > 100 else ''}")
    print()
    
    # Step 3: 세션 시작
    print("⏳ 상담을 시작하는 중...")
    print()
    
    session_id = "cli_session_001"
    
    try:
        result = orchestrator.start_session(
            session_id=session_id,
            user_initial_text=initial_text,
            user_emotion=initial_emotion
        )
        
        print("="*80)
        print("💬 상담사:")
        print("="*80)
        print()
        print(f"  {result['greeting']}")
        print()
        print(f"  (감지된 감정: {result['detected_emotion']}, 톤: {result['tone']})")
        print()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4-8: 멀티턴 대화
    print("="*80)
    print("이제 자유롭게 대화를 이어가세요.")
    print("="*80)
    print()
    
    turn_count = 1
    
    while True:
        try:
            user_input = input(f"나 (턴 {turn_count}): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["종료", "exit", "quit", "끝"]:
                print("\n상담을 종료합니다. 오늘도 좋은 하루 되세요! 😊")
                break
            
            # 감정 추출 및 응답 생성
            ti = TurnInput(utterance=user_input)
            output = orchestrator.run(ti, session_id=session_id)
            
            # 응답 출력
            print()
            print(f"상담사: {output.response}")
            print()
            print(f"  (감정: {max(output.emotion.aggregated.items(), key=lambda x: x[1])[0]}, "
                  f"톤: {output.meta['tone']}, "
                  f"누적 턴: {output.meta['turn_count']})")
            print()
            
            turn_count += 1
            
        except KeyboardInterrupt:
            print("\n\n상담을 종료합니다.")
            break
        except EOFError:
            print("\n상담을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("계속 진행하시겠습니까? (y/n): ", end="")
            choice = input().strip().lower()
            if choice != 'y':
                break
    
    # 감정 변화 요약
    print()
    print("="*80)
    print("📈 상담 요약")
    print("="*80)
    
    history = orchestrator.session_manager.get_or_create_session(session_id)
    trajectory = history.get_emotion_trajectory()
    
    print(f"총 턴 수: {len(history.turns)}")
    print(f"감정 변화: {' → '.join(trajectory)}")
    print()
    print("상담해 주셔서 감사합니다! 💚")
    print("="*80)
    print()
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
