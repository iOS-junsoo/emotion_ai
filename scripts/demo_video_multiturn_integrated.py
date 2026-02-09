"""
비디오 스트림 + 멀티턴 대화 통합 데모

실시간 비디오에서 감정을 추출하고,
각 발화마다 멀티턴 대화로 LLM 응답 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from ai_core.service.video_stream_processor import VideoStreamProcessor
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def print_separator(char="=", length=80):
    """구분선 출력"""
    print(char * length)


def main():
    parser = argparse.ArgumentParser(description="비디오 스트림 + 멀티턴 대화 통합 데모")
    
    # 비디오 관련
    parser.add_argument("--video", type=str, required=True, help="비디오 파일 경로")
    parser.add_argument("--gpu", type=int, default=3, help="사용할 GPU 번호")
    parser.add_argument("--fast", action="store_true", help="실시간 처리 모드")
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper 모델 크기 (기본: fast 모드는 tiny, 일반은 base)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="전사 언어 (기본: ko)"
    )
    
    # 감정 가중치
    parser.add_argument("--face-weight", type=float, default=0.45, help="표정 가중치")
    parser.add_argument("--audio-weight", type=float, default=0.35, help="음성 가중치")
    parser.add_argument("--text-weight", type=float, default=0.20, help="텍스트 가중치")
    
    # 대화 관련
    parser.add_argument("--session-id", type=str, default="video_session_001", help="세션 ID")
    parser.add_argument("--disable-multiturn", action="store_true", help="멀티턴 비활성화")
    
    # LLM 백엔드
    parser.add_argument(
        "--backend",
        type=str,
        choices=["vllm", "transformers"],
        default="transformers",
        help="LLM 백엔드 선택 (기본값: transformers)"
    )
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
    
    # 비디오 파일 확인
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return 1
    
    # 설정 출력
    print_separator()
    print("🎬 비디오 스트림 + 멀티턴 대화 통합 시스템")
    print_separator()
    print(f"📹 비디오: {video_path}")
    print(f"🎮 GPU: {args.gpu}")
    print(f"⚡ 실시간 모드: {'활성화' if args.fast else '비활성화'}")
    print(f"📊 감정 가중치: 표정 {args.face_weight*100:.0f}%, 음성 {args.audio_weight*100:.0f}%, 텍스트 {args.text_weight*100:.0f}%")
    print(f"💬 멀티턴 대화: {'✅ 활성화' if not args.disable_multiturn else '❌ 비활성화'}")
    print(f"🆔 세션 ID: {args.session_id}")
    print(f"🤖 LLM 백엔드: {args.backend}")
    if args.backend == "transformers":
        if args.load_in_8bit:
            print(f"   8비트 양자화: 활성화")
        if args.model_name:
            print(f"   모델: {args.model_name}")
    print_separator()
    print()
    
    # 1. VideoStreamProcessor 초기화
    print("🔧 비디오 스트림 프로세서 초기화 중...")
    
    # Whisper 모델 크기 결정
    if args.whisper_model:
        whisper_model = args.whisper_model
    else:
        whisper_model = "tiny" if args.fast else "base"
    
    print(f"   Whisper 모델: {whisper_model}")
    print(f"   언어: {args.language}")
    
    video_processor = VideoStreamProcessor(
        face_capture_interval=2.0 if args.fast else 1.0,
        whisper_model=whisper_model,
        language=args.language,
        gpu_id=args.gpu,
        verbose=False,  # 비디오 처리는 조용히
        face_weight=args.face_weight,
        audio_weight=args.audio_weight,
        text_weight=args.text_weight
    )
    
    # 2. TurnOrchestrator 초기화
    print("🔧 멀티턴 대화 orchestrator 초기화 중...")
    # VideoStreamProcessor가 CUDA_VISIBLE_DEVICES를 설정했으므로
    # TurnOrchestrator는 프로세스 내에서 0번 GPU를 사용해야 함
    orchestrator = TurnOrchestrator(
        enable_multiturn=not args.disable_multiturn,
        llm_backend=args.backend,
        gpu_id=0,  # VideoStreamProcessor가 이미 CUDA_VISIBLE_DEVICES 설정함
        load_in_8bit=args.load_in_8bit,
        model_name=args.model_name
    )
    
    # 3. 모델 워밍업
    print("🔥 모델 워밍업 중...")
    video_processor.warmup()
    print("✅ 초기화 완료!")
    print()
    
    # 4. 콜백 함수 정의
    turn_count = [0]  # mutable container for closure
    
    def on_turn_complete(turn_result):
        """각 발화 종료 시 호출"""
        turn_count[0] += 1
        
        print_separator()
        print(f"🎤 턴 {turn_count[0]}: 발화 감지 (시간: {turn_result.duration_sec:.2f}초)")
        print_separator("-")
        
        # 전사된 텍스트
        print(f"📝 전사 텍스트:")
        print(f"   {turn_result.transcribed_text}")
        print()
        
        # 감정 분석
        emotion_no_neutral = turn_result.aggregated_emotion_no_neutral
        if emotion_no_neutral:
            top_emotion = max(emotion_no_neutral.items(), key=lambda x: x[1])
            print(f"😊 통합 감정 (neutral 제외):")
            print(f"   주요: {top_emotion[0]} ({top_emotion[1]*100:.1f}%)")
            print(f"   전체: {', '.join([f'{k}: {v*100:.1f}%' for k, v in sorted(emotion_no_neutral.items(), key=lambda x: -x[1])[:3]])}")
        else:
            print(f"😊 통합 감정: neutral만 존재")
        print()
        
        # LLM 응답 생성 (텍스트가 있을 때만)
        if turn_result.transcribed_text.strip():
            try:
                print("🤖 상담사 응답 생성 중...")
                
                # TurnInput 생성
                turn_input = TurnInput(utterance=turn_result.transcribed_text)
                
                # 멀티턴 응답 생성
                if not args.disable_multiturn:
                    output = orchestrator.run(turn_input, session_id=args.session_id)
                    print(f"💬 대화 맥락: {output.meta.get('turn_count', 1)}턴 누적")
                else:
                    output = orchestrator.run(turn_input)
                    print(f"💬 대화 맥락: 단일 턴 (히스토리 없음)")
                
                print()
                print(f"🎯 응답 전략: {output.meta['tone']} / {output.meta['strategy']}")
                print()
                print(f"💬 상담사:")
                print(f"   {output.response}")
                print()
                
            except Exception as e:
                print(f"⚠️  LLM 응답 생성 실패: {e}")
                print(f"   vLLM 서버가 실행 중인지 확인하세요.")
                print()
        else:
            print("⚠️  전사된 텍스트가 없어 LLM 응답을 생성하지 않습니다.")
            print()
    
    # 5. 비디오 처리 시작
    try:
        print_separator()
        print("🎬 비디오 처리 시작!")
        print_separator()
        print()
        
        results = video_processor.process_video_stream(
            str(video_path),
            callback=on_turn_complete
        )
        
        # 6. 최종 요약
        print()
        print_separator()
        print("✅ 처리 완료!")
        print_separator()
        print(f"📊 총 턴 수: {len(results)}")
        print(f"⏱️  총 발화 시간: {sum(r.duration_sec for r in results):.2f}초")
        print()
        
        # 감정 변화 추이 (멀티턴일 때만)
        if not args.disable_multiturn:
            try:
                history = orchestrator.session_manager.get_or_create_session(args.session_id)
                emotion_trajectory = history.get_emotion_trajectory()
                
                if emotion_trajectory:
                    print("📈 감정 변화 추이:")
                    print(f"   {' → '.join(emotion_trajectory)}")
                    print()
            except:
                pass
        
        # 전체 대화 요약
        if not args.disable_multiturn:
            try:
                history = orchestrator.session_manager.get_or_create_session(args.session_id)
                if history.turns:
                    print("💬 대화 요약:")
                    for i, turn in enumerate(history.turns[-5:], 1):  # 마지막 5턴
                        print(f"   턴 {turn.turn_id}:")
                        print(f"     사용자: {turn.user_utterance[:50]}...")
                        print(f"     감정: {turn.emotion_label}")
                        print(f"     상담사: {turn.assistant_response[:50]}...")
                        print()
            except:
                pass
        
        print_separator()
        print("🎉 모든 처리가 완료되었습니다!")
        print_separator()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
