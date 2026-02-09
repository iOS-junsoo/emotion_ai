#!/usr/bin/env python3
"""
비디오 스트림 처리 데모

사용법:
    python scripts/demo_video_stream.py --video <비디오_파일_경로>
    
예시:
    python scripts/demo_video_stream.py --video test_video/sample.mp4
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.service.video_stream_processor import VideoStreamProcessor, TurnResult


def on_turn_complete(result: TurnResult) -> None:
    """턴 완료 시 호출되는 콜백"""
    print("\n" + "="*80)
    print(f"📊 턴 결과 요약")
    print("="*80)
    print(f"발화 시간: {result.duration_sec:.2f}초")
    print(f"표정 샘플: {result.face_samples}개")
    print(f"음성 청크: {result.audio_chunks}개")
    print(f"\n📝 전사된 텍스트:")
    print(f"  {result.transcribed_text}")
    print(f"\n😊 표정 감정:")
    print_emotion(result.face_emotion)
    print(f"\n🎤 음성 감정:")
    print_emotion(result.audio_emotion)
    print(f"\n✍️  텍스트 감정:")
    print_emotion(result.text_emotion)
    print(f"\n🎯 통합 감정 (neutral 포함):")
    print_emotion(result.aggregated_emotion)
    print(f"\n🏆 최종 감정 (neutral 제외):")
    if result.aggregated_emotion_no_neutral:
        print_emotion(result.aggregated_emotion_no_neutral)
    else:
        print("  neutral만 존재")
    print("="*80 + "\n")


def print_emotion(emotion: dict) -> None:
    """감정 분포를 보기 좋게 출력"""
    sorted_emotions = sorted(emotion.items(), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_emotions[:3]:  # 상위 3개만
        bar = "█" * int(prob * 30)
        print(f"  {label:10s} {prob*100:5.1f}% {bar}")


def main():
    parser = argparse.ArgumentParser(description="비디오 스트림 감정 분석 데모")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="입력 비디오 파일 경로"
    )
    parser.add_argument(
        "--face-interval",
        type=float,
        default=1.0,
        help="표정 캡처 간격 (초, 기본값: 1.0)"
    )
    parser.add_argument(
        "--short-pause",
        type=int,
        default=300,
        help="짧은 휴지기 (ms, 기본값: 300)"
    )
    parser.add_argument(
        "--long-pause",
        type=int,
        default=1500,
        help="긴 휴지기/발화 종료 (ms, 기본값: 1500)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 모델 크기 (기본값: base)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="언어 코드 (기본값: ko)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="사용할 GPU 번호 (기본값: 0)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="실시간 처리 모드 (Whisper tiny, 표정 2초 간격, 디버깅 출력 최소화)"
    )
    parser.add_argument(
        "--face-weight",
        type=float,
        default=0.45,
        help="표정 감정 가중치 (기본값: 0.45, 45%%)"
    )
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=0.35,
        help="음성 감정 가중치 (기본값: 0.35, 35%%)"
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.20,
        help="텍스트 감정 가중치 (기본값: 0.20, 20%%)"
    )
    
    args = parser.parse_args()
    
    # 실시간 처리 모드 설정
    if args.fast:
        # Whisper tiny 모델 사용
        if args.whisper_model == "base":
            args.whisper_model = "tiny"
        # 표정 캡처 간격 2초로 증가 (기본값이 아닌 경우만)
        if args.face_interval == 1.0:
            args.face_interval = 2.0
        verbose = False
        print("⚡ 실시간 처리 모드 활성화")
    else:
        verbose = True
    
    # 비디오 파일 확인
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return 1
    
    print(f"🎬 비디오 스트림 감정 분석 시작")
    print(f"  비디오: {video_path}")
    print(f"  표정 캡처 간격: {args.face_interval}초")
    print(f"  짧은 휴지기: {args.short_pause}ms")
    print(f"  긴 휴지기: {args.long_pause}ms")
    print(f"  Whisper 모델: {args.whisper_model}")
    print(f"  언어: {args.language}")
    print(f"  GPU: {args.gpu}")
    print(f"  감정 가중치: 표정 {args.face_weight*100:.0f}%, 음성 {args.audio_weight*100:.0f}%, 텍스트 {args.text_weight*100:.0f}%")
    print()
    
    # 프로세서 초기화
    processor = VideoStreamProcessor(
        face_capture_interval=args.face_interval,
        short_pause_ms=args.short_pause,
        long_pause_ms=args.long_pause,
        whisper_model=args.whisper_model,
        language=args.language,
        gpu_id=args.gpu,
        verbose=verbose,
        face_weight=args.face_weight,
        audio_weight=args.audio_weight,
        text_weight=args.text_weight
    )
    
    # 모델 워밍업
    processor.warmup()
    
    # 비디오 처리
    try:
        results = processor.process_video_stream(
            str(video_path),
            callback=on_turn_complete
        )
        
        # 최종 요약
        print("\n" + "="*80)
        print(f"✅ 처리 완료!")
        print("="*80)
        print(f"총 턴 수: {len(results)}")
        
        total_duration = sum(r.duration_sec for r in results)
        print(f"총 발화 시간: {total_duration:.2f}초")
        
        # 전체 텍스트
        all_text = " ".join(r.transcribed_text for r in results if r.transcribed_text)
        print(f"\n📝 전체 전사 텍스트:")
        print(f"  {all_text}")
        
        # 평균 통합 감정 (가중치 적용)
        if results:
            from ai_core.emotion.smoothing import average_softmax
            from ai_core.service.video_stream_processor import exclude_neutral_and_normalize, weighted_average_softmax
            
            # 각 턴의 face, audio, text 감정을 평균 낸 후 가중치 적용
            avg_face = average_softmax([r.face_emotion for r in results])
            avg_audio = average_softmax([r.audio_emotion for r in results])
            avg_text = average_softmax([r.text_emotion for r in results])
            
            avg_emotion = weighted_average_softmax(
                [avg_face, avg_audio, avg_text],
                [args.face_weight, args.audio_weight, args.text_weight]
            )
            print(f"\n🎯 전체 평균 감정 (가중치 적용, neutral 포함):")
            print_emotion(avg_emotion)
            
            avg_emotion_no_neutral = exclude_neutral_and_normalize(avg_emotion)
            if avg_emotion_no_neutral:
                print(f"\n🏆 전체 최종 감정 (neutral 제외):")
                print_emotion(avg_emotion_no_neutral)
            else:
                print(f"\n🏆 전체 최종 감정: neutral만 존재")
        
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
