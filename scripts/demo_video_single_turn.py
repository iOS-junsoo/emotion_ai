"""
비디오 단일 턴 처리 + LLM 응답 생성

전체 비디오를 하나의 턴으로 처리하여 감정 분석 및 LLM 응답 생성
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import cv2
import soundfile as sf
import subprocess
import tempfile
import os

from ai_core.emotion.face_deepface import FaceEmotionEstimator
from ai_core.emotion.audio_emotion import AudioEmotionEstimator
from ai_core.emotion.text_emotion import TextEmotionEstimator
from ai_core.emotion.smoothing import average_softmax
from ai_core.stt.whisper_stt import WhisperSTT
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def print_separator(char="=", length=80):
    """구분선 출력"""
    print(char * length)


def weighted_average_softmax(
    emotion_dicts: list, 
    weights: list
) -> dict:
    """가중 평균 계산"""
    if not emotion_dicts:
        return {}
    
    # 가중치 정규화
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        weights = [w / weight_sum for w in weights]
    
    # 모든 감정 레이블 수집
    all_labels = set()
    for d in emotion_dicts:
        all_labels.update(d.keys())
    
    # 가중 평균 계산
    result = {}
    for label in all_labels:
        weighted_sum = sum(
            d.get(label, 0.0) * w 
            for d, w in zip(emotion_dicts, weights)
        )
        result[label] = weighted_sum
    
    # 재정규화
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    
    return result


def process_video_as_single_turn(
    video_path: str,
    whisper_model: str = "base",
    language: str = "ko",
    gpu_id: int = 3,
    face_weight: float = 0.45,
    audio_weight: float = 0.35,
    text_weight: float = 0.20,
    sample_interval: int = 30  # 프레임 샘플링 간격
):
    """
    비디오를 단일 턴으로 처리
    
    Returns:
        dict: {
            'face_emotion': dict,
            'audio_emotion': dict,
            'text_emotion': dict,
            'aggregated_emotion': dict,
            'transcribed_text': str,
            'duration_sec': float,
            'face_samples': int
        }
    """
    print_separator()
    print("📹 비디오 처리 시작 (단일 턴)")
    print_separator()
    print(f"비디오: {video_path}")
    print(f"GPU: {gpu_id}")
    print(f"Whisper 모델: {whisper_model}")
    print(f"언어: {language}")
    print(f"감정 가중치: 표정 {face_weight*100:.0f}%, 음성 {audio_weight*100:.0f}%, 텍스트 {text_weight*100:.0f}%")
    print()
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 모델 초기화
    print("🔧 모델 초기화 중...")
    face_estimator = FaceEmotionEstimator()
    audio_estimator = AudioEmotionEstimator(gpu_id=0)
    text_estimator = TextEmotionEstimator(gpu_id=0)
    stt = WhisperSTT(model_size=whisper_model, device=device)
    print("✅ 모델 초기화 완료!")
    print()
    
    # 1. 오디오 추출
    print("🎵 오디오 추출 중...")
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        # ffmpeg로 오디오 추출
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',  # 모노
            '-y', audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 오디오 로드
        audio_data, sr = sf.read(audio_path)
        audio_data = audio_data.astype(np.float32)
        duration_sec = len(audio_data) / sr
        print(f"   오디오 길이: {duration_sec:.2f}초")
        print()
        
        # 2. 비디오 로드 및 얼굴 감정 추출
        print("😊 얼굴 감정 추출 중...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   총 프레임: {total_frames}, FPS: {fps:.2f}")
        
        face_emotions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 일정 간격으로 샘플링
            if frame_idx % sample_interval == 0:
                try:
                    face_emotion = face_estimator.infer(frame)
                    face_emotions.append(face_emotion)
                    if len(face_emotions) % 10 == 0:
                        print(f"   처리 중... {len(face_emotions)}개 샘플")
                except Exception as e:
                    # 얼굴이 감지되지 않는 프레임은 건너뜀
                    pass
            
            frame_idx += 1
        
        cap.release()
        
        if face_emotions:
            face_emotion = average_softmax(face_emotions)
            print(f"   완료! 총 {len(face_emotions)}개 샘플")
            print(f"   주요 감정: {max(face_emotion.items(), key=lambda x: x[1])}")
        else:
            face_emotion = {"neutral": 1.0}
            print("   ⚠️  얼굴 감정을 추출할 수 없습니다. neutral로 설정")
        print()
        
        # 3. 음성 감정 추출
        print("🎤 음성 감정 추출 중...")
        if len(audio_data) > 0:
            audio_emotion = audio_estimator.infer(audio_data.tolist(), sr)
            print(f"   주요 감정: {max(audio_emotion.items(), key=lambda x: x[1])}")
        else:
            audio_emotion = {"neutral": 1.0}
            print("   ⚠️  오디오가 없습니다. neutral로 설정")
        print()
        
        # 4. STT (텍스트 변환)
        print("📝 음성을 텍스트로 변환 중...")
        if len(audio_data) > 0:
            transcribed_text = stt.transcribe(audio_data, sr, language)
            print(f"   전사 결과: '{transcribed_text}'")
        else:
            transcribed_text = ""
            print("   ⚠️  오디오가 없습니다.")
        print()
        
        # 5. 텍스트 감정 추출
        print("✍️  텍스트 감정 추출 중...")
        if transcribed_text.strip():
            text_emotion = text_estimator.infer(transcribed_text)
            print(f"   주요 감정: {max(text_emotion.items(), key=lambda x: x[1])}")
        else:
            text_emotion = {"neutral": 1.0}
            print("   ⚠️  텍스트가 없습니다. neutral로 설정")
        print()
        
        # 6. 감정 통합
        print("🎯 감정 통합 중...")
        aggregated_emotion = weighted_average_softmax(
            [face_emotion, audio_emotion, text_emotion],
            [face_weight, audio_weight, text_weight]
        )
        print(f"   통합 감정: {max(aggregated_emotion.items(), key=lambda x: x[1])}")
        print()
        
        result = {
            'face_emotion': face_emotion,
            'audio_emotion': audio_emotion,
            'text_emotion': text_emotion,
            'aggregated_emotion': aggregated_emotion,
            'transcribed_text': transcribed_text,
            'duration_sec': duration_sec,
            'face_samples': len(face_emotions)
        }
        
        return result
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(audio_path):
            os.remove(audio_path)


def main():
    parser = argparse.ArgumentParser(description="비디오 단일 턴 처리 + LLM 응답")
    
    # 비디오 관련
    parser.add_argument("--video", type=str, required=True, help="비디오 파일 경로")
    parser.add_argument("--gpu", type=int, default=3, help="사용할 GPU 번호")
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper 모델 크기 (기본: small)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="전사 언어 (기본: ko)"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=30,
        help="얼굴 감정 추출 프레임 간격 (기본: 30)"
    )
    
    # 감정 가중치
    parser.add_argument("--face-weight", type=float, default=0.45, help="표정 가중치")
    parser.add_argument("--audio-weight", type=float, default=0.35, help="음성 가중치")
    parser.add_argument("--text-weight", type=float, default=0.20, help="텍스트 가중치")
    
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
        help="8비트 양자화 사용 (메모리 절약)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="모델 이름 (기본: Qwen/Qwen2-7B-Instruct)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default="video_single_turn_001",
        help="세션 ID"
    )
    
    args = parser.parse_args()
    
    # 비디오 파일 확인
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return 1
    
    try:
        # 1. 비디오 처리 (감정 추출 + STT)
        result = process_video_as_single_turn(
            str(video_path),
            whisper_model=args.whisper_model,
            language=args.language,
            gpu_id=args.gpu,
            face_weight=args.face_weight,
            audio_weight=args.audio_weight,
            text_weight=args.text_weight,
            sample_interval=args.sample_interval
        )
        
        # 2. TurnOrchestrator 초기화 (LLM 응답 생성)
        print_separator()
        print("🤖 LLM 초기화 중...")
        print_separator()
        print(f"백엔드: {args.backend}")
        if args.load_in_8bit:
            print("8비트 양자화: 활성화")
        if args.model_name:
            print(f"모델: {args.model_name}")
        print()
        
        orchestrator = TurnOrchestrator(
            enable_multiturn=True,
            llm_backend=args.backend,
            gpu_id=0,  # CUDA_VISIBLE_DEVICES가 이미 설정됨
            load_in_8bit=args.load_in_8bit,
            model_name=args.model_name
        )
        
        print("✅ LLM 초기화 완료!")
        print()
        
        # 3. 초기 상담 시작
        print_separator()
        print("💬 상담 세션 시작")
        print_separator()
        print()
        
        if not result['transcribed_text'].strip():
            print("⚠️  전사된 텍스트가 없어 상담을 시작할 수 없습니다.")
            print("더 좋은 품질의 비디오를 사용하거나 --whisper-model 옵션을 조정하세요.")
            return 1
        
        # 주요 감정 추출
        main_emotion = max(result['aggregated_emotion'].items(), key=lambda x: x[1])[0]
        
        print(f"📝 사용자 입력: {result['transcribed_text'][:100]}...")
        print(f"😊 감지된 감정: {main_emotion}")
        print()
        
        # 초기 인사말 생성
        session_result = orchestrator.start_session(
            session_id=args.session_id,
            user_initial_text=result['transcribed_text'],
            user_emotion=main_emotion
        )
        
        print_separator()
        print("💬 상담사 응답:")
        print_separator()
        print()
        print(f"  {session_result['greeting']}")
        print()
        print(f"  (감정: {session_result['detected_emotion']}, 톤: {session_result['tone']})")
        print()
        
        # 4. 결과 요약
        print_separator()
        print("📊 결과 요약")
        print_separator()
        print(f"비디오 길이: {result['duration_sec']:.2f}초")
        print(f"얼굴 샘플: {result['face_samples']}개")
        print(f"전사된 텍스트 길이: {len(result['transcribed_text'])}자")
        print()
        
        print("감정 분석:")
        print(f"  얼굴: {max(result['face_emotion'].items(), key=lambda x: x[1])}")
        print(f"  음성: {max(result['audio_emotion'].items(), key=lambda x: x[1])}")
        print(f"  텍스트: {max(result['text_emotion'].items(), key=lambda x: x[1])}")
        print(f"  통합: {max(result['aggregated_emotion'].items(), key=lambda x: x[1])}")
        print()
        
        print_separator()
        print("✅ 처리 완료!")
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
