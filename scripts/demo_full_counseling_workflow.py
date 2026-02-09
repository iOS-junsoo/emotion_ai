"""
전체 상담 워크플로우 데모

워크플로우:
1. 사용자가 현재 자신의 감정 선택
2. 사용자가 현재 자신의 상황 설명 (텍스트)
3. LLM이 초기 인사말 생성 (감정 기반 프롬프팅)
4. LLM 인사말 출력
5. 비디오를 한 번의 대화 턴으로 처리
6. 비디오에서 얼굴+음성+텍스트 감정 추출
7. 추출된 감정에 따른 응답 스타일로 LLM 프롬프팅
8. LLM 응답 생성 및 출력
"""

import sys
import os
from pathlib import Path

# 모든 불필요한 출력 숨기기 (import 전에 설정)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 숨김
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Tokenizers 경고 숨김
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Transformers 경고 최소화
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # CUDA 동기화 비활성화
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # TensorFlow oneDNN 최적화 비활성화
os.environ['GLOG_minloglevel'] = '3'  # Google logging 레벨
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # GPU 할당자

# absl 로깅 완전히 비활성화 (TensorFlow 내부)
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# stderr를 임시로 리다이렉트 (TensorFlow/MediaPipe 경고 억제)
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """stderr를 임시로 억제"""
    import sys
    import os
    old_stderr = sys.stderr
    devnull = open(os.devnull, 'w')
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = old_stderr
        devnull.close()

# 전역 stderr 필터 (TensorFlow/absl 로그만 선택적으로 필터링)
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        
    def write(self, text):
        # TensorFlow/absl 로그 패턴 필터링
        if any(pattern in text for pattern in [
            "WARNING: All log messages before absl::InitializeLog()",
            "I0000 00:00:",
            "Created device",
            "Using CUDA malloc",
            "gpu_device.cc",
            "gpu_process_state.cc"
        ]):
            return  # 무시
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def fileno(self):
        return self.original_stderr.fileno()

# stderr 필터 적용
import sys
sys.stderr = FilteredStderr(sys.stderr)

import warnings
warnings.filterwarnings('ignore')

# Logging 레벨 조정
import logging
logging.getLogger('speechbrain').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# Transformers 특정 경고 억제
import transformers
transformers.logging.set_verbosity_error()

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

# Mock 모드 체크 (환경 변수)
USE_MOCK_MODE = os.getenv('MOCK_MODE', 'false').lower() in ['true', '1', 'yes']

if USE_MOCK_MODE:
    print("🎭 Mock 모드 활성화됨 (GPU 불필요)\n")
    from ai_core.mock_models import (
        MockFaceEmotionEstimator as FaceEmotionEstimator,
        MockAudioEmotionEstimator as AudioEmotionEstimator,
        MockTextEmotionEstimator as TextEmotionEstimator,
        MockWhisperSTT as WhisperSTT,
        MockLLMClient
    )
else:
    from ai_core.emotion.face_deepface import FaceEmotionEstimator
    from ai_core.emotion.audio_emotion import AudioEmotionEstimator
    from ai_core.emotion.text_emotion import TextEmotionEstimator
    from ai_core.stt.whisper_stt import WhisperSTT

from ai_core.emotion.smoothing import average_softmax
from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def print_separator(char="=", length=80):
    """구분선 출력"""
    print(char * length)


def select_emotion():
    """Step 1: 초기 감정 선택"""
    print_separator()
    print("😊 현재 당신의 감정은 어떤가요?")
    print_separator()
    print()
    
    emotions = {
        "1": ("sad", "슬픔 😢"),
        "2": ("angry", "분노 😠"),
        "3": ("happy", "기쁨 😊"),
        "4": ("anxious", "불안 😰"),
        "5": ("fear", "두려움 😨"),
        "6": ("neutral", "평온 😐"),
        "7": (None, "자동 감지")
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


def get_initial_text():
    """Step 2: 초기 상황 설명"""
    print_separator()
    print("📝 무엇을 이야기하고 싶으신가요?")
    print_separator()
    print("(여러 줄 입력 가능, 빈 줄로 Enter 시 완료)")
    print()
    
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    
    if not lines:
        return None
    
    text = " ".join(lines)
    print()
    print(f"✓ 입력 완료")
    print()
    return text


def weighted_average_softmax(emotion_dicts: list, weights: list) -> dict:
    """가중 평균 계산"""
    if not emotion_dicts:
        return {}
    
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        weights = [w / weight_sum for w in weights]
    
    all_labels = set()
    for d in emotion_dicts:
        all_labels.update(d.keys())
    
    result = {}
    for label in all_labels:
        weighted_sum = sum(
            d.get(label, 0.0) * w 
            for d, w in zip(emotion_dicts, weights)
        )
        result[label] = weighted_sum
    
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    
    return result


def get_final_emotion(emotion_dict: dict) -> str:
    """
    최종 감정 결정 (neutral이면 2번째로 높은 감정 사용)
    """
    sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 1순위가 neutral이 아니면 그대로 사용
    if sorted_emotions[0][0] != "neutral":
        return sorted_emotions[0][0]
    
    # 1순위가 neutral이면 2순위 사용
    if len(sorted_emotions) > 1:
        return sorted_emotions[1][0]
    
    # neutral만 있으면 neutral 반환
    return "neutral"


def process_video_turn(
    video_path: str,
    face_estimator,
    audio_estimator,
    text_estimator,
    stt,
    sample_rate: int = 16000,
    face_weight: float = 0.45,
    audio_weight: float = 0.35,
    text_weight: float = 0.20,
    sample_interval: int = 30,
    language: str = "ko"
):
    """
    Step 5-6: 비디오를 한 턴으로 처리하여 감정 추출
    """
    print()
    print("📹 사용자 응답 분석 중...")
    
    # Mock 모드: 실제 비디오 처리 건너뛰기
    if USE_MOCK_MODE:
        import time
        time.sleep(0.5)  # 처리 시뮬레이션
        
        # Mock 데이터 생성
        face_emotion = face_estimator.infer(None)
        audio_emotion = audio_estimator.infer([0.0], sample_rate)
        transcribed_text = stt.transcribe(np.array([0.0]), sample_rate, language)
        text_emotion = text_estimator.infer(transcribed_text)
        
        aggregated_emotion = weighted_average_softmax(
            [face_emotion, audio_emotion, text_emotion],
            [face_weight, audio_weight, text_weight]
        )
        
        final_emotion = get_final_emotion(aggregated_emotion)
        
        print()
        print(f"✅ 분석 완료!")
        print()
        
        # 각 모달리티 결과 출력
        def get_display_emotion(emotion_dict):
            sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
            if sorted_emotions[0][0] != "neutral":
                return sorted_emotions[0]
            if len(sorted_emotions) > 1:
                return sorted_emotions[1]
            return sorted_emotions[0]
        
        face_display = get_display_emotion(face_emotion)
        audio_display = get_display_emotion(audio_emotion)
        text_display = get_display_emotion(text_emotion)
        
        print(f"   😊 얼굴 감정: {face_display[0]} ({face_display[1]*100:.1f}%)")
        print(f"   🎤 음성 감정: {audio_display[0]} ({audio_display[1]*100:.1f}%)")
        print(f"   ✍️  텍스트 감정: {text_display[0]} ({text_display[1]*100:.1f}%)")
        print()
        print(f"   🎯 최종 감정: {final_emotion}")
        print(f"   📝 전사된 내용: \"{transcribed_text[:60]}{'...' if len(transcribed_text) > 60 else ''}\"")
        print()
        
        return {
            'face_emotion': face_emotion,
            'audio_emotion': audio_emotion,
            'text_emotion': text_emotion,
            'aggregated_emotion': aggregated_emotion,
            'final_emotion': final_emotion,
            'transcribed_text': transcribed_text,
            'duration_sec': 5.0  # Mock 길이
        }
    
    # 1. 오디오 추출
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',
            '-y', audio_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        audio_data, sr = sf.read(audio_path)
        audio_data = audio_data.astype(np.float32)
        duration_sec = len(audio_data) / sr
        
        # 2. 얼굴 감정 추출
        cap = cv2.VideoCapture(video_path)
        
        face_emotions = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                try:
                    # DeepFace TensorFlow 경고 억제
                    with suppress_stderr():
                        face_emotion = face_estimator.infer(frame)
                    face_emotions.append(face_emotion)
                except:
                    pass
            
            frame_idx += 1
        
        cap.release()
        
        if face_emotions:
            face_emotion = average_softmax(face_emotions)
        else:
            face_emotion = {"neutral": 1.0}
        
        # 3. 음성 감정 추출
        if len(audio_data) > 0:
            audio_emotion = audio_estimator.infer(audio_data.tolist(), sr)
        else:
            audio_emotion = {"neutral": 1.0}
        
        # 4. STT
        if len(audio_data) > 0:
            transcribed_text = stt.transcribe(audio_data, sr, language)
        else:
            transcribed_text = ""
        
        # 5. 텍스트 감정 추출
        if transcribed_text.strip():
            text_emotion = text_estimator.infer(transcribed_text)
        else:
            text_emotion = {"neutral": 1.0}
        
        # 6. 감정 통합
        aggregated_emotion = weighted_average_softmax(
            [face_emotion, audio_emotion, text_emotion],
            [face_weight, audio_weight, text_weight]
        )
        
        # neutral이면 2순위 감정 사용
        final_emotion = get_final_emotion(aggregated_emotion)
        
        print()
        print(f"✅ 분석 완료!")
        print()
        
        # 각 모달리티 결과 출력 (neutral이면 2순위)
        def get_display_emotion(emotion_dict):
            """neutral이면 2순위 감정 반환"""
            sorted_emotions = sorted(emotion_dict.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_emotions[0][0] != "neutral":
                return sorted_emotions[0]
            
            if len(sorted_emotions) > 1:
                return sorted_emotions[1]
            
            return sorted_emotions[0]
        
        face_display = get_display_emotion(face_emotion)
        audio_display = get_display_emotion(audio_emotion)
        text_display = get_display_emotion(text_emotion)
        
        print(f"   😊 얼굴 감정: {face_display[0]} ({face_display[1]*100:.1f}%)")
        print(f"   🎤 음성 감정: {audio_display[0]} ({audio_display[1]*100:.1f}%)")
        print(f"   ✍️  텍스트 감정: {text_display[0]} ({text_display[1]*100:.1f}%)")
        print()
        print(f"   🎯 최종 감정: {final_emotion}")
        print(f"   📝 전사된 내용: \"{transcribed_text[:60]}{'...' if len(transcribed_text) > 60 else ''}\"")
        print()
        
        return {
            'face_emotion': face_emotion,
            'audio_emotion': audio_emotion,
            'text_emotion': text_emotion,
            'aggregated_emotion': aggregated_emotion,
            'final_emotion': final_emotion,
            'transcribed_text': transcribed_text,
            'duration_sec': duration_sec
        }
        
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


def main():
    parser = argparse.ArgumentParser(description="전체 상담 워크플로우 데모")
    
    parser.add_argument("--video", type=str, required=True, help="비디오 파일 경로")
    parser.add_argument("--gpu", type=int, default=3, help="GPU 번호")
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper 모델"
    )
    parser.add_argument("--language", type=str, default="ko", help="언어")
    parser.add_argument("--sample-interval", type=int, default=30, help="프레임 샘플링 간격")
    
    # 감정 가중치
    parser.add_argument("--face-weight", type=float, default=0.45)
    parser.add_argument("--audio-weight", type=float, default=0.35)
    parser.add_argument("--text-weight", type=float, default=0.20)
    
    # LLM
    parser.add_argument("--backend", type=str, choices=["vllm", "transformers"], default="transformers")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    
    # 상호작용 모드
    parser.add_argument("--auto", action="store_true", help="자동 모드")
    parser.add_argument("--initial-emotion", type=str, default="sad")
    parser.add_argument("--initial-text", type=str, default="요즘 너무 힘들어요. 스트레스가 심해서 잠도 못 자고 있어요.")
    
    args = parser.parse_args()
    
    # 비디오 확인
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        return 1
    
    try:
        print()
        print_separator("=")
        print("💬 AI 감정 상담 시스템")
        print_separator("=")
        print()
        
        # Step 1 & 2: 사용자 입력
        if args.auto:
            initial_emotion = args.initial_emotion
            initial_text = args.initial_text
            print(f"초기 감정: {initial_emotion}")
            print(f"초기 상황: {initial_text}")
            print()
        else:
            initial_emotion = select_emotion()
            initial_text = get_initial_text()
            
            if not initial_text:
                print("❌ 텍스트가 입력되지 않았습니다.")
                return 1
        
        # GPU 설정
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        
        # 모델 초기화
        if USE_MOCK_MODE:
            print("⚙️  시스템 초기화 중... (Mock 모드)")
            face_estimator = FaceEmotionEstimator()
            audio_estimator = AudioEmotionEstimator(gpu_id=0)
            text_estimator = TextEmotionEstimator(gpu_id=0)
            stt = WhisperSTT(model_size=args.whisper_model, device='cpu')
            
            # Mock LLM으로 교체
            orchestrator = TurnOrchestrator(
                enable_multiturn=True,
                llm_backend=args.backend,
                gpu_id=0,
                load_in_8bit=args.load_in_8bit,
                model_name=args.model_name
            )
            # Mock LLM 주입
            orchestrator.llm = MockLLMClient()
        else:
            print("⚙️  시스템 초기화 중...")
            
            with suppress_stderr():
                face_estimator = FaceEmotionEstimator()
                audio_estimator = AudioEmotionEstimator(gpu_id=0)
                text_estimator = TextEmotionEstimator(gpu_id=0)
            
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            stt = WhisperSTT(model_size=args.whisper_model, device=device)
            
            orchestrator = TurnOrchestrator(
                enable_multiturn=True,
                llm_backend=args.backend,
                gpu_id=0,
                load_in_8bit=args.load_in_8bit,
                model_name=args.model_name
            )
        
        print("✅ 초기화 완료!\n")
        
        # Step 3-4: 초기 인사말 생성
        print_separator()
        print("💬 상담 시작")
        print_separator()
        print()
        
        session_id = "counseling_session_001"
        
        print("⏳ 상담사가 인사말을 준비하는 중...\n")
        
        session_result = orchestrator.start_session(
            session_id=session_id,
            user_initial_text=initial_text,
            user_emotion=initial_emotion
        )
        
        print_separator("-")
        print("🤖 상담사")
        print_separator("-")
        print()
        print(f"{session_result['greeting']}")
        print()
        
        # Step 5-6: 비디오 처리 (실시간 화상 상담)
        print_separator()
        print("👤 당신의 응답")
        print_separator()
        print()
        
        video_result = process_video_turn(
            str(video_path),
            face_estimator,
            audio_estimator,
            text_estimator,
            stt,
            sample_rate=16000,
            face_weight=args.face_weight,
            audio_weight=args.audio_weight,
            text_weight=args.text_weight,
            sample_interval=args.sample_interval,
            language=args.language
        )
        
        if not video_result['transcribed_text'].strip():
            print("⚠️  음성이 명확하지 않아 상담을 계속할 수 없습니다.")
            return 1
        
        # Step 7-8: 감정 기반 LLM 응답 생성
        print_separator()
        print("💭 상담사 응답 준비 중")
        print_separator()
        print()
        
        print(f"⏳ 감정({video_result['final_emotion']})에 맞춰 응답 생성 중...\n")
        
        # TurnInput 생성
        turn_input = TurnInput(utterance=video_result['transcribed_text'])
        
        # 감정 기반 응답 생성
        llm_output = orchestrator.run(turn_input, session_id=session_id)
        
        print_separator("-")
        print("🤖 상담사")
        print_separator("-")
        print()
        print(f"{llm_output.response}")
        print()
        
        # 최종 요약
        print_separator()
        print("📊 상담 요약")
        print_separator()
        print()
        
        history = orchestrator.session_manager.get_or_create_session(session_id)
        trajectory = history.get_emotion_trajectory()
        
        # 감정 변화
        if len(trajectory) >= 2:
            print(f"💭 감정 변화: {trajectory[0]} → {trajectory[-1]}")
        else:
            print(f"💭 감정: {trajectory[0] if trajectory else 'unknown'}")
        print()
        
        # LLM을 활용한 상담 내용 요약 및 행동 제안 생성
        print("⏳ 상담 내용을 정리하는 중...\n")
        
        # 대화 내용 수집
        conversation_text = f"초기 상황: {initial_text}\n\n"
        conversation_text += f"상담사 인사: {session_result['greeting']}\n\n"
        conversation_text += f"사용자 응답: {video_result['transcribed_text']}\n\n"
        conversation_text += f"상담사 응답: {llm_output.response}"
        
        # 요약 및 제안 생성 프롬프트
        summary_prompt = f"""다음은 감정 상담 대화 내용입니다:

{conversation_text}

위 상담 내용을 바탕으로 다음 두 가지를 간단명료하게 작성해주세요:

1. **상담 내용 요약** (2-3문장): 사용자가 겪고 있는 핵심 문제와 상담에서 다룬 주요 내용을 요약해주세요.

2. **행동 제안** (3-4개 항목): 사용자의 기분이 나아지고 문제 해결에 도움이 될 수 있는 구체적인 행동을 제안해주세요.

아래 형식으로 작성해주세요:
요약: [요약 내용]

행동 제안:
- [제안 1]
- [제안 2]
- [제안 3]
- [제안 4]"""

        try:
            # LLM으로 요약 및 제안 생성
            summary_response = orchestrator.llm.chat(
                system_prompt="당신은 전문 상담사입니다. 상담 내용을 간결하게 요약하고 실용적인 행동을 제안하세요.",
                user_prompt=summary_prompt,
                max_new_tokens=500,
                temperature=0.7
            )
            
            # 응답 파싱
            if "요약:" in summary_response and "행동 제안:" in summary_response:
                parts = summary_response.split("행동 제안:")
                summary_part = parts[0].replace("요약:", "").strip()
                action_part = parts[1].strip()
                
                print("📝 상담 내용:")
                print(f"   {summary_part}")
                print()
                
                print("💡 추천 행동:")
                # 각 행동 항목 출력
                for line in action_part.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                        # - 또는 번호 제거하고 출력
                        clean_line = line.lstrip('-•0123456789.').strip()
                        if clean_line:
                            print(f"   • {clean_line}")
                print()
            else:
                # 파싱 실패 시 전체 출력
                print(summary_response)
                print()
                
        except Exception as e:
            print(f"⚠️  요약 생성 중 오류 발생: {e}")
            print()
        
        print_separator()
        print("✅ 상담이 완료되었습니다!")
        print_separator()
        print()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  중단되었습니다.")
        return 1
    except Exception as e:
        print(f"\n\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
