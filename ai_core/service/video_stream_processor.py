# ai_core/service/video_stream_processor.py
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import time
from dataclasses import dataclass, field

from ai_core.emotion.face_deepface import FaceEmotionEstimator
from ai_core.emotion.audio_emotion import AudioEmotionEstimator
from ai_core.emotion.text_emotion import TextEmotionEstimator
from ai_core.emotion.audio_vad import AudioVAD
from ai_core.emotion.smoothing import average_softmax
from ai_core.stt.whisper_stt import WhisperSTT


def weighted_average_softmax(
    emotion_dicts: List[Dict[str, float]], 
    weights: List[float]
) -> Dict[str, float]:
    """
    여러 감정 딕셔너리를 가중치를 적용하여 평균.
    
    Args:
        emotion_dicts: 감정 확률 딕셔너리 리스트
        weights: 각 딕셔너리에 적용할 가중치 (합이 1.0이어야 함)
    
    Returns:
        가중 평균된 감정 딕셔너리
    """
    if not emotion_dicts:
        return {}
    
    if len(emotion_dicts) != len(weights):
        raise ValueError("emotion_dicts와 weights의 길이가 같아야 합니다")
    
    # 가중치 합이 1.0인지 확인
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 1e-6:
        # 자동으로 정규화
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


def exclude_neutral_and_normalize(emotion_dict: Dict[str, float]) -> Dict[str, float]:
    """
    neutral을 제외하고 나머지 감정을 재정규화.
    
    Args:
        emotion_dict: 감정 확률 딕셔너리
    
    Returns:
        neutral이 제외되고 재정규화된 감정 딕셔너리
    """
    # neutral 제외
    filtered = {k: v for k, v in emotion_dict.items() if k != 'neutral'}
    
    if not filtered:
        # neutral만 있는 경우, 빈 딕셔너리 반환
        return {}
    
    # 재정규화
    total = sum(filtered.values())
    if total > 0:
        return {k: v / total for k, v in filtered.items()}
    else:
        return filtered


@dataclass
class TurnResult:
    """한 턴(발화)의 결과"""
    face_emotion: Dict[str, float]
    audio_emotion: Dict[str, float]
    text_emotion: Dict[str, float]
    aggregated_emotion: Dict[str, float]
    aggregated_emotion_no_neutral: Dict[str, float]  # neutral 제외한 통합 감정
    transcribed_text: str
    duration_sec: float
    face_samples: int = 0
    audio_chunks: int = 0


class VideoStreamProcessor:
    """
    비디오 스트림 실시간 처리기.
    
    파이프라인:
    1. 표정: 1초마다 프레임 캡처 → 감정 추출 → 표정 버퍼 저장
    2. 음성: VAD로 짧은 휴지기 탐지 → 음성 감정 추출 → 음성 버퍼 저장 + STT로 텍스트 누적
    3. 발화 종료: VAD가 긴 휴지기 감지 → 버퍼 평균 계산 + 텍스트 감정 추출 → 결과 반환
    """

    def __init__(
        self,
        face_capture_interval: float = 1.0,  # 표정 캡처 간격 (초)
        short_pause_ms: int = 300,  # 짧은 휴지기 (ms)
        long_pause_ms: int = 1500,  # 긴 휴지기 (ms)
        audio_chunk_duration: float = 0.5,  # 오디오 청크 길이 (초)
        sample_rate: int = 16000,
        whisper_model: str = "base",
        language: str = "ko",
        gpu_id: int = 0,  # 사용할 GPU 번호
        verbose: bool = True,  # 디버깅 출력 여부
        face_weight: float = 0.45,  # 표정 감정 가중치
        audio_weight: float = 0.35,  # 음성 감정 가중치
        text_weight: float = 0.20  # 텍스트 감정 가중치
    ) -> None:
        # 파라미터
        self.face_capture_interval = face_capture_interval
        self.audio_chunk_duration = audio_chunk_duration
        self.sample_rate = sample_rate
        self.language = language
        self.gpu_id = gpu_id
        self.verbose = verbose
        
        # 감정 통합 가중치
        self.face_weight = face_weight
        self.audio_weight = audio_weight
        self.text_weight = text_weight
        
        # 가중치 합이 1.0인지 확인
        weight_sum = face_weight + audio_weight + text_weight
        if abs(weight_sum - 1.0) > 1e-6:
            print(f"⚠️ 가중치 합이 1.0이 아닙니다 ({weight_sum:.3f}). 자동으로 정규화합니다.")
            self.face_weight = face_weight / weight_sum
            self.audio_weight = audio_weight / weight_sum
            self.text_weight = text_weight / weight_sum
        
        # GPU 설정 - 프로세스 시작 시 즉시 설정
        import torch
        import os
        
        # 환경변수를 가장 먼저 설정 (다른 라이브러리 로드 전)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # TensorFlow GPU 메모리 증가 허용
        
        if torch.cuda.is_available():
            device = "cuda:0"  # CUDA_VISIBLE_DEVICES 설정 후에는 항상 0번
            torch.cuda.set_device(0)
        else:
            device = "cpu"
        
        # 감정 분석 모델 (CUDA_VISIBLE_DEVICES 설정 후에는 모두 device=0 사용)
        self.face_estimator = FaceEmotionEstimator()
        self.audio_estimator = AudioEmotionEstimator(gpu_id=0)  # 프로세스 내에서는 0번
        self.text_estimator = TextEmotionEstimator(gpu_id=0)  # 프로세스 내에서는 0번
        
        # VAD 및 STT
        # threshold를 낮춰서 테스트 비디오의 사인파 톤도 감지 가능하도록 설정
        self.vad = AudioVAD(
            threshold=0.01,  # 0.5 → 0.01로 낮춤 (에너지 기반 VAD용)
            short_pause_ms=short_pause_ms,
            long_pause_ms=long_pause_ms
        )
        
        self.stt = WhisperSTT(model_size=whisper_model, device=device)
        
        # 버퍼
        self.face_buffer: List[Dict[str, float]] = []
        self.audio_buffer: List[Dict[str, float]] = []
        self.audio_chunks: List[np.ndarray] = []  # STT용 오디오 누적
        self.transcribed_text: str = ""
        
        # 상태
        self.last_face_capture_time: float = 0.0
        self.turn_start_time: float = 0.0
        self.is_speaking: bool = False
        self.silence_start_time: Optional[float] = None
        
        # 콜백
        self.on_turn_complete: Optional[Callable[[TurnResult], None]] = None

    def warmup(self) -> None:
        """모든 모델 워밍업"""
        self.face_estimator.warmup()
        self.audio_estimator.warmup()
        self.text_estimator.warmup()
        self.stt.warmup()

    def reset_buffers(self) -> None:
        """버퍼 초기화"""
        self.face_buffer.clear()
        self.audio_buffer.clear()
        self.audio_chunks.clear()
        self.transcribed_text = ""

    def process_frame(self, frame_bgr: np.ndarray, current_time: float) -> None:
        """
        프레임 처리 (1초마다 감정 추출).
        
        Args:
            frame_bgr: OpenCV BGR 이미지
            current_time: 현재 타임스탬프 (초)
        """
        # 1초마다 캡처
        if current_time - self.last_face_capture_time >= self.face_capture_interval:
            face_emotion = self.face_estimator.infer(frame_bgr)
            self.face_buffer.append(face_emotion)
            self.last_face_capture_time = current_time
            
            if self.verbose:
                print(f"[표정] 캡처 (시간: {current_time:.1f}s, 버퍼 크기: {len(self.face_buffer)})")

    def process_audio_chunk(
        self,
        audio_chunk: np.ndarray,
        current_time: float
    ) -> Optional[str]:
        """
        오디오 청크 처리.
        
        Args:
            audio_chunk: 오디오 numpy array (1D, float32)
            current_time: 현재 타임스탬프 (초)
        
        Returns:
            "short_pause": 짧은 휴지기 감지 (음성 감정 추출)
            "long_pause": 긴 휴지기 감지 (발화 종료)
            None: 발화 중
        """
        if len(audio_chunk) == 0:
            return None
        
        # VAD로 발화 여부 확인
        is_speech = self.vad.detect(audio_chunk.tolist(), self.sample_rate)
        
        if is_speech:
            # 발화 시작
            if not self.is_speaking:
                self.is_speaking = True
                self.turn_start_time = current_time
                if self.verbose:
                    print(f"\n[발화 시작] 시간: {current_time:.1f}s")
            
            # 오디오 누적 (STT용)
            self.audio_chunks.append(audio_chunk)
            self.silence_start_time = None
            
        else:
            # 무음 구간
            if self.is_speaking:
                if self.silence_start_time is None:
                    self.silence_start_time = current_time
                
                silence_duration_ms = (current_time - self.silence_start_time) * 1000
                
                # 긴 휴지기 (발화 종료)
                if silence_duration_ms >= self.vad.long_pause_ms:
                    if self.verbose:
                        print(f"[긴 휴지기 감지] 발화 종료 (무음: {silence_duration_ms:.0f}ms)")
                    return "long_pause"
                
                # 짧은 휴지기 (음성 감정 추출)
                elif silence_duration_ms >= self.vad.short_pause_ms:
                    if len(self.audio_chunks) > 0:
                        if self.verbose:
                            print(f"[짧은 휴지기 감지] 음성 감정 추출 (무음: {silence_duration_ms:.0f}ms)")
                        return "short_pause"
        
        return None

    def extract_audio_emotion(self) -> None:
        """현재까지 누적된 오디오의 감정 추출 및 버퍼 저장"""
        if len(self.audio_chunks) == 0:
            return
        
        # 오디오 청크 결합
        audio = np.concatenate(self.audio_chunks)
        
        # 감정 추출
        audio_emotion = self.audio_estimator.infer(audio.tolist(), self.sample_rate)
        self.audio_buffer.append(audio_emotion)
        
        if self.verbose:
            print(f"  → 음성 감정 추출 완료 (버퍼 크기: {len(self.audio_buffer)})")
            print(f"     주요 감정: {max(audio_emotion.items(), key=lambda x: x[1])}")

    def transcribe_audio(self) -> None:
        """현재까지 누적된 오디오를 텍스트로 변환"""
        if len(self.audio_chunks) == 0:
            return
        
        # STT
        text = self.stt.transcribe_stream(
            self.audio_chunks,
            self.sample_rate,
            self.language
        )
        
        if text:
            self.transcribed_text += " " + text
            if self.verbose:
                print(f"  → STT: '{text}'")

    def finalize_turn(self, current_time: float) -> TurnResult:
        """
        발화 종료 시 최종 감정 집계 및 결과 반환.
        
        Returns:
            TurnResult 객체
        """
        if self.verbose:
            print(f"\n[발화 종료 처리] 시간: {current_time:.1f}s")
        
        # 마지막 오디오 청크 처리
        if len(self.audio_chunks) > 0:
            self.extract_audio_emotion()
            self.transcribe_audio()
        
        # 1. 표정 버퍼 평균
        if len(self.face_buffer) > 0:
            face_emotion = average_softmax(self.face_buffer)
            if self.verbose:
                print(f"  → 표정 감정 평균 (샘플 수: {len(self.face_buffer)})")
                print(f"     {max(face_emotion.items(), key=lambda x: x[1])}")
        else:
            face_emotion = {"neutral": 1.0}
        
        # 2. 음성 버퍼 평균
        if len(self.audio_buffer) > 0:
            audio_emotion = average_softmax(self.audio_buffer)
            if self.verbose:
                print(f"  → 음성 감정 평균 (청크 수: {len(self.audio_buffer)})")
                print(f"     {max(audio_emotion.items(), key=lambda x: x[1])}")
        else:
            audio_emotion = {"neutral": 1.0}
        
        # 3. 전체 텍스트 감정 추출
        text = self.transcribed_text.strip()
        if text:
            text_emotion = self.text_estimator.infer(text)
            if self.verbose:
                print(f"  → 텍스트 감정 추출")
                print(f"     텍스트: '{text}'")
                print(f"     {max(text_emotion.items(), key=lambda x: x[1])}")
        else:
            text_emotion = {"neutral": 1.0}
            text = ""
        
        # 4. 3개 감정 통합 (가중 평균)
        aggregated_emotion = weighted_average_softmax(
            [face_emotion, audio_emotion, text_emotion],
            [self.face_weight, self.audio_weight, self.text_weight]
        )
        if self.verbose:
            print(f"  → 통합 감정 (가중치 적용, neutral 포함):")
            print(f"     {max(aggregated_emotion.items(), key=lambda x: x[1])}")
        
        # 5. neutral 제외한 최종 감정
        aggregated_emotion_no_neutral = exclude_neutral_and_normalize(aggregated_emotion)
        if self.verbose:
            if aggregated_emotion_no_neutral:
                print(f"  → 최종 감정 (neutral 제외):")
                print(f"     {max(aggregated_emotion_no_neutral.items(), key=lambda x: x[1])}")
            else:
                print(f"  → 최종 감정: neutral만 존재")
        
        # 결과 생성
        duration = current_time - self.turn_start_time
        result = TurnResult(
            face_emotion=face_emotion,
            audio_emotion=audio_emotion,
            text_emotion=text_emotion,
            aggregated_emotion=aggregated_emotion,
            aggregated_emotion_no_neutral=aggregated_emotion_no_neutral,
            transcribed_text=text,
            duration_sec=duration,
            face_samples=len(self.face_buffer),
            audio_chunks=len(self.audio_buffer)
        )
        
        # 버퍼 초기화
        self.reset_buffers()
        self.is_speaking = False
        self.silence_start_time = None
        
        # 콜백 호출
        if self.on_turn_complete:
            self.on_turn_complete(result)
        
        return result

    def process_video_stream(
        self,
        video_path: str,
        callback: Optional[Callable[[TurnResult], None]] = None
    ) -> List[TurnResult]:
        """
        비디오 파일을 처리하여 턴별 결과 반환.
        
        Args:
            video_path: 비디오 파일 경로
            callback: 턴 완료 시 호출할 콜백 함수
        
        Returns:
            TurnResult 리스트
        """
        import cv2
        import soundfile as sf
        import subprocess
        import tempfile
        import os
        
        self.on_turn_complete = callback
        results = []
        
        print(f"비디오 파일 처리 시작: {video_path}")
        
        # 1. 비디오에서 오디오 추출
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            audio_path = tmp_audio.name
        
        try:
            # ffmpeg로 오디오 추출
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',  # 모노
                '-y', audio_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # 오디오 로드
            audio_data, sr = sf.read(audio_path)
            audio_data = audio_data.astype(np.float32)
            
            # 비디오 로드
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  FPS: {fps}, 샘플레이트: {sr}Hz, 오디오 길이: {len(audio_data)/sr:.2f}초")
            
            # 오디오 청크 크기
            chunk_samples = int(self.audio_chunk_duration * self.sample_rate)
            
            frame_idx = 0
            audio_idx = 0
            
            while True:
                # 프레임 읽기
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_idx / fps
                
                # 표정 처리
                self.process_frame(frame, current_time)
                
                # 오디오 처리 (현재 프레임 시간에 해당하는 오디오)
                audio_start = int(current_time * self.sample_rate)
                audio_end = min(audio_start + chunk_samples, len(audio_data))
                
                if audio_start < len(audio_data):
                    audio_chunk = audio_data[audio_start:audio_end]
                    
                    pause_type = self.process_audio_chunk(audio_chunk, current_time)
                    
                    if pause_type == "short_pause":
                        # 짧은 휴지기: 음성 감정 추출 + STT
                        self.extract_audio_emotion()
                        self.transcribe_audio()
                        self.audio_chunks.clear()  # 처리 완료한 청크는 비움
                    
                    elif pause_type == "long_pause":
                        # 긴 휴지기: 발화 종료
                        result = self.finalize_turn(current_time)
                        results.append(result)
                
                frame_idx += 1
            
            # 마지막 턴 처리 (비디오 끝났는데 발화 중이면)
            if self.is_speaking:
                result = self.finalize_turn(len(audio_data) / self.sample_rate)
                results.append(result)
            
            cap.release()
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        print(f"\n비디오 처리 완료! 총 {len(results)}개 턴 감지")
        return results
