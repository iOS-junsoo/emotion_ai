# ai_core/emotion/audio_vad.py
from typing import List, Tuple
import numpy as np
import torch
import torchaudio

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    # silero-vad가 없으면 간단한 에너지 기반 VAD
    load_silero_vad = None
    get_speech_timestamps = None


class AudioVAD:
    """
    silero VAD 또는 WebRTC VAD를 사용한 음성/무음 구간 추정.
    짧은 휴지기 vs 긴 휴지기 구분 기능 포함.
    """

    def __init__(
        self, 
        threshold: float = 0.5, 
        use_silero: bool = True,
        short_pause_ms: int = 300,  # 짧은 휴지기: 300ms
        long_pause_ms: int = 1500   # 긴 휴지기 (발화 종료): 1500ms
    ) -> None:
        self.threshold = threshold
        self.use_silero = use_silero and load_silero_vad is not None
        self.model = None
        self.sample_rate = 16000  # silero-vad 기본 샘플레이트
        
        # 휴지기 구분 파라미터
        self.short_pause_ms = short_pause_ms
        self.long_pause_ms = long_pause_ms
        
        if self.use_silero:
            try:
                self.model, self.utils = load_silero_vad()
            except Exception as e:
                # 조용히 fallback
                self.use_silero = False

    def detect(self, audio_chunk: List[float], sample_rate: int = 16000) -> bool:
        """
        오디오 청크가 발화 구간인지 판단.
        
        Args:
            audio_chunk: 오디오 샘플 리스트
            sample_rate: 샘플레이트
        
        Returns:
            True if speech, False if silence
        """
        if not audio_chunk:
            return False
        
        if self.use_silero and self.model is not None:
            try:
                # numpy array로 변환
                audio_np = np.array(audio_chunk, dtype=np.float32)
                
                # 리샘플링 (필요시)
                if sample_rate != self.sample_rate:
                    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    audio_tensor = resampler(audio_tensor)
                    audio_np = audio_tensor.squeeze(0).numpy()
                
                # VAD 예측
                speech_timestamps = get_speech_timestamps(
                    audio_np,
                    self.model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold
                )
                
                return len(speech_timestamps) > 0
            except Exception as e:
                print(f"silero-vad 에러: {e}, 에너지 기반 VAD로 폴백")
        
        # 폴백: 에너지 기반 VAD
        mean_abs = np.mean(np.abs(audio_chunk))
        return mean_abs >= self.threshold

    def get_speech_segments(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        min_segment_duration: float = 0.5
    ) -> List[Tuple[float, float]]:
        """
        오디오에서 발화 구간 리스트 반환.
        
        Args:
            audio: 오디오 numpy array
            sample_rate: 샘플레이트
            min_segment_duration: 최소 구간 길이 (초)
        
        Returns:
            [(start_time, end_time), ...] 리스트
        """
        if self.use_silero and self.model is not None:
            try:
                # 리샘플링
                if sample_rate != self.sample_rate:
                    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                    resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.squeeze(0).numpy()
                
                speech_timestamps = get_speech_timestamps(
                    audio,
                    self.model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold
                )
                
                segments = []
                for ts in speech_timestamps:
                    start = ts['start'] / self.sample_rate
                    end = ts['end'] / self.sample_rate
                    if (end - start) >= min_segment_duration:
                        segments.append((start, end))
                
                return segments
            except Exception as e:
                print(f"silero-vad 구간 추출 에러: {e}")
        
        # 폴백: 전체를 하나의 구간으로
        return [(0.0, len(audio) / sample_rate)]

    def detect_pause_type(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        window_ms: int = 100
    ) -> str:
        """
        현재 오디오가 짧은 휴지기인지, 긴 휴지기인지 판단.
        
        Args:
            audio: 오디오 numpy array
            sample_rate: 샘플레이트
            window_ms: 분석 윈도우 크기 (ms)
        
        Returns:
            "speech": 발화 중
            "short_pause": 짧은 휴지기 (300ms 이상)
            "long_pause": 긴 휴지기/발화 종료 (1500ms 이상)
        """
        if len(audio) == 0:
            return "speech"
        
        # 에너지 기반 무음 판단
        window_samples = int(window_ms * sample_rate / 1000)
        
        # 오디오를 윈도우 단위로 분석
        silence_duration_ms = 0
        
        for i in range(0, len(audio), window_samples):
            window = audio[i:i + window_samples]
            if len(window) < window_samples // 2:
                break
            
            # 에너지 계산
            energy = np.mean(np.abs(window))
            
            # 무음 판단: threshold보다 낮으면 무음으로 간주
            # threshold가 낮을 때(0.01)는 그대로 사용, 높을 때(0.5)는 2%만 사용
            silence_threshold = min(self.threshold * 0.02, self.threshold * 0.5)
            
            if energy < silence_threshold:
                silence_duration_ms += window_ms
            else:
                # 발화가 감지되면 무음 카운터 리셋
                silence_duration_ms = 0
        
        # 휴지기 분류
        if silence_duration_ms >= self.long_pause_ms:
            return "long_pause"
        elif silence_duration_ms >= self.short_pause_ms:
            return "short_pause"
        else:
            return "speech"

    def get_speech_segments_with_pauses(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_segment_duration: float = 0.5
    ) -> List[Tuple[float, float, str]]:
        """
        오디오에서 발화 구간 리스트 및 휴지기 타입 반환.
        
        Args:
            audio: 오디오 numpy array
            sample_rate: 샘플레이트
            min_segment_duration: 최소 구간 길이 (초)
        
        Returns:
            [(start_time, end_time, pause_type), ...] 리스트
            pause_type: "speech", "short_pause", "long_pause"
        """
        segments = self.get_speech_segments(audio, sample_rate, min_segment_duration)
        
        result = []
        prev_end = 0.0
        
        for start, end in segments:
            # 이전 구간과의 간격 계산
            if start > prev_end:
                gap_duration_ms = (start - prev_end) * 1000
                
                if gap_duration_ms >= self.long_pause_ms:
                    pause_type = "long_pause"
                elif gap_duration_ms >= self.short_pause_ms:
                    pause_type = "short_pause"
                else:
                    pause_type = "speech"
            else:
                pause_type = "speech"
            
            result.append((start, end, pause_type))
            prev_end = end
        
        return result