# ai_core/stt/whisper_stt.py
from typing import Optional
import numpy as np
import torch
import whisper


class WhisperSTT:
    """
    Whisper 기반 음성-텍스트 변환기.
    """

    def __init__(self, model_size: str = "base", device: Optional[str] = None) -> None:
        """
        Args:
            model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
            device: 디바이스 (cuda, cpu, None=자동)
        """
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 16000  # Whisper 기본 샘플레이트

    def warmup(self) -> None:
        """모델 로드 및 워밍업"""
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
        except Exception as e:
            print(f"Warning: Whisper 모델 로드 실패: {e}")
            self.model = None

    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        language: str = "ko"
    ) -> str:
        """
        오디오를 텍스트로 변환.
        
        Args:
            audio: 오디오 numpy array (1D, float32, [-1, 1] 범위)
            sample_rate: 샘플레이트
            language: 언어 코드 (ko, en 등)
        
        Returns:
            변환된 텍스트
        """
        if self.model is None:
            self.warmup()
        
        if self.model is None:
            return ""
        
        if len(audio) == 0:
            return ""
        
        try:
            # 샘플레이트 확인
            if sample_rate != self.sample_rate:
                import torchaudio
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                audio_tensor = resampler(audio_tensor)
                audio = audio_tensor.squeeze(0).numpy()
            
            # Whisper는 float32, [-1, 1] 범위 필요
            audio = audio.astype(np.float32)
            
            # 너무 짧은 오디오는 무시
            if len(audio) < self.sample_rate * 0.3:  # 0.3초 미만
                return ""
            
            # 전사
            result = self.model.transcribe(
                audio,
                language=language,
                fp16=(self.device == "cuda"),
                verbose=False
            )
            
            text = result.get("text", "").strip()
            return text
            
        except Exception as e:
            print(f"Whisper 전사 에러: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def transcribe_stream(
        self,
        audio_chunks: list,
        sample_rate: int = 16000,
        language: str = "ko"
    ) -> str:
        """
        여러 오디오 청크를 하나로 합쳐서 전사.
        
        Args:
            audio_chunks: numpy array 리스트
            sample_rate: 샘플레이트
            language: 언어 코드
        
        Returns:
            변환된 텍스트
        """
        if not audio_chunks:
            return ""
        
        # 청크들을 하나로 결합
        audio = np.concatenate(audio_chunks)
        
        return self.transcribe(audio, sample_rate, language)
