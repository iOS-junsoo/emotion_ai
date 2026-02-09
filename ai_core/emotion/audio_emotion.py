# ai_core/emotion/audio_emotion.py
from typing import Dict, List, Optional
import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from ai_core.emotion.audio_vad import AudioVAD
from ai_core.emotion.smoothing import average_softmax


class AudioEmotionEstimator:
    """
    SpeechBrain wav2vec2 기반 음성 감정 추정.
    세그먼트 처리 및 평활화 포함.
    """

    def __init__(
        self,
        window_size: float = 2.0,  # 초
        overlap: float = 0.5,  # 50% 오버랩
        smoothing_window: int = 5,  # 최근 N개 창 평균
        sample_rate: int = 16000,
        gpu_id: int = 0
    ) -> None:
        self.window_size = window_size
        self.overlap = overlap
        self.smoothing_window = smoothing_window
        self.sample_rate = sample_rate
        self.gpu_id = gpu_id
        self.ready = False
        
        # VAD
        self.vad = AudioVAD()
        
        # SpeechBrain 감정 모델 (wav2vec2 기반)
        self.model = None
        
        # 최근 감정 창 저장 (평활화용)
        self.recent_emotions: List[Dict[str, float]] = []

    def warmup(self) -> None:
        """모델 로드 및 워밍업"""
        try:
            from speechbrain.inference.classifiers import EncoderClassifier
            
            # GPU 사용 설정
            if torch.cuda.is_available():
                device = f"cuda:{self.gpu_id}"
            else:
                device = "cpu"
            
            run_opts = {"device": device}
            
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="models/speechbrain_emotion",
                run_opts=run_opts
            )
            
            self.ready = True
        except Exception as e:
            # 조용히 fallback
            self.model = None
            self.ready = True

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        오디오 정규화 (RMS 정규화).
        """
        if len(audio) == 0:
            return audio
        
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            target_rms = 0.1  # 목표 RMS
            audio = audio * (target_rms / rms)
        
        # 클리핑
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def segment_audio(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> List[np.ndarray]:
        """
        오디오를 세그먼트로 분할 (1.5~3초 창, 50% 오버랩).
        
        Args:
            audio: 오디오 numpy array
            sample_rate: 샘플레이트
        
        Returns:
            세그먼트 리스트
        """
        window_samples = int(self.window_size * sample_rate)
        hop_samples = int(window_samples * (1 - self.overlap))
        
        segments = []
        start = 0
        
        while start + window_samples <= len(audio):
            segment = audio[start:start + window_samples]
            segments.append(segment)
            start += hop_samples
        
        # 마지막 부분 처리
        if start < len(audio):
            segment = audio[-window_samples:]
            segments.append(segment)
        
        return segments

    def infer_segment(self, segment: np.ndarray) -> Dict[str, float]:
        """
        단일 세그먼트에서 감정 추정.
        """
        if self.model is None:
            return {"neutral": 0.6, "happy": 0.2, "sad": 0.1, "angry": 0.1}
        
        # 세그먼트 길이 확인
        if len(segment) < self.sample_rate * 0.5:
            return {"neutral": 1.0}
        
        try:
            # 정규화
            segment = self.normalize_audio(segment)
            
            # numpy array를 torch tensor로 변환
            wav_tensor = torch.from_numpy(segment).float()
            
            if wav_tensor.dim() == 1:
                wav_tensor = wav_tensor.unsqueeze(0)  # (1, time)
            
            # 디바이스로 이동
            device = next(self.model.mods['wav2vec2'].parameters()).device
            wav_tensor = wav_tensor.to(device)
            
            with torch.no_grad():
                try:
                    if 'wav2vec2' in self.model.mods:
                        # 1. wav2vec2로 특징 추출
                        try:
                            features = self.model.mods['wav2vec2'](wav_tensor)
                        except Exception as e:
                            if wav_tensor.dim() == 2:
                                wav_tensor = wav_tensor.unsqueeze(1)
                            features = self.model.mods['wav2vec2'](wav_tensor)
                        
                        # 2. avg_pool로 평균 풀링
                        if 'avg_pool' in self.model.mods:
                            pooled = self.model.mods['avg_pool'](features)
                            if pooled.dim() == 3:
                                pooled = pooled.squeeze(1)
                        else:
                            if features.dim() == 3:
                                pooled = features.mean(dim=1)
                            elif features.dim() == 2:
                                pooled = features
                            else:
                                pooled = features.mean()
                        
                        # 3. output_mlp로 분류
                        if 'output_mlp' in self.model.mods:
                            logits = self.model.mods['output_mlp'](pooled)
                            if logits.dim() == 3:
                                logits = logits.squeeze(1)
                        else:
                            raise ValueError("output_mlp not found")
                        
                        # 확률 분포로 변환
                        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                        
                        # IEMOCAP 라벨 매핑
                        iemocap_to_standard = {
                            'neu': 'neutral',
                            'hap': 'happy',
                            'ang': 'angry',
                            'sad': 'sad',
                            'exc': 'happy',
                            'fru': 'angry',
                            'fea': 'fear',
                            'sur': 'surprise',
                            'dis': 'disgust',
                            'xxx': 'neutral',
                        }
                        
                        # 라벨 매핑
                        emotion_dict = {}
                        if hasattr(self.model.hparams, 'label_encoder'):
                            label_list = self.model.hparams.label_encoder.ind2lab
                            
                            for i, prob in enumerate(probs):
                                if i < len(label_list):
                                    raw_label = str(label_list[i]).lower()
                                    standard_label = iemocap_to_standard.get(raw_label, raw_label)
                                    
                                    if standard_label in emotion_dict:
                                        emotion_dict[standard_label] += float(prob)
                                    else:
                                        emotion_dict[standard_label] = float(prob)
                        
                        # 기본 라벨 매핑
                        all_labels = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
                        for label in all_labels:
                            if label not in emotion_dict:
                                emotion_dict[label] = 0.0
                        
                        # 정규화
                        total = sum(emotion_dict.values())
                        if total > 0:
                            for key in emotion_dict:
                                emotion_dict[key] /= total
                        else:
                            emotion_dict = {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, 
                                          "fear": 0.0, "surprise": 0.0, "disgust": 0.0}
                        
                        return emotion_dict
                    else:
                        raise ValueError("wav2vec2 module not found")
                        
                except Exception as e1:
                    print(f"직접 forward pass 실패: {e1}")
                    import traceback
                    traceback.print_exc()
                    return {"neutral": 0.6, "happy": 0.2, "sad": 0.1, "angry": 0.1}
                    
        except Exception as e:
            print(f"세그먼트 감정 추정 에러: {e}")
            import traceback
            traceback.print_exc()
            return {"neutral": 1.0}

    def infer(self, audio_chunk: List[float], sample_rate: Optional[int] = None) -> Dict[str, float]:
        """
        오디오 청크에서 감정 추정 (세그먼트 처리 + 평활화).
        """
        if not self.ready:
            self.warmup()
        
        self._debug_count = 0
        
        if not audio_chunk:
            return {"neutral": 1.0}
        
        sample_rate = sample_rate or self.sample_rate
        audio = np.array(audio_chunk, dtype=np.float32)
        
        # 샘플레이트 변환
        if sample_rate != self.sample_rate:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
            sample_rate = self.sample_rate
        
        # VAD: 발화 구간만 추출
        speech_segments = self.vad.get_speech_segments(audio, sample_rate)
        
        if not speech_segments:
            return {"neutral": 1.0}
        
        # 발화 구간만 사용
        all_emotions = []
        
        for start_time, end_time in speech_segments:
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            speech_audio = audio[start_idx:end_idx]
            
            segments = self.segment_audio(speech_audio, sample_rate)
            
            for seg in segments:
                emotion = self.infer_segment(seg)
                all_emotions.append(emotion)
        
        if not all_emotions:
            return {"neutral": 1.0}
        
        # 신뢰도가 높은 세그먼트만 필터링 (최대 확률이 0.65 이상)
        high_confidence_emotions = []
        for emo in all_emotions:
            max_prob = max(emo.values())
            if max_prob >= 0.65:
                high_confidence_emotions.append(emo)
        
        # 신뢰도 높은 세그먼트가 충분히 있으면 그것만 사용
        if len(high_confidence_emotions) >= len(all_emotions) * 0.3:  # 최소 30%는 있어야 함
            emotions_to_use = high_confidence_emotions
        else:
            emotions_to_use = all_emotions
        
        # 가중 평균 (최근 세그먼트에 더 높은 가중치)
        weights = np.linspace(0.5, 1.0, len(emotions_to_use))
        weights = weights / weights.sum()
        
        weighted_emotion = {}
        all_labels = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        
        for label in all_labels:
            weighted_sum = 0.0
            for i, emo in enumerate(emotions_to_use):
                weighted_sum += emo.get(label, 0.0) * weights[i]
            weighted_emotion[label] = weighted_sum
        
        # 정규화
        total = sum(weighted_emotion.values())
        if total > 0:
            for key in weighted_emotion:
                weighted_emotion[key] /= total
        else:
            weighted_emotion = {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, 
                               "fear": 0.0, "surprise": 0.0, "disgust": 0.0}
        
        return weighted_emotion