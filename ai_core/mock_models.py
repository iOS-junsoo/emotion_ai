"""
Mock 모델들 - GPU 없이 API 구조 테스트용

환경 변수 MOCK_MODE=true 설정 시 사용됩니다.
실제 AI 모델 대신 즉시 응답하는 더미 모델을 사용합니다.
"""

from typing import Dict, List, Optional
import time
import random


class MockLLMClient:
    """Mock LLM 클라이언트 (GPU 불필요)"""
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'Mock-LLM')
        print(f"🎭 Mock LLM 초기화됨 (GPU 불필요)")
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """단일 턴 채팅 (Mock 응답)"""
        time.sleep(0.1)  # 실제 모델 시뮬레이션
        
        # 감정 키워드 기반 간단한 응답 생성
        if "슬픔" in user_prompt or "sad" in system_prompt.lower():
            return "많이 힘드셨겠어요. 그 감정을 충분히 이해합니다. 천천히 이야기 나눠봐요."
        elif "분노" in user_prompt or "angry" in system_prompt.lower():
            return "화가 나시는 것 같네요. 그 감정을 표현하는 것이 중요합니다. 무슨 일이 있었나요?"
        elif "불안" in user_prompt or "anxious" in system_prompt.lower():
            return "불안한 마음이 드시는군요. 그 감정을 함께 나눠봐요. 어떤 부분이 걱정되시나요?"
        else:
            return "안녕하세요. 오늘 기분이 어떠신가요? 편하게 이야기 나눠봐요."
    
    def chat_multiturn(
        self,
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        current_user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """멀티턴 채팅 (Mock 응답)"""
        time.sleep(0.1)
        
        # 간단한 문맥 기반 응답
        responses = [
            "그렇군요. 그 부분에 대해 좀 더 자세히 말씀해주실 수 있나요?",
            "충분히 이해합니다. 그런 상황이라면 힘드셨을 것 같아요.",
            "당신의 감정을 공감합니다. 조금씩 나아질 거예요.",
            "좋은 생각이에요. 그 방향으로 한 걸음씩 나아가봐요."
        ]
        
        return random.choice(responses)


class MockFaceEmotionEstimator:
    """Mock 얼굴 감정 분석 (GPU 불필요)"""
    
    def __init__(self, **kwargs):
        self.is_warm = False
        print(f"🎭 Mock 얼굴 감정 분석 초기화됨")
    
    def warmup(self):
        self.is_warm = True
    
    def infer(self, frame_bgr) -> Dict[str, float]:
        """Mock 얼굴 감정 (랜덤)"""
        emotions = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        result = {e: random.random() for e in emotions}
        
        # 정규화
        total = sum(result.values())
        result = {k: v/total for k, v in result.items()}
        
        return result


class MockAudioEmotionEstimator:
    """Mock 음성 감정 분석 (GPU 불필요)"""
    
    def __init__(self, **kwargs):
        self.ready = False
        print(f"🎭 Mock 음성 감정 분석 초기화됨")
    
    def warmup(self):
        self.ready = True
    
    def infer(self, audio_chunk, sample_rate=None) -> Dict[str, float]:
        """Mock 음성 감정 (랜덤)"""
        emotions = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        result = {e: random.random() for e in emotions}
        
        # 정규화
        total = sum(result.values())
        result = {k: v/total for k, v in result.items()}
        
        return result


class MockTextEmotionEstimator:
    """Mock 텍스트 감정 분석 (GPU 불필요)"""
    
    def __init__(self, **kwargs):
        self.model_ready = False
        print(f"🎭 Mock 텍스트 감정 분석 초기화됨")
    
    def warmup(self):
        self.model_ready = True
    
    def infer(self, text: str) -> Dict[str, float]:
        """Mock 텍스트 감정 (키워드 기반)"""
        text_lower = text.lower()
        
        # 간단한 키워드 기반 감정 판단
        if any(word in text_lower for word in ["힘들", "슬프", "우울", "외로"]):
            return {"sad": 0.6, "neutral": 0.2, "angry": 0.1, "happy": 0.05, "fear": 0.05}
        elif any(word in text_lower for word in ["화", "짜증", "분노", "억울"]):
            return {"angry": 0.6, "sad": 0.2, "neutral": 0.1, "fear": 0.05, "disgust": 0.05}
        elif any(word in text_lower for word in ["불안", "걱정", "두려"]):
            return {"fear": 0.6, "neutral": 0.2, "sad": 0.1, "angry": 0.05, "surprise": 0.05}
        elif any(word in text_lower for word in ["기쁨", "행복", "좋아", "즐거"]):
            return {"happy": 0.6, "neutral": 0.2, "surprise": 0.1, "sad": 0.05, "angry": 0.05}
        else:
            return {"neutral": 0.5, "sad": 0.2, "happy": 0.15, "angry": 0.1, "fear": 0.05}


class MockWhisperSTT:
    """Mock Whisper STT (GPU 불필요)"""
    
    def __init__(self, **kwargs):
        self.model_size = kwargs.get('model_size', 'mock')
        self.device = kwargs.get('device', 'cpu')
        self.model = True  # Mock 모델
        print(f"🎭 Mock Whisper STT 초기화됨")
    
    def warmup(self):
        pass
    
    def transcribe(self, audio, sample_rate, language="ko") -> str:
        """Mock STT (더미 텍스트 반환)"""
        time.sleep(0.1)
        
        # 오디오 길이 기반 더미 텍스트
        duration = len(audio) / sample_rate
        
        mock_texts = [
            "요즘 너무 힘들어요. 일도 많고 스트레스가 심해서 잠도 못 자고 있어요.",
            "회사 상사와의 관계가 좋지 않아요. 매일 출근하는 게 두렵습니다.",
            "가족 문제로 고민이 많아요. 어떻게 해야 할지 모르겠어요.",
            "친구와 싸웠는데 화해할 방법을 모르겠어요.",
            "시험 결과가 좋지 않아서 자신감이 떨어졌어요."
        ]
        
        return random.choice(mock_texts)
    
    def transcribe_stream(self, audio_chunks, sample_rate=16000, language="ko") -> str:
        """Mock 스트림 STT"""
        if not audio_chunks:
            return ""
        
        # 전체 오디오 합치기
        import numpy as np
        audio = np.concatenate(audio_chunks)
        return self.transcribe(audio, sample_rate, language)
