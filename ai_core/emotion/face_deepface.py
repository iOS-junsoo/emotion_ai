# ai_core/emotion/face_deepface.py
from typing import Dict, Any, Optional
import numpy as np
import os
import cv2

from deepface import DeepFace


class FaceEmotionEstimator:
    """
    DeepFace를 사용한 얼굴 기반 감정 추정.
    GPU 사용 가능.
    """

    def __init__(self, model_name: str = "VGG-Face") -> None:
        self.model_name = model_name
        self.is_warm = False

    def warmup(self) -> None:
        """모델 워밍업"""
        self.is_warm = True

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        """
        BGR 프레임에서 감정 추정.
        
        Args:
            frame_bgr: OpenCV BGR 이미지 (numpy array, shape: (H, W, 3))
        
        Returns:
            감정 확률 분포 딕셔너리
        """
        if not self.is_warm:
            self.warmup()
        
        try:
            # DeepFace 분석 (emotion만, silent=True로 출력 최소화)
            result = DeepFace.analyze(
                frame_bgr,
                actions=["emotion"],
                enforce_detection=False,
                silent=True
            )
            
            # 결과가 리스트인 경우 첫 번째 요소 사용
            if isinstance(result, list):
                result = result[0]
            
            # 감정 확률 분포 추출
            emotion_dict = result.get("emotion", {})
            
            if not emotion_dict:
                raise ValueError("No emotion data in result")
            
            # 키 이름 정규화
            normalized = {}
            for key, value in emotion_dict.items():
                # DeepFace는 angry, disgust, fear, happy, sad, surprise, neutral 사용
                normalized[key.lower()] = float(value) / 100.0  # 0-100을 0-1로 변환
            
            return normalized
            
        except Exception as e:
            # 에러 상세 정보 출력
            print(f"DeepFace inference error: {e}")
            import traceback
            traceback.print_exc()
            
            # 얼굴 감지 실패 또는 기타 에러 시 중립 반환
            return {
                "neutral": 1.0,
                "happy": 0.0,
                "sad": 0.0,
                "angry": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "disgust": 0.0
            }