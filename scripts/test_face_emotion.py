# scripts/test_face_emotion.py
import cv2
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.emotion.face_deepface import FaceEmotionEstimator

def format_emotion_percent(emotion_dict):
    """감정 분포를 퍼센트로 포맷팅"""
    formatted = {}
    for key, value in emotion_dict.items():
        formatted[key] = f"{value * 100:.2f}%"
    return formatted

# 방법 1: 웹캠 사용
def test_webcam():
    estimator = FaceEmotionEstimator()
    cap = cv2.VideoCapture(0)
    
    print("웹캠에서 얼굴 감정 추출 시작 (q로 종료)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR 프레임을 그대로 사용
        emotion = estimator.infer(frame)
        emotion_percent = format_emotion_percent(emotion)
        print(f"감정 분포: {emotion_percent}")
        
        # 화면에 표시
        cv2.imshow('Face Emotion', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 방법 2: 이미지 파일 사용
def test_image(image_path: str):
    estimator = FaceEmotionEstimator()
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    
    emotion = estimator.infer(frame)
    emotion_percent = format_emotion_percent(emotion)
    print(f"감정 분포: {emotion_percent}")
    
    # 가장 높은 감정 표시
    max_emotion = max(emotion.items(), key=lambda x: x[1])
    print(f"주요 감정: {max_emotion[0]} ({max_emotion[1]*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image(sys.argv[1])
    else:
        test_webcam()