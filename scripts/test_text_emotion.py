# scripts/test_text_emotion.py
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.emotion.text_emotion import TextEmotionEstimator

def test_text_emotion(text: str):
    """텍스트 감정 추정 테스트"""
    estimator = TextEmotionEstimator()
    
    emotion = estimator.infer(text)
    
    print(f"입력 텍스트: {text}")
    print("감정 분포:")
    for key, value in sorted(emotion.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {value*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_text_emotion(" ".join(sys.argv[1:]))
    else:
        # 테스트 예시
        test_texts = [
            "요즘 너무 우울하고 힘들어요",
            "오늘 정말 기분이 좋아요!",
            "화가 나서 참을 수가 없어요",
            "무서워서 잠을 못 자겠어요"
        ]
        for text in test_texts:
            test_text_emotion(text)
            print()