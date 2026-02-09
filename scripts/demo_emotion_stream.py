import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.emotion.face_deepface import FaceEmotionEstimator
from ai_core.emotion.audio_emotion import AudioEmotionEstimator
from ai_core.emotion.text_emotion import TextEmotionEstimator
from ai_core.emotion.smoothing import average_softmax


def main() -> None:
    face = FaceEmotionEstimator()
    audio = AudioEmotionEstimator()
    text = TextEmotionEstimator()

    # 더미 입력
    frame = b""  # 더미 프레임
    audio_chunk = [0.1] * 16000
    utterance = "오늘 좀 우울해요"

    dist = average_softmax([
        face.infer(frame),
        audio.infer(audio_chunk),
        text.infer(utterance),
    ])
    print("Aggregated emotion:", dist)


if __name__ == "__main__":
    main()


