# scripts/test_audio_emotion.py
import sys
from pathlib import Path
import soundfile as sf
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.emotion.audio_emotion import AudioEmotionEstimator

def test_audio_file(audio_path: str):
    """오디오 파일에서 감정 추정"""
    estimator = AudioEmotionEstimator()
    
    # 오디오 로드
    audio, sample_rate = sf.read(audio_path)
    
    # 모노로 변환 (스테레오인 경우)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 감정 추정
    emotion = estimator.infer(audio.tolist(), sample_rate)
    
    # 퍼센트로 출력
    print("감정 분포:")
    for key, value in sorted(emotion.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {value*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_audio_file(sys.argv[1])
    else:
        print("사용법: python scripts/test_audio_emotion.py <audio_file.wav>")