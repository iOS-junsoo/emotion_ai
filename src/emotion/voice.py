# 음성 감정 인식 테스트 파일
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

# 모델 로드
model_path = "C:/Users/junsu/Desktop/capstone-ai-counselin/models/base/voice-emotion"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
model.eval()

emotion_names = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

print("=" * 50)
print("음성 감정 인식 테스트")
print("종료하려면 'quit' 입력")
print("=" * 50)

while True:
    path = input("\nwav 파일 경로: ")
    if path.lower() == 'quit':
        break

    try:
        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

        print(f"\n  예측 감정: {emotion_names[probs.argmax().item()]} ({probs.max().item():.1%})")
        print(f"  감정별 점수:")
        for idx in probs.argsort(descending=True):
            emo = emotion_names[idx]
            score = probs[idx].item()
            bar = "█" * int(score * 30)
            print(f"    {emo:10s}: {score:5.1%} {bar}")
    except Exception as e:
        print(f"  에러: {e}")