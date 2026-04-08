# 텍스트 감정 추출 모델 테스트 파일
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델 로드
model_path = "C:/Users/junsu/Desktop/capstone-ai-counselin/models/base/text-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

emotion_names = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

print("=" * 50)
print("텍스트 감정 인식 테스트")
print("종료하려면 'quit' 입력")
print("=" * 50)

while True:
    text = input("\n입력: ")
    if text.lower() == 'quit':
        break

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

    print(f"\n  예측 감정: {emotion_names[probs.argmax().item()]} ({probs.max().item():.1%})")
    print(f"  감정별 점수:")
    for idx in probs.argsort(descending=True):
        emo = emotion_names[idx]
        score = probs[idx].item()
        bar = "█" * int(score * 30)
        print(f"    {emo:10s}: {score:5.1%} {bar}")