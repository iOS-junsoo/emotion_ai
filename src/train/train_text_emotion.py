import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# ============ 1. 데이터 로드 ============
print("=" * 50)
print("1. 데이터 로드")
print("=" * 50)

df1 = pd.read_csv('data/raw/5차년도_2차.csv', encoding='cp949')
df2 = pd.read_csv('data/raw/5차년도.csv', encoding='cp949')
df = pd.concat([df1, df2], ignore_index=True)

emotion_map = {
    'happiness': 0, 'sadness': 1, 'angry': 2,
    'surprise': 3, 'fear': 4, 'disgust': 5, 'neutral': 6
}
emotion_names = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

df = df[['발화문', '상황']].dropna()
df['label'] = df['상황'].map(emotion_map)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"총 데이터: {len(df)}개")
print(f"감정 분포:\n{df['상황'].value_counts()}")

# ============ 2. 학습/검증 분할 ============
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['발화문'].tolist(), df['label'].tolist(),
    test_size=0.15, random_state=42, stratify=df['label'].tolist()
)
print(f"\n학습: {len(train_texts)}개, 검증: {len(val_texts)}개")

# ============ 3. 데이터셋 클래스 ============
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# ============ 4. 모델 로드 ============
print("\n" + "=" * 50)
print("4. 모델 로드")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7)
model.to(device)
print(f"디바이스: {device}")

# ============ 5. 학습 ============
print("\n" + "=" * 50)
print("5. 학습 시작")
print("=" * 50)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
num_epochs = 5
num_training_steps = num_epochs * len(train_loader)
scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

best_val_acc = 0

for epoch in range(num_epochs):
    # 학습
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)

    # 검증
    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total

    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Acc: {val_acc:.4f}")

    # 최고 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = "models/base/text-emotion"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  >> 최고 모델 저장! ({save_path})")

# ============ 6. 최종 평가 ============
print("\n" + "=" * 50)
print("6. 최종 평가")
print("=" * 50)
print(f"최고 검증 정확도: {best_val_acc:.4f}")
print(classification_report(all_labels, all_preds, target_names=emotion_names))