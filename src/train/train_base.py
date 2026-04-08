# CACTUS 상담사 모델(Qwen 2.5 3B) 학습 스크립트 -> 코랩으로 돌림

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import torch
import json
import os

# ============ 1. 모델 로드 ============
print("=" * 50)
print("1. 모델 로드")
print("=" * 50)

model_name = "Qwen/Qwen2.5-3B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)
print(f"모델 로드 완료! GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ============ 2. LoRA 설정 ============
print("\n" + "=" * 50)
print("2. LoRA 설정")
print("=" * 50)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============ 3. 데이터 로드 ============
print("\n" + "=" * 50)
print("3. 데이터 로드")
print("=" * 50)

def load_and_format(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted = []
    for item in data:
        text = tokenizer.apply_chat_template(
            item['messages'],
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted.append({"text": text})
    
    return Dataset.from_list(formatted)

train_dataset = load_and_format("data/processed/cactus_train.json")
val_dataset = load_and_format("data/processed/cactus_val.json")

print(f"학습 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")

sample_tokens = tokenizer(train_dataset[0]['text'], return_tensors="pt")
print(f"첫 번째 샘플 토큰 수: {sample_tokens.input_ids.shape[1]}")

# ============ 4. 학습 설정 ============
print("\n" + "=" * 50)
print("4. 학습 시작")
print("=" * 50)

output_dir = "models/base/cbt-counselor"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=50,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

total_steps = (len(train_dataset) // (1 * 8)) * 3
print(f"예상 총 스텝: {total_steps}")
print(f"GPU 메모리: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

trainer.train()

# ============ 5. 모델 저장 ============
print("\n" + "=" * 50)
print("5. 모델 저장")
print("=" * 50)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"모델 저장 완료: {output_dir}")