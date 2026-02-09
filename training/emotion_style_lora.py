"""
감정별 응답 스타일 LoRA 파인튜닝

목표:
- 사용자 감정(sad, angry, happy 등)에 맞는 응답 스타일 학습
- 각 감정에 대한 적절한 공감 표현 및 톤 조절
"""

from typing import List, Dict, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from datasets import Dataset
import json


def prepare_emotion_dataset(data_path: str) -> Dataset:
    """
    감정별 대화 데이터셋 준비
    
    데이터 형식 예시 (JSONL):
    {
        "emotion": "sad",
        "user": "요즘 너무 힘들어요. 아무것도 하기 싫어요.",
        "assistant": "정말 힘드시겠어요. 그런 마음이 드는 건 자연스러운 일이에요. 천천히 이야기 나눠볼까요?"
    }
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Qwen 형식으로 포맷팅
    formatted_data = []
    for item in data:
        # 감정 정보를 시스템 프롬프트에 포함
        conversation = f"""<|im_start|>system
당신은 공감적인 상담 보조자입니다. 사용자의 감정은 '{item['emotion']}'입니다. 이에 맞게 응답하세요.<|im_end|>
<|im_start|>user
{item['user']}<|im_end|>
<|im_start|>assistant
{item['assistant']}<|im_end|>"""
        
        formatted_data.append({"text": conversation})
    
    return Dataset.from_list(formatted_data)


def create_lora_config() -> LoraConfig:
    """
    LoRA 설정
    
    - r: LoRA rank (낮을수록 파라미터 수 적음, 4-16 권장)
    - lora_alpha: LoRA scaling factor (보통 r의 2배)
    - target_modules: 어댑터를 적용할 레이어 (Qwen2 기준)
    - lora_dropout: 드롭아웃 비율
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # LoRA rank
        lora_alpha=16,  # scaling factor
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],  # Qwen2 attention & MLP layers
        bias="none",
        inference_mode=False
    )


def train_emotion_lora(
    base_model_name: str = "Qwen/Qwen2-7B-Instruct",
    data_path: str = "./data/emotion_conversations.jsonl",
    output_dir: str = "./models/qwen2-emotion-lora",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 4,
    max_length: int = 512,
    gpu_id: int = 0
) -> None:
    """
    감정 기반 응답 스타일 LoRA 파인튜닝
    
    Args:
        base_model_name: 베이스 모델 이름
        data_path: 학습 데이터 경로 (JSONL)
        output_dir: 출력 디렉토리
        num_epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        gradient_accumulation_steps: 그래디언트 누적 스텝
        max_length: 최대 시퀀스 길이
        gpu_id: 사용할 GPU 번호
    """
    
    # GPU 설정
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 LoRA 파인튜닝 시작")
    print(f"  베이스 모델: {base_model_name}")
    print(f"  데이터: {data_path}")
    print(f"  출력: {output_dir}")
    print(f"  디바이스: {device}")
    print()
    
    # 1. 토크나이저 로드
    print("📖 토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 베이스 모델 로드
    print("🤖 베이스 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True
    )
    
    # 3. LoRA 어댑터 적용
    print("🔧 LoRA 어댑터 적용 중...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. 데이터셋 준비
    print("📊 데이터셋 준비 중...")
    dataset = prepare_emotion_dataset(data_path)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 5. 학습 설정
    print("⚙️  학습 설정 중...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,  # bfloat16 사용 (A100, H100, L40S 등 지원)
        optim="adamw_torch",
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False
    )
    
    # 6. 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM이므로 MLM 비활성화
    )
    
    # 7. 트레이너
    print("🏋️  트레이너 초기화 중...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # 8. 학습 시작
    print("\n" + "="*80)
    print("🎯 학습 시작!")
    print("="*80 + "\n")
    
    trainer.train()
    
    # 9. LoRA 어댑터 저장
    print(f"\n💾 LoRA 어댑터 저장 중... -> {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ 학습 완료!")
    print(f"  LoRA 어댑터 위치: {output_dir}")
    print(f"  사용 방법:")
    print(f"    from peft import PeftModel")
    print(f"    base_model = AutoModelForCausalLM.from_pretrained('{base_model_name}')")
    print(f"    model = PeftModel.from_pretrained(base_model, '{output_dir}')")


def merge_lora_to_base(
    base_model_name: str,
    lora_adapter_path: str,
    output_merged_path: str
) -> None:
    """
    LoRA 어댑터를 베이스 모델과 병합하여 단일 모델 생성
    
    Args:
        base_model_name: 베이스 모델 이름
        lora_adapter_path: LoRA 어댑터 경로
        output_merged_path: 병합된 모델 저장 경로
    """
    print("🔀 LoRA 어댑터 병합 시작...")
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # LoRA 어댑터 로드
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # 병합
    merged_model = model.merge_and_unload()
    
    # 저장
    print(f"💾 병합된 모델 저장 중... -> {output_merged_path}")
    merged_model.save_pretrained(output_merged_path)
    
    # 토크나이저도 복사
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_merged_path)
    
    print("✅ 병합 완료!")


if __name__ == "__main__":
    # 예시 실행
    train_emotion_lora(
        base_model_name="Qwen/Qwen2-7B-Instruct",
        data_path="./data/emotion_conversations.jsonl",
        output_dir="./models/qwen2-emotion-lora",
        num_epochs=3,
        batch_size=2,
        learning_rate=2e-4,
        gradient_accumulation_steps=8,
        gpu_id=3  # GPU 3번 사용
    )
