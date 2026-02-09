from typing import Any


def train_lora(data_path: str, output_dir: str) -> Any:
    """
    TRL+PEFT QLoRA 학습 스텁.
    실제 구현에서는 TRL SFTTrainer 및 PEFT 설정을 구성합니다.
    """
    print(f"Training with data={data_path}, output_dir={output_dir}")
    return {"status": "ok"}


if __name__ == "__main__":
    train_lora("./data/processed/sft/train.jsonl", "./models/qwen-lora")


