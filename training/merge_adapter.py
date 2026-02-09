def merge_lora(base_model_path: str, lora_path: str, out_path: str) -> None:
    """
    LoRA 병합 스텁. 실제 구현에서는 PEFT 가중치를 base에 병합합니다.
    """
    print(f"Merging LoRA: base={base_model_path}, lora={lora_path} -> {out_path}")


if __name__ == "__main__":
    merge_lora("./models/qwen-base", "./models/qwen-lora", "./models/qwen-merged")


