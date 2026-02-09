from typing import Dict


class LocalQwenClient:
    """
    (옵션) 로컬 Transformers 기반 Qwen 호출 스텁.
    실제 구현에서는 pipeline 또는 generate를 호출합니다.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        # 더미 응답
        return f"[{self.model_name}] {system_prompt} :: {user_prompt}"


