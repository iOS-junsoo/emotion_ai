"""
LoRA Switcher Module - 감정별 LoRA 동적 스위칭
==============================================
Multi-adapter 방식: 7개 감정 LoRA를 미리 로드하고 set_adapter로 즉시 전환

사용법:
    switcher = LoRASwitcher(base_model_name, cbt_adapter_path, lora_dir)
    switcher.load_all()
    
    # 매 턴마다
    switcher.switch("sad")
    response = switcher.generate(messages)
"""

import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class LoRASwitcher:
    """감정별 LoRA 동적 스위칭 모듈"""

    EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    DEFAULT_EMOTION = "neutral"

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        cbt_adapter_path: str = None,
        lora_dir: str = None,
        load_in_4bit: bool = True,
    ):
        """
        Args:
            base_model_name: HuggingFace 베이스 모델 이름
            cbt_adapter_path: CBT LoRA 어댑터 경로 (merge 용)
            lora_dir: 감정별 LoRA가 저장된 디렉토리 (하위에 happy/, sad/, ... 폴더)
            load_in_4bit: 4bit 양자화 여부
        """
        self.base_model_name = base_model_name
        self.cbt_adapter_path = cbt_adapter_path
        self.lora_dir = lora_dir
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None
        self.current_emotion = None
        self.loaded_adapters = []  # 로드된 어댑터 이름 목록

    def load_all(self):
        """베이스 모델 + CBT merge + 감정별 LoRA 7개 전부 로드"""
        print("=" * 50)
        print("LoRA Switcher 초기화 시작")
        print("=" * 50)

        start = time.time()

        # 1. 토크나이저 로드
        print("\n[1/3] 토크나이저 로드...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 베이스 모델 로드 + CBT LoRA 병합
        print("[2/3] 베이스 모델 + CBT LoRA 병합...")
        bnb_config = None
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.cbt_adapter_path:
            print(f"  CBT LoRA 로드: {self.cbt_adapter_path}")
            base_model = PeftModel.from_pretrained(base_model, self.cbt_adapter_path)
            base_model = base_model.merge_and_unload()
            print("  CBT LoRA 병합 완료")

        # 3. 감정별 LoRA 멀티 어댑터 로드
        print("[3/3] 감정별 LoRA 로드...")

        first_adapter = True
        for emotion in self.EMOTIONS:
            adapter_path = os.path.join(self.lora_dir, emotion)

            if not os.path.exists(adapter_path):
                print(f"  [{emotion}] 경로 없음, 스킵: {adapter_path}")
                continue

            if first_adapter:
                # 첫 번째 어댑터: PeftModel로 감싸기
                self.model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    adapter_name=emotion,
                )
                first_adapter = False
            else:
                # 이후 어댑터: load_adapter로 추가
                self.model.load_adapter(adapter_path, adapter_name=emotion)

            self.loaded_adapters.append(emotion)
            print(f"  [{emotion}] 로드 완료 ✓")

        if not self.loaded_adapters:
            raise RuntimeError("로드된 LoRA 어댑터가 없습니다!")

        # 기본 어댑터 설정
        default = self.DEFAULT_EMOTION if self.DEFAULT_EMOTION in self.loaded_adapters else self.loaded_adapters[0]
        self.switch(default)
        self.model.eval()

        elapsed = time.time() - start
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        print(f"\n{'=' * 50}")
        print(f"LoRA Switcher 초기화 완료!")
        print(f"  로드된 어댑터: {self.loaded_adapters}")
        print(f"  현재 활성: {self.current_emotion}")
        print(f"  소요 시간: {elapsed:.1f}초")
        print(f"  GPU 메모리: {vram:.2f} GB")
        print(f"{'=' * 50}")

    def switch(self, emotion: str) -> bool:
        """
        감정별 LoRA 전환

        Args:
            emotion: 전환할 감정 (happy, sad, angry, ...)

        Returns:
            True: 전환 성공, False: 이미 같은 감정이라 전환 불필요
        """
        if emotion == self.current_emotion:
            return False

        if emotion not in self.loaded_adapters:
            print(f"[경고] '{emotion}' 어댑터 없음. 유지: {self.current_emotion}")
            return False

        self.model.set_adapter(emotion)
        prev = self.current_emotion
        self.current_emotion = emotion
        print(f"[LoRA 전환] {prev} → {emotion}")
        return True

    def generate(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        현재 활성 LoRA로 응답 생성

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            repetition_penalty: 반복 패널티

        Returns:
            생성된 응답 텍스트
        """
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=repetition_penalty,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return response

    def get_status(self) -> dict:
        """현재 상태 반환"""
        return {
            "current_emotion": self.current_emotion,
            "loaded_adapters": self.loaded_adapters,
            "vram_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
        }


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    # ── 경로 설정 (로컬 환경에 맞게 수정) ──
    BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    CBT_ADAPTER = r"C:\Users\junsu\Desktop\capstone-ai-counselin\models\base\cbt-counselor"
    LORA_DIR = r"C:\Users\junsu\Desktop\capstone-ai-counselin\models\lora"

    # ── 초기화 ──
    switcher = LoRASwitcher(
        base_model_name=BASE_MODEL,
        cbt_adapter_path=CBT_ADAPTER,
        lora_dir=LORA_DIR,
    )
    switcher.load_all()

    # ── 테스트: 같은 입력, 다른 감정 ──
    test_input = "네 괜찮아요. 아무 일도 아니에요."

    print(f"\n테스트 입력: \"{test_input}\"")
    print("=" * 60)

    for emotion in ["neutral", "sad", "angry"]:
        switcher.switch(emotion)

        messages = [
            {
                "role": "system",
                "content": f"당신은 CBT 기반 심리 상담사입니다. 내담자의 현재 감정: {emotion}",
            },
            {"role": "user", "content": test_input},
        ]

        response = switcher.generate(messages)
        print(f"[{emotion:8s}] {response}")
        print()

    # ── 상태 확인 ──
    print(switcher.get_status())