"""
Transformers 기반 LLM 클라이언트 (vLLM 대안)

vLLM 설치 없이 Hugging Face Transformers로 직접 사용
"""

from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TransformersLLMClient:
    """
    Hugging Face Transformers 기반 LLM 클라이언트
    vLLM이 없을 때 사용 가능
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        device: str = "cuda:0",
        gpu_id: int = 0,
        load_in_8bit: bool = False
    ):
        self.model_name = model_name
        
        # GPU 설정 (CUDA_VISIBLE_DEVICES 사용하지 않음)
        if torch.cuda.is_available():
            self.device = f"cuda:{gpu_id}"
        else:
            self.device = "cpu"
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 모델 로드 옵션
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if load_in_8bit:
            # 8비트 양자화 (메모리 절반으로 줄임)
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": gpu_id}  # 특정 GPU에 로드
        else:
            # 일반 로딩
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = {"": gpu_id}  # 특정 GPU에 로드
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """단일 턴 채팅"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._generate(messages, max_new_tokens, temperature, top_p)
    
    def chat_multiturn(
        self,
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        current_user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """멀티턴 채팅"""
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": current_user_prompt})
        
        return self._generate(messages, max_new_tokens, temperature, top_p)
    
    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """텍스트 생성"""
        # 템플릿 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 토크나이징
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
