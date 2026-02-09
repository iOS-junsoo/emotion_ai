from typing import Optional, List, Dict
import os
import requests


class VLLMOpenAIClient:
    """
    OpenAI 호환(vLLM 등) 서버 클라이언트.
    멀티턴 대화 지원.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model_name: Optional[str] = None) -> None:
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-xxx")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "Qwen2-7B-Instruct")

    def chat(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """단일 턴 대화 (하위 호환성 유지)"""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        resp = requests.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    
    def chat_multiturn(
        self,
        system_prompt: str,
        history_messages: List[Dict[str, str]],
        current_user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        멀티턴 대화 지원
        
        Args:
            system_prompt: 시스템 프롬프트
            history_messages: 이전 대화 히스토리 [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
            current_user_prompt: 현재 사용자 입력
            max_new_tokens: 최대 생성 토큰 수
            temperature: 샘플링 온도
            top_p: Top-p 샘플링
        
        Returns:
            LLM 응답
        """
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 메시지 구성: system + history + current user
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": current_user_prompt})
        
        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        resp = requests.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


