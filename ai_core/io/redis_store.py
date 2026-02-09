from typing import Optional, Dict, Any
import os

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - 선택 의존성
    redis = None  # type: ignore


class RedisStore:
    """
    상태 저장/조회(선택) 스텁.
    """

    def __init__(self, url: Optional[str] = None) -> None:
        self.url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = redis.Redis.from_url(self.url) if redis else None

    def set_json(self, key: str, value: Dict[str, Any]) -> None:
        if not self.client:
            return
        self.client.set(key, str(value))

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        raw = self.client.get(key)
        if not raw:
            return None
        try:
            # 간단 파서 (실사용 시 json 모듈 권장)
            return eval(raw.decode("utf-8"))
        except Exception:
            return None


