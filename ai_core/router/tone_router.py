from typing import Dict


def route_tone(emotion_dist: Dict[str, float]) -> Dict[str, str]:
    """
    감정 분포를 바탕으로 응답 톤/전략을 간단 매핑.
    """
    if not emotion_dist:
        return {"tone": "neutral", "strategy": "reflective_listening"}
    dominant = max(emotion_dist.items(), key=lambda x: x[1])[0]
    mapping = {
        "happy": {"tone": "encouraging", "strategy": "reinforce_positive"},
        "sad": {"tone": "warm", "strategy": "validate_emotion"},
        "angry": {"tone": "calm", "strategy": "deescalate"},
        "neutral": {"tone": "neutral", "strategy": "reflective_listening"},
    }
    return mapping.get(dominant, {"tone": "neutral", "strategy": "reflective_listening"})


