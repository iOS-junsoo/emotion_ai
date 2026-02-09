from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np


class EmotionSnapshot(BaseModel):
    face: Optional[Dict[str, float]] = Field(default=None)
    audio: Optional[Dict[str, float]] = Field(default=None)
    text: Optional[Dict[str, float]] = Field(default=None)
    aggregated: Optional[Dict[str, float]] = Field(default=None)


class TurnInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    utterance: str
    audio_chunk: Optional[List[float]] = None
    frame_bgr: Optional[np.ndarray] = None  # OpenCV BGR 이미지 (numpy array)


class TurnOutput(BaseModel):
    response: str
    emotion: EmotionSnapshot
    meta: Dict[str, Any] = Field(default_factory=dict)  # str -> Any로 변경하여 유연성 확보