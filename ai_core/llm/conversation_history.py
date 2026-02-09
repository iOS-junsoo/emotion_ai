from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """대화의 한 턴"""
    turn_id: int
    timestamp: datetime
    user_utterance: str
    assistant_response: str
    emotion: Dict[str, float]  # 해당 턴의 감정 분포
    emotion_label: str  # 주요 감정 레이블 (예: "sad", "happy")
    tone: str  # 라우팅된 톤
    strategy: str  # 라우팅된 전략


@dataclass
class ConversationHistory:
    """대화 히스토리 관리"""
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    max_history_turns: int = 10  # 최대 보관 턴 수
    
    def add_turn(
        self,
        user_utterance: str,
        assistant_response: str,
        emotion: Dict[str, float],
        tone: str,
        strategy: str
    ) -> None:
        """새로운 턴 추가"""
        # 주요 감정 추출
        emotion_label = max(emotion.items(), key=lambda x: x[1])[0]
        
        turn = ConversationTurn(
            turn_id=len(self.turns) + 1,
            timestamp=datetime.now(),
            user_utterance=user_utterance,
            assistant_response=assistant_response,
            emotion=emotion,
            emotion_label=emotion_label,
            tone=tone,
            strategy=strategy
        )
        
        self.turns.append(turn)
        
        # 최대 턴 수 제한
        if len(self.turns) > self.max_history_turns:
            self.turns = self.turns[-self.max_history_turns:]
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """최근 n개 턴 반환"""
        return self.turns[-n:] if self.turns else []
    
    def get_emotion_trajectory(self, n: int = 5) -> List[str]:
        """최근 n개 턴의 감정 변화 추이"""
        recent = self.get_recent_turns(n)
        return [turn.emotion_label for turn in recent]
    
    def format_for_prompt(self, include_emotions: bool = True) -> str:
        """
        LLM 프롬프트용 대화 히스토리 포맷팅
        
        Args:
            include_emotions: 감정 정보 포함 여부
        
        Returns:
            포맷된 대화 히스토리 문자열
        """
        if not self.turns:
            return "대화 시작"
        
        lines = ["=== 이전 대화 ==="]
        for turn in self.get_recent_turns(5):  # 최근 5턴
            if include_emotions:
                lines.append(f"[{turn.emotion_label}] 사용자: {turn.user_utterance}")
            else:
                lines.append(f"사용자: {turn.user_utterance}")
            lines.append(f"상담사: {turn.assistant_response}")
        
        lines.append("==================")
        return "\n".join(lines)
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """
        OpenAI API 형식의 messages 배열 생성
        
        Returns:
            [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
        """
        messages = []
        for turn in self.get_recent_turns(5):  # 최근 5턴
            messages.append({
                "role": "user",
                "content": f"[감정: {turn.emotion_label}] {turn.user_utterance}"
            })
            messages.append({
                "role": "assistant",
                "content": turn.assistant_response
            })
        return messages
    
    def clear(self) -> None:
        """대화 히스토리 초기화"""
        self.turns.clear()


class ConversationSessionManager:
    """여러 세션의 대화 히스토리 관리"""
    
    def __init__(self) -> None:
        self.sessions: Dict[str, ConversationHistory] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationHistory:
        """세션 가져오기 또는 생성"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationHistory(session_id=session_id)
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str) -> None:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
