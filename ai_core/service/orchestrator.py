from typing import Dict, Optional
from ai_core.emotion.face_deepface import FaceEmotionEstimator
from ai_core.emotion.audio_emotion import AudioEmotionEstimator
from ai_core.emotion.text_emotion import TextEmotionEstimator
from ai_core.emotion.smoothing import average_softmax
from ai_core.router.tone_router import route_tone
from ai_core.io.schemas import TurnInput, TurnOutput, EmotionSnapshot
from ai_core.llm.conversation_history import ConversationHistory, ConversationSessionManager

# LLM 클라이언트 import (둘 다 시도)
VLLMOpenAIClient = None
TransformersLLMClient = None

try:
    from ai_core.llm.vllm_client import VLLMOpenAIClient
except ImportError:
    pass

try:
    from ai_core.llm.transformers_client import TransformersLLMClient
except ImportError:
    pass

if VLLMOpenAIClient is None and TransformersLLMClient is None:
    raise ImportError("vLLM과 Transformers 둘 다 사용할 수 없습니다. 최소 하나는 설치되어야 합니다.")

# 기본 백엔드 결정
DEFAULT_LLM_BACKEND = "vllm" if VLLMOpenAIClient is not None else "transformers"


class TurnOrchestrator:
    """
    멀티턴 대화 처리:
    - 멀티모달 감정 집계
    - 대화 히스토리 관리
    - 감정 변화 추적
    - 감정 기반 응답 스타일 프롬프팅
    - LLM 호출 (멀티턴 지원)
    """

    def __init__(
        self, 
        enable_multiturn: bool = True,
        llm_backend: Optional[str] = None,
        gpu_id: int = 0,
        load_in_8bit: bool = False,
        model_name: Optional[str] = None
    ) -> None:
        self.face = FaceEmotionEstimator()
        self.audio = AudioEmotionEstimator()
        self.text = TextEmotionEstimator()
        
        # LLM 클라이언트 초기화
        if llm_backend is None:
            llm_backend = DEFAULT_LLM_BACKEND
        
        # Transformers 클라이언트 옵션
        transformers_kwargs = {
            "gpu_id": gpu_id,
            "load_in_8bit": load_in_8bit
        }
        if model_name:
            transformers_kwargs["model_name"] = model_name
        
        if llm_backend == "vllm":
            if VLLMOpenAIClient is not None:
                try:
                    self.llm = VLLMOpenAIClient()
                except Exception as e:
                    if TransformersLLMClient is not None:
                        self.llm = TransformersLLMClient(**transformers_kwargs)
                    else:
                        raise RuntimeError("vLLM 초기화 실패, Transformers도 사용 불가")
            else:
                if TransformersLLMClient is not None:
                    self.llm = TransformersLLMClient(**transformers_kwargs)
                else:
                    raise RuntimeError("vLLM을 사용할 수 없고, Transformers도 사용 불가")
        else:  # transformers
            if TransformersLLMClient is not None:
                self.llm = TransformersLLMClient(**transformers_kwargs)
            else:
                raise RuntimeError("Transformers를 사용할 수 없습니다.")
        
        # 멀티턴 대화 지원
        self.enable_multiturn = enable_multiturn
        self.session_manager = ConversationSessionManager()
    
    def _get_emotion_response_guide(self, emotion: str) -> str:
        """
        감정별 응답 스타일 가이드 반환
        
        Args:
            emotion: 감정 레이블 (sad, angry, happy, anxious, fear, neutral 등)
        
        Returns:
            해당 감정에 맞는 응답 스타일 가이드
        """
        emotion_guides = {
            "sad": """
**슬픔(sad) 감정에 대한 응답 스타일:**
- 부드럽고 따뜻한 어조 사용
- 공감과 위로의 표현 ("정말 힘드시겠어요", "그런 마음이 드는 게 당연해요")
- 천천히, 차분하게 대화 이어가기
- 성급한 해결책 제시보다는 경청과 감정 인정 우선
- 적절한 침묵과 여유 제공
- 예: "많이 힘드셨군요. 그런 감정을 느끼는 것도 괜찮아요. 천천히 이야기 나눠봐요."
""",
            "angry": """
**분노(angry) 감정에 대한 응답 스타일:**
- 차분하면서도 수용적인 태도
- 감정을 인정하고 타당화 ("화가 나시는 게 충분히 이해돼요")
- 판단하지 않고 들어주기
- 감정 표출을 허용하되, 건설적인 방향으로 유도
- 공간과 자율성 존중
- 예: "화가 나시는 마음, 충분히 이해해요. 그 감정을 느끼셔도 괜찮아요. 어떤 부분이 가장 답답하신가요?"
""",
            "happy": """
**행복(happy) 감정에 대한 응답 스타일:**
- 밝고 긍정적인 어조
- 기쁨을 함께 나누고 축하하기
- 활기차고 따뜻한 반응
- 긍정적 변화나 성취 강화
- 앞으로의 희망적인 전망 언급
- 예: "정말 좋으시겠어요! 축하드려요. 그 기쁜 마음을 충분히 느끼세요. 어떤 기분이신지 더 들려주시겠어요?"
""",
            "anxious": """
**불안(anxious) 감정에 대한 응답 스타일:**
- 차분하고 안정적인 어조
- 불안을 정상화하고 수용 ("불안한 마음이 드는 건 자연스러워요")
- 구체적이고 실용적인 조언 제공
- 작은 단계로 나누어 제시
- 통제감과 예측 가능성 강조
- 예: "불안하시군요. 그런 감정을 느끼는 건 자연스러운 일이에요. 함께 차근차근 생각해 볼까요?"
""",
            "fear": """
**두려움(fear) 감정에 대한 응답 스타일:**
- 안전하고 보호적인 분위기 조성
- 두려움을 인정하고 공감
- 현재 순간에 집중하도록 도움
- 과거 경험이나 강점 상기
- 점진적이고 부드러운 접근
- 예: "무서우시겠어요. 그 두려움이 크게 느껴지시는군요. 지금 이 순간, 당신은 안전해요. 천천히 함께 이야기해봐요."
""",
            "surprised": """
**놀람(surprised) 감정에 대한 응답 스타일:**
- 상황을 이해하고 정리 도움
- 당황스러움에 공감
- 침착하게 상황 파악 지원
- 필요시 정보나 설명 제공
- 예: "예상치 못한 일이셨군요. 놀라셨겠어요. 천천히 상황을 정리해볼까요?"
""",
            "neutral": """
**중립(neutral) 감정에 대한 응답 스타일:**
- 자연스럽고 편안한 어조
- 열린 질문으로 대화 확장
- 사용자의 관심사 탐색
- 필요한 정보 제공
- 유연하게 대응
- 예: "네, 이해했어요. 혹시 더 나누고 싶은 이야기가 있으신가요?"
"""
        }
        
        return emotion_guides.get(emotion, emotion_guides["neutral"])

    def aggregate_emotion(self, ti: TurnInput) -> Dict[str, float]:
        dist_list = []
        if ti.frame_bgr is not None:
            dist_list.append(self.face.infer(ti.frame_bgr))
        if ti.audio_chunk is not None:
            dist_list.append(self.audio.infer(ti.audio_chunk))
        if ti.utterance:
            dist_list.append(self.text.infer(ti.utterance))
        return average_softmax(dist_list)
    
    def generate_initial_greeting(
        self,
        user_initial_text: str,
        user_emotion: Optional[str] = None
    ) -> str:
        """
        Step 2: 사용자 입력 기반 상담 시작 문장 생성
        
        Args:
            user_initial_text: 사용자가 입력한 초기 상담 내용
            user_emotion: 사용자가 선택한 초기 감정 (선택사항)
        
        Returns:
            상담 시작 인사 문장
        """
        # 감정이 제공된 경우 감정별 응답 가이드 포함
        emotion_guide = ""
        if user_emotion:
            emotion_guide = self._get_emotion_response_guide(user_emotion)
        
        system_prompt = f"""당신은 따뜻하고 공감적인 전문 상담사입니다.
사용자가 상담을 시작하려고 합니다.

{emotion_guide}

**상담 시작 시 지침:**
1. 따뜻하고 환영하는 인사로 시작
2. 사용자의 상황과 감정을 인정
3. 안전하고 편안한 분위기 조성
4. 자연스럽게 대화를 이어갈 수 있도록 열린 질문 포함
5. 2-3문장으로 간결하게 표현"""

        user_prompt = f"""사용자가 다음과 같이 상담을 시작했습니다:

"{user_initial_text}"
"""
        
        if user_emotion:
            user_prompt += f"\n사용자의 현재 감정: {user_emotion}\n"
        
        user_prompt += """
이에 맞는 따뜻한 상담 시작 문장을 생성해주세요.
자연스럽고 공감적인 어조로, 사용자가 편안하게 이야기를 이어갈 수 있도록 해주세요."""

        return self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=200,
            temperature=0.8  # 약간 더 창의적인 인사말
        )
    
    def start_session(
        self,
        session_id: str,
        user_initial_text: str,
        user_emotion: Optional[str] = None
    ) -> Dict:
        """
        Step 3: 상담 세션 시작 및 초기 대화 구성
        
        Args:
            session_id: 세션 ID
            user_initial_text: 사용자 초기 입력
            user_emotion: 사용자 초기 감정
        
        Returns:
            초기 응답 정보 (인사말, 세션 정보 등)
        """
        # 세션 생성 또는 가져오기
        history = self.session_manager.get_or_create_session(session_id)
        
        # 텍스트 감정 추출 (사용자가 감정을 선택하지 않은 경우)
        if not user_emotion:
            ti = TurnInput(utterance=user_initial_text)
            emotion_dist = self.aggregate_emotion(ti)
            user_emotion = max(emotion_dist.items(), key=lambda x: x[1])[0]
        else:
            # 사용자가 선택한 감정으로 간단한 분포 생성
            emotion_dist = {user_emotion: 0.9, "neutral": 0.1}
        
        # Step 2: 초기 인사말 생성
        initial_greeting = self.generate_initial_greeting(
            user_initial_text=user_initial_text,
            user_emotion=user_emotion
        )
        
        # 대화 히스토리에 첫 턴 저장
        route = route_tone(emotion_dist)
        history.add_turn(
            user_utterance=user_initial_text,
            assistant_response=initial_greeting,
            emotion=emotion_dist,
            tone=route["tone"],
            strategy=route["strategy"]
        )
        
        return {
            "session_id": session_id,
            "greeting": initial_greeting,
            "detected_emotion": user_emotion,
            "emotion_distribution": emotion_dist,
            "tone": route["tone"],
            "strategy": route["strategy"],
            "message": "상담 세션이 시작되었습니다."
        }

    def run(self, ti: TurnInput, session_id: Optional[str] = None) -> TurnOutput:
        """
        턴 처리 (멀티턴 지원)
        
        Args:
            ti: 턴 입력
            session_id: 세션 ID (멀티턴 대화를 위해 필요)
        
        Returns:
            TurnOutput
        """
        # 1. 감정 추출
        aggregated = self.aggregate_emotion(ti)
        emotion = EmotionSnapshot(aggregated=aggregated)
        route = route_tone(aggregated)
        
        # 2. 멀티턴 대화 처리
        if self.enable_multiturn and session_id:
            history = self.session_manager.get_or_create_session(session_id)
            reply = self._generate_multiturn_response(ti, aggregated, route, history)
        else:
            # 단일 턴 (하위 호환성)
            reply = self._generate_single_turn_response(ti, route)
        
        # 3. 대화 히스토리 저장
        if self.enable_multiturn and session_id:
            history.add_turn(
                user_utterance=ti.utterance,
                assistant_response=reply,
                emotion=aggregated,
                tone=route["tone"],
                strategy=route["strategy"]
            )

        return TurnOutput(
            response=reply,
            emotion=emotion,
            meta={
                "tone": route["tone"],
                "strategy": route["strategy"],
                "session_id": session_id,
                "turn_count": len(history.turns) if session_id else 1
            }
        )
    
    def _generate_single_turn_response(self, ti: TurnInput, route: Dict) -> str:
        """단일 턴 응답 생성 (기존 방식)"""
        system_prompt = "당신은 공감적이고 전문적인 상담 보조자입니다."
        user_prompt = f"[tone={route['tone']}, strategy={route['strategy']}] {ti.utterance}"
        return self.llm.chat(system_prompt, user_prompt)
    
    def _generate_multiturn_response(
        self,
        ti: TurnInput,
        emotion: Dict[str, float],
        route: Dict,
        history: ConversationHistory
    ) -> str:
        """
        멀티턴 응답 생성 (감정 기반 응답 스타일 프롬프팅)
        
        LoRA 대신 프롬프트로 감정에 맞는 응답 스타일 부여
        """
        # 감정 변화 추이
        emotion_trajectory = history.get_emotion_trajectory(n=5)
        current_emotion = max(emotion.items(), key=lambda x: x[1])[0]
        
        # neutral 제외한 주요 감정
        emotion_no_neutral = {k: v for k, v in emotion.items() if k != "neutral"}
        if emotion_no_neutral:
            primary_emotion = max(emotion_no_neutral.items(), key=lambda x: x[1])[0]
        else:
            primary_emotion = "neutral"
        
        # 감정별 응답 스타일 가이드
        emotion_guide = self._get_emotion_response_guide(primary_emotion)
        
        # 감정 변화 분석
        emotion_change_note = ""
        if len(emotion_trajectory) >= 2:
            prev_emotion = emotion_trajectory[-2] if len(emotion_trajectory) >= 2 else emotion_trajectory[-1]
            if prev_emotion != primary_emotion:
                emotion_change_note = f"""
**중요: 감정 변화 감지**
사용자의 감정이 '{prev_emotion}'에서 '{primary_emotion}'으로 변화했습니다.
이 변화를 인지하고 적절히 반응해주세요.
예: "조금 {primary_emotion} 감정이 느껴지시네요. 이전보다 어떤 변화가 있으신가요?"
"""
        
        # 시스템 프롬프트 (감정 기반 응답 스타일 강화)
        system_prompt = f"""당신은 공감적이고 전문적인 상담 보조자입니다. 한국어로 답변하세요.

**현재 상황:**
- 사용자의 주요 감정: {primary_emotion}
- 현재 감정 분포: {', '.join([f'{k}: {v*100:.1f}%' for k, v in sorted(emotion.items(), key=lambda x: -x[1])[:3]])}
- 최근 감정 변화: {' → '.join(emotion_trajectory) if emotion_trajectory else '상담 시작'}
- 권장 응답 톤: {route['tone']}
- 권장 전략: {route['strategy']}

{emotion_change_note}

{emotion_guide}

**핵심 지침:**
1. 위의 감정별 응답 스타일을 **반드시** 따르세요
2. 이전 대화의 흐름과 맥락을 기억하고 일관성 있게 대화하세요
3. 감정 변화가 있다면 이를 자연스럽게 언급하세요
4. 2-3문장으로 간결하게 응답하세요
5. 사용자가 편안하게 느낄 수 있도록 공감과 지지를 표현하세요"""

        # 대화 히스토리
        history_messages = history.get_messages_for_api()
        
        # 현재 사용자 입력
        current_user_prompt = f"[현재 감정: {primary_emotion}] {ti.utterance}"
        
        return self.llm.chat_multiturn(
            system_prompt=system_prompt,
            history_messages=history_messages,
            current_user_prompt=current_user_prompt,
            temperature=0.7,
            top_p=0.9
        )
    
    def clear_session(self, session_id: str) -> None:
        """세션 대화 히스토리 초기화"""
        self.session_manager.delete_session(session_id)


