"""
CounselingSession - 전체 상담 흐름 통합 관리
- StepManager: 5-Step 플래닝
- LoRASwitcher: 감정별 LoRA 스위칭
- HistoryManager: 대화 히스토리 요약 관리
- EmotionFusion: 멀티모달 감정 융합
- 감정 버퍼: 얼굴/음성/텍스트
"""

import time
from step_manager import StepManager
from lora_switcher import LoRASwitcher
from history_manager import HistoryManager
# from emotion.fusion import EmotionFusion  # 기존 fusion 모듈


class FaceEmotionBuffer:
    """얼굴 감정 버퍼 - 발화 중 1초마다 쌓임"""

    def __init__(self):
        self.buffer = []

    def add(self, emotion_scores):
        """DeepFace 결과 저장 (7개 감정 점수)"""
        self.buffer.append({
            'timestamp': time.time(),
            'scores': emotion_scores,
        })

    def aggregate(self):
        """버퍼 감정 평균 계산 후 클리어"""
        if not self.buffer:
            return None

        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        avg = {}
        for emo in emotions:
            avg[emo] = sum(b['scores'].get(emo, 0) for b in self.buffer) / len(self.buffer)

        count = len(self.buffer)
        self.buffer.clear()
        return avg, count

    def clear(self):
        self.buffer.clear()


class VoiceProcessor:
    """음성 처리 - 발화 중 청크 누적, 완료 후 한번에 처리"""

    def __init__(self):
        self.audio_chunks = []
        # self.emotion_model = load_wav2vec2_model()  # 실제 구현 시 로드

    def add_chunk(self, audio_chunk):
        """음성 청크 누적"""
        self.audio_chunks.append(audio_chunk)

    def get_full_audio(self):
        """전체 오디오 합치기"""
        if not self.audio_chunks:
            return None
        full_audio = b''.join(self.audio_chunks)
        return full_audio

    def get_emotion(self, full_audio):
        """전체 오디오 → 음성 감정 추출 (wav2vec2)"""
        # probs = self.emotion_model.predict(full_audio)
        # return probs
        pass

    def clear(self):
        self.audio_chunks.clear()


class TextProcessor:
    """텍스트 처리 - 전체 오디오 → STT → 텍스트 감정"""

    def __init__(self):
        pass
        # self.stt_model = load_whisper_model()
        # self.emotion_model = load_text_emotion_model()

    def transcribe(self, full_audio):
        """음성 → 텍스트 (Whisper small)"""
        # text = self.stt_model.transcribe(full_audio)
        # return text
        pass

    def get_emotion(self, text):
        """텍스트 → 감정 추출 (klue/bert)"""
        # probs = self.emotion_model.predict(text)
        # return probs
        pass


class EmotionFusion:
    """멀티모달 감정 융합 (기존 fusion.py 연결)"""

    WEIGHTS = {
        'text': 0.40,
        'voice': 0.35,
        'face': 0.25,
    }
    EMOTIONS = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

    def fuse(self, text_probs, voice_probs, face_probs):
        """3개 모달리티 감정 융합"""
        fused = {}
        for emo in self.EMOTIONS:
            fused[emo] = (
                text_probs.get(emo, 0) * self.WEIGHTS['text'] +
                voice_probs.get(emo, 0) * self.WEIGHTS['voice'] +
                face_probs.get(emo, 0) * self.WEIGHTS['face']
            )

        final_emotion = max(fused, key=fused.get)

        modality_emotions = {
            'text': max(text_probs, key=text_probs.get) if text_probs else None,
            'voice': max(voice_probs, key=voice_probs.get) if voice_probs else None,
            'face': max(face_probs, key=face_probs.get) if face_probs else None,
        }
        unique_emotions = set(e for e in modality_emotions.values() if e)
        conflict = len(unique_emotions) > 1

        # neutral 우회: 가중 융합 결과가 neutral이지만
        # 모달리티 중 하나라도 다른 감정이면 → 그 감정을 사용
        # (비언어적 감정 신호를 우선하여 실제 감정을 포착)
        lora_emotion = final_emotion
        if final_emotion == 'neutral':
            non_neutral = [
                e for e in modality_emotions.values()
                if e and e != 'neutral'
            ]
            if non_neutral:
                # 비중립 감정 중 가중치 점수가 가장 높은 것 선택
                # (voice, face 순 — 비언어적 신호 우선)
                best_non_neutral = max(
                    non_neutral,
                    key=lambda e: fused.get(e, 0)
                )
                lora_emotion = best_non_neutral

        return {
            'emotion': final_emotion,        # 실제 fusion 결과 (기록용)
            'lora_emotion': lora_emotion,     # LoRA 스위칭용 (neutral 우회 적용)
            'scores': fused,
            'modality_emotions': modality_emotions,
            'conflict': conflict,
        }


class CounselingSession:
    """전체 상담 세션 관리"""

    def __init__(self, topic, emotion, detail, lora_switcher=None):
        # 핵심 모듈
        self.step_manager = StepManager(topic, emotion, detail)
        self.lora_switcher = lora_switcher
        self.fusion = EmotionFusion()

        # 대화 히스토리 관리 (요약 포함)
        self.history_manager = HistoryManager(
            max_recent_turns=5,
            lora_switcher=lora_switcher,
        )

        # 감정 처리 모듈
        self.face_buffer = FaceEmotionBuffer()
        self.voice_processor = VoiceProcessor()
        self.text_processor = TextProcessor()

        # 상태
        self.emotion_history = []
        self.current_emotion = emotion

    # ─── 발화 중 (실시간) ───

    def on_video_frame(self, face_emotion_scores):
        """웹캠 프레임 수신 → 얼굴 감정 버퍼에 저장"""
        self.face_buffer.add(face_emotion_scores)

    def on_audio_chunk(self, audio_chunk):
        """음성 청크 수신 → 누적"""
        self.voice_processor.add_chunk(audio_chunk)

    # ─── 발화 완료 ───

    async def on_user_done(self):
        """사용자 발화 완료 시 전체 처리"""

        # 1. 전체 오디오 합치기
        full_audio = self.voice_processor.get_full_audio()

        # 2. 병렬 처리: (STT → 텍스트 감정) + (음성 감정)
        user_text = self.text_processor.transcribe(full_audio)
        text_emotion = self.text_processor.get_emotion(user_text)
        voice_emotion = self.voice_processor.get_emotion(full_audio)

        # 3. 얼굴 감정: 버퍼 평균
        face_result = self.face_buffer.aggregate()
        face_emotion = face_result[0] if face_result else None

        # 4. 멀티모달 융합
        fusion_result = self.fusion.fuse(
            text_probs=text_emotion or {},
            voice_probs=voice_emotion or {},
            face_probs=face_emotion or {},
        )
        current_emotion = fusion_result['emotion']           # 실제 fusion 결과 (기록용)
        lora_emotion = fusion_result['lora_emotion']         # LoRA 스위칭용 (neutral 우회)

        # 5. 감정 히스토리 기록
        turn_num = self.history_manager.get_status()['total_turns'] + 1
        self.emotion_history.append({
            'turn': turn_num,
            'step': self.step_manager.current_step,
            'step_name': self.step_manager.get_current_step_info()['name'],
            'emotion': current_emotion,
            'lora_emotion': lora_emotion,
            'scores': fusion_result['scores'],
            'modality_emotions': fusion_result['modality_emotions'],
            'conflict': fusion_result['conflict'],
        })
        self.current_emotion = current_emotion

        # 6. LLM 응답 생성 (LoRA 스위칭에는 lora_emotion 사용)
        response_data = await self.generate_response(user_text, lora_emotion)

        # 7. 버퍼 클리어
        self.voice_processor.clear()

        return response_data

    # ─── LLM 응답 생성 ───

    async def generate_response(self, user_text, current_emotion):
        """LLM 응답 생성 (스텝 + 턴 수 + LoRA 관리)"""

        # 1. 스텝 전환 확인
        if self.step_manager.check_user_response_for_transition(user_text):
            # 이전 스텝 대화 요약
            prev_step = self.step_manager.get_current_step_info()
            self.history_manager.on_step_transition(
                self.step_manager.current_step,
                prev_step['name'],
            )

            self.step_manager.advance_step()

            # LoRA 스위칭
            if self.lora_switcher:
                self.lora_switcher.switch(current_emotion)

            # 새 스텝 상담사 첫 발화 생성
            opening = self._generate_step_opening(
                self.step_manager.current_step, current_emotion
            )

            step_info = self.step_manager.get_current_step_info()
            return {
                "response": opening,
                "current_step": self.step_manager.current_step,
                "step_name": step_info['name'],
                "step_turns": f"0/{step_info['max_turns']}",
                "emotion": current_emotion,
                "is_step_opening": True,
                "plan": self.step_manager.get_plan_for_frontend(),
                "user_text": user_text,
            }

        # 2. 턴 수 증가
        self.step_manager.increment_turn()

        # 3. LoRA 스위칭 (감정 변화 시)
        if self.lora_switcher:
            self.lora_switcher.switch(current_emotion)

        # 4. 시스템 프롬프트
        system_prompt = self.step_manager.get_current_system_prompt(current_emotion)

        # 5. 사용자 메시지를 히스토리에 추가
        self.history_manager.add_user_message(user_text)

        # 6. 컨텍스트 구성 (이전 스텝 요약 + 최근 대화)
        messages = self.history_manager.build_context_messages(system_prompt)

        # 7. LLM 응답 생성
        if self.lora_switcher:
            response = self.lora_switcher.generate(messages)
        else:
            response = "[LLM 미연결] 테스트 응답입니다."

        # 8. 상담사 응답을 히스토리에 추가
        self.history_manager.add_assistant_message(response)

        # 9. 스텝 전환 제안 감지
        self.step_manager.detect_transition_proposal(response)

        # 10. 마지막 스텝 완료 체크
        if self.step_manager.current_step == 5:
            step_info = self.step_manager.get_current_step_info()
            if step_info['current_turns'] >= step_info['max_turns']:
                self.step_manager.complete_session()

        step_info = self.step_manager.get_current_step_info()
        return {
            "response": response,
            "current_step": self.step_manager.current_step,
            "step_name": step_info['name'],
            "step_turns": f"{step_info['current_turns']}/{step_info['max_turns']}",
            "emotion": current_emotion,
            "is_step_opening": False,
            "plan": self.step_manager.get_plan_for_frontend(),
            "user_text": user_text,
        }

    # ─── 상담사 첫 발화 ───

    def _generate_step_opening(self, step_num, current_emotion):
        """스텝 시작 시 상담사 첫 발화 생성"""
        opening_prompt = self.step_manager.get_step_opening_prompt(
            step_num, current_emotion
        )

        # 이전 스텝 요약을 포함한 컨텍스트 구성
        messages = self.history_manager.build_context_messages(opening_prompt)

        if self.lora_switcher:
            opening = self.lora_switcher.generate(messages)
        else:
            opening = f"[Step {step_num} 시작] 테스트 첫 발화입니다."

        self.history_manager.add_assistant_message(opening)
        return opening

    # ─── 초기화 ───

    def start_session(self):
        """상담 세션 시작 → Step 1 상담사 첫 발화 반환"""
        if self.lora_switcher:
            self.lora_switcher.switch(self.current_emotion)

        opening = self._generate_step_opening(1, self.current_emotion)

        return {
            "opening_message": opening,
            "plan": self.step_manager.get_plan_for_frontend(),
            "emotion": self.current_emotion,
        }

    # ─── 강제 스텝 전환 (UI 버튼) ───

    def force_next_step(self, current_emotion):
        """사용자가 '다음' 버튼 클릭 시"""
        # 현재 스텝 요약
        prev_step = self.step_manager.get_current_step_info()
        self.history_manager.on_step_transition(
            self.step_manager.current_step,
            prev_step['name'],
        )

        if self.step_manager.force_next_step():
            opening = self._generate_step_opening(
                self.step_manager.current_step, current_emotion
            )
            step_info = self.step_manager.get_current_step_info()
            return {
                "response": opening,
                "current_step": self.step_manager.current_step,
                "step_name": step_info['name'],
                "step_turns": f"0/{step_info['max_turns']}",
                "emotion": current_emotion,
                "is_step_opening": True,
                "plan": self.step_manager.get_plan_for_frontend(),
            }
        return None

    # ─── 상태 조회 ───

    def get_status(self):
        """현재 세션 상태"""
        history_status = self.history_manager.get_status()
        return {
            "current_step": self.step_manager.current_step,
            "step_name": self.step_manager.get_current_step_info()['name'],
            "current_emotion": self.current_emotion,
            "total_turns": history_status['total_turns'],
            "is_completed": self.step_manager.is_completed(),
            "emotion_history": self.emotion_history,
            "step_summaries": self.history_manager.get_step_summaries(),
        }

    @property
    def chat_history(self):
        """리포트 등 외부에서 전체 히스토리 접근용"""
        return self.history_manager.get_full_history()


# ============ 테스트 코드 (GPU 없이) ============
if __name__ == "__main__":
    session = CounselingSession(
        topic="직장 스트레스",
        emotion="분노",
        detail="상사가 제 성과를 자기 것처럼 가져가요",
        lora_switcher=None,
    )

    print("=== 세션 시작 ===")
    start_data = session.start_session()
    print(f"  첫 발화: {start_data['opening_message']}")
    print(f"  플랜:")
    for step in start_data['plan']['steps']:
        status = "●" if step['status'] == 'active' else "○"
        print(f"    {status} {step['step']}. {step['name']}")

    print(f"\n=== 상태 ===")
    status = session.get_status()
    print(f"  스텝: {status['step_name']}")
    print(f"  감정: {status['current_emotion']}")
    print(f"  완료: {status['is_completed']}")
    print(f"  히스토리 관리: {session.history_manager.get_status()}")