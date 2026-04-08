"""
상담 리포트 생성 모듈
- Step 5 완료 후 호출
- 감정 변화 추이 + 상담 요약 + 행동 지침
- LLM으로 자연어 요약 생성
"""

from datetime import datetime


# 리포트 생성용 프롬프트
REPORT_PROMPT = """당신은 CBT 기반 심리 상담 리포트를 작성하는 전문가입니다.
아래 상담 내용을 바탕으로 리포트를 작성해주세요.

[상담 정보]
- 주제: {topic}
- 초기 감정: {initial_emotion}
- 최종 감정: {final_emotion}
- 총 턴 수: {total_turns}

[대화 내용]
{chat_history}

[턴별 감정 변화]
{emotion_timeline}

다음 항목을 포함하여 리포트를 작성해주세요:

1. 주요 고민 요약 (2~3줄)
2. 발견된 자동적 사고 패턴 (부정적 사고 패턴)
3. 상담을 통해 발견한 새로운 관점
4. 행동 계획 (구체적 실천 사항)
5. 상담사 종합 코멘트 (격려 메시지)

자연스러운 한국어로 작성하세요."""


class ReportGenerator:
    """상담 리포트 생성"""

    def __init__(self, lora_switcher=None):
        self.lora_switcher = lora_switcher

    def generate(self, session):
        """CounselingSession으로부터 리포트 생성"""

        # 기본 데이터 수집
        status = session.get_status()
        emotion_history = status['emotion_history']

        report_data = {
            "generated_at": datetime.now().isoformat(),
            "topic": session.step_manager.topic,
            "initial_emotion": session.step_manager.plan['initial_emotion'],
            "final_emotion": emotion_history[-1]['emotion'] if emotion_history else None,
            "total_turns": status['total_turns'],
            "emotion_timeline": self._build_emotion_timeline(emotion_history),
            "emotion_change": self._analyze_emotion_change(emotion_history),
            "step_summary": self._build_step_summary(emotion_history),
            "summary": self._generate_summary(session),
        }

        return report_data

    def _build_emotion_timeline(self, emotion_history):
        """턴별 감정 변화 타임라인"""
        timeline = []
        for entry in emotion_history:
            timeline.append({
                "turn": entry['turn'],
                "step": entry['step'],
                "step_name": entry['step_name'],
                "emotion": entry['emotion'],
                "conflict": entry['conflict'],
                "modality": entry.get('modality_emotions', {}),
            })
        return timeline

    def _analyze_emotion_change(self, emotion_history):
        """감정 변화 분석"""
        if not emotion_history:
            return {"improved": False, "summary": "데이터 없음"}

        start = emotion_history[0]['emotion']
        end = emotion_history[-1]['emotion']

        # 감정별 긍정/부정 점수
        emotion_valence = {
            'happy': 1.0,
            'neutral': 0.5,
            'surprise': 0.3,
            'sad': -0.5,
            'fear': -0.6,
            'disgust': -0.7,
            'angry': -0.8,
        }

        start_score = emotion_valence.get(start, 0)
        end_score = emotion_valence.get(end, 0)
        improved = end_score > start_score

        # 감정 전환 횟수
        changes = 0
        for i in range(1, len(emotion_history)):
            if emotion_history[i]['emotion'] != emotion_history[i-1]['emotion']:
                changes += 1

        return {
            "start_emotion": start,
            "end_emotion": end,
            "improved": improved,
            "change_count": changes,
            "start_score": start_score,
            "end_score": end_score,
        }

    def _build_step_summary(self, emotion_history):
        """스텝별 주요 감정 요약"""
        step_emotions = {}
        for entry in emotion_history:
            step = entry['step']
            if step not in step_emotions:
                step_emotions[step] = {
                    'name': entry['step_name'],
                    'emotions': [],
                    'turns': 0,
                }
            step_emotions[step]['emotions'].append(entry['emotion'])
            step_emotions[step]['turns'] += 1

        summary = []
        for step_num in sorted(step_emotions.keys()):
            data = step_emotions[step_num]
            # 가장 많이 나온 감정
            from collections import Counter
            most_common = Counter(data['emotions']).most_common(1)[0][0]
            summary.append({
                'step': step_num,
                'name': data['name'],
                'turns': data['turns'],
                'dominant_emotion': most_common,
            })

        return summary

    def _generate_summary(self, session):
        """LLM으로 자연어 리포트 생성"""
        # 대화 히스토리 텍스트화
        chat_text = ""
        for msg in session.chat_history:
            role = "내담자" if msg['role'] == 'user' else "상담사"
            chat_text += f"{role}: {msg['content']}\n"

        # 감정 타임라인 텍스트화
        emotion_text = ""
        for entry in session.emotion_history:
            emotion_text += (
                f"턴 {entry['turn']} (Step {entry['step']} {entry['step_name']}): "
                f"{entry['emotion']}"
            )
            if entry['conflict']:
                emotion_text += " (모달리티 간 충돌 감지)"
            emotion_text += "\n"

        # 프롬프트 구성
        status = session.get_status()
        prompt = REPORT_PROMPT.format(
            topic=session.step_manager.topic,
            initial_emotion=session.step_manager.plan['initial_emotion'],
            final_emotion=status['emotion_history'][-1]['emotion'] if status['emotion_history'] else "없음",
            total_turns=status['total_turns'],
            chat_history=chat_text[:3000],  # 너무 길면 자르기
            emotion_timeline=emotion_text,
        )

        # LLM으로 요약 생성
        if self.lora_switcher:
            messages = [{"role": "user", "content": prompt}]
            summary_text = self.lora_switcher.generate(messages, max_new_tokens=500)
            return summary_text
        else:
            return "[LLM 미연결] 리포트 요약은 LLM 연결 후 생성됩니다."

    def to_markdown(self, report_data):
        """리포트를 마크다운으로 변환"""
        md = f"""# 상담 리포트

**생성일시**: {report_data['generated_at']}
**상담 주제**: {report_data['topic']}
**총 턴 수**: {report_data['total_turns']}턴

---

## 감정 변화

| 구분 | 감정 |
|------|------|
| 시작 | {report_data['emotion_change']['start_emotion']} |
| 종료 | {report_data['emotion_change']['end_emotion']} |
| 개선 여부 | {'✅ 개선됨' if report_data['emotion_change']['improved'] else '⚠️ 추가 상담 권장'} |
| 감정 전환 횟수 | {report_data['emotion_change']['change_count']}회 |

---

## 감정 변화 타임라인

"""
        for entry in report_data['emotion_timeline']:
            conflict = " ⚠️" if entry['conflict'] else ""
            md += f"- **턴 {entry['turn']}** ({entry['step_name']}): {entry['emotion']}{conflict}\n"

        md += "\n---\n\n## 스텝별 요약\n\n"

        for step in report_data['step_summary']:
            md += f"- **{step['name']}** ({step['turns']}턴): 주요 감정 - {step['dominant_emotion']}\n"

        md += f"\n---\n\n## 상담 요약\n\n{report_data['summary']}\n"

        return md


# ============ 테스트 코드 ============
if __name__ == "__main__":
    from counseling_session import CounselingSession

    # 세션 생성 (GPU 없이)
    session = CounselingSession(
        topic="직장 스트레스",
        emotion="분노",
        detail="상사가 제 성과를 자기 것처럼 가져가요",
        lora_switcher=None,
    )

    # 테스트용 가짜 데이터 주입
    session.chat_history = [
        {"role": "assistant", "content": "안녕하세요. 직장에서 많이 힘드셨나 봐요."},
        {"role": "user", "content": "네, 상사가 계속 부당하게 대해서..."},
        {"role": "assistant", "content": "정말 속상하셨겠어요."},
        {"role": "user", "content": "솔직히 제가 부족한 건가 싶기도 해요."},
        {"role": "assistant", "content": "정말 그렇게 생각하세요? 다른 가능성은 없을까요?"},
        {"role": "user", "content": "음... 동료들은 제가 잘한다고 해주긴 해요."},
        {"role": "assistant", "content": "이번 주에 해볼 수 있는 작은 것이 있을까요?"},
        {"role": "user", "content": "퇴근 시간을 지켜보려고요."},
    ]

    session.emotion_history = [
        {"turn": 1, "step": 1, "step_name": "공감 형성", "emotion": "angry", "scores": {}, "modality_emotions": {"text": "angry", "voice": "angry", "face": "angry"}, "conflict": False},
        {"turn": 2, "step": 2, "step_name": "문제 탐색", "emotion": "sad", "scores": {}, "modality_emotions": {"text": "sad", "voice": "sad", "face": "angry"}, "conflict": True},
        {"turn": 3, "step": 3, "step_name": "사고 전환", "emotion": "neutral", "scores": {}, "modality_emotions": {"text": "neutral", "voice": "neutral", "face": "neutral"}, "conflict": False},
        {"turn": 4, "step": 4, "step_name": "행동 계획", "emotion": "happy", "scores": {}, "modality_emotions": {"text": "happy", "voice": "neutral", "face": "happy"}, "conflict": True},
    ]

    # 리포트 생성
    generator = ReportGenerator(lora_switcher=None)
    report = generator.generate(session)

    # 마크다운 출력
    md = generator.to_markdown(report)
    print(md)
