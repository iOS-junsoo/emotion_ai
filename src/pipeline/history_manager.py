"""
HistoryManager - 대화 히스토리 요약 관리
==========================================
- 스텝 내: 최근 N턴 유지, 이전 턴은 요약으로 압축
- 스텝 전환 시: 해당 스텝 대화를 LLM으로 요약
- 다음 스텝: 이전 스텝 요약 + 현재 스텝 대화만 컨텍스트에 포함

토큰 예산 (Qwen 2.5 3B 기준):
  컨텍스트: 32,768 토큰
  system prompt: ~300 토큰
  생성 여유: ~512 토큰
  이전 스텝 요약: ~500 토큰
  현재 스텝 대화: ~2,000 토큰 (최근 5턴)
  → 충분한 여유 확보 + 추론 속도 최적화

위치: src/pipeline/history_manager.py
"""


# 스텝 요약 생성용 프롬프트
STEP_SUMMARY_PROMPT = """다음은 CBT 상담의 '{step_name}' 단계에서 나눈 대화입니다.
이 대화의 핵심 내용을 3~5줄로 요약해주세요.
반드시 포함할 내용:
- 내담자의 주요 발언/감정
- 상담사가 파악한 핵심 포인트
- 이 단계에서 도출된 결론이나 발견

[대화 내용]
{conversation}

[요약]"""

# 스텝 내 중간 요약 프롬프트
MID_STEP_SUMMARY_PROMPT = """다음은 CBT 상담 중 현재 단계에서 이전에 나눈 대화입니다.
핵심 내용만 2~3줄로 압축해주세요.

[대화 내용]
{conversation}

[압축 요약]"""


class HistoryManager:
    """대화 히스토리 관리 + 요약"""

    def __init__(self, max_recent_turns=5, lora_switcher=None):
        """
        Args:
            max_recent_turns: 현재 스텝에서 유지할 최근 턴 수
            lora_switcher: LLM 요약 생성용 (None이면 간단 요약)
        """
        self.max_recent_turns = max_recent_turns
        self.lora_switcher = lora_switcher

        # 스텝별 요약 저장
        self.step_summaries = {}  # {step_num: "요약 텍스트"}

        # 현재 스텝 대화
        self.current_step_history = []  # [{"role": "user/assistant", "content": "..."}]

        # 현재 스텝 내 이전 턴 요약 (중간 요약)
        self.current_step_early_summary = None

        # 전체 원본 히스토리 (리포트용)
        self.full_history = []

    # ─── 대화 추가 ───

    def add_message(self, role, content):
        """메시지 추가"""
        msg = {"role": role, "content": content}
        self.current_step_history.append(msg)
        self.full_history.append(msg)

        # 현재 스텝 대화가 길어지면 중간 요약
        self._check_mid_step_summary()

    def add_user_message(self, text):
        """사용자 메시지 추가"""
        self.add_message("user", text)

    def add_assistant_message(self, text):
        """상담사 메시지 추가"""
        self.add_message("assistant", text)

    # ─── 스텝 전환 ───

    def on_step_transition(self, completed_step_num, step_name):
        """
        스텝 전환 시 호출 — 현재 스텝 대화를 요약하고 초기화

        Args:
            completed_step_num: 완료된 스텝 번호 (1~5)
            step_name: 완료된 스텝 이름 ("공감 형성" 등)
        """
        # 현재 스텝 대화를 요약
        summary = self._summarize_step(step_name)
        self.step_summaries[completed_step_num] = {
            "step_num": completed_step_num,
            "step_name": step_name,
            "summary": summary,
            "turn_count": len(self.current_step_history) // 2,
        }

        # 현재 스텝 초기화
        self.current_step_history = []
        self.current_step_early_summary = None

    # ─── 컨텍스트 구성 ───

    def build_context_messages(self, system_prompt):
        """
        LLM에 보낼 메시지 리스트 구성

        구조:
        1. system prompt (현재 스텝 지시 + 감정 + 턴 힌트)
        2. 이전 스텝 요약 (있으면)
        3. 현재 스텝 중간 요약 (있으면)
        4. 최근 N턴 대화

        Returns:
            list: [{"role": "...", "content": "..."}]
        """
        messages = []

        # 1. 시스템 프롬프트 + 이전 스텝 요약 통합
        enriched_prompt = system_prompt

        # 이전 스텝 요약이 있으면 시스템 프롬프트에 포함
        if self.step_summaries:
            summary_text = self._format_previous_summaries()
            enriched_prompt += f"\n\n[이전 상담 요약]\n{summary_text}"

        # 현재 스텝 중간 요약이 있으면 포함
        if self.current_step_early_summary:
            enriched_prompt += f"\n\n[현재 단계 이전 대화 요약]\n{self.current_step_early_summary}"

        messages.append({"role": "system", "content": enriched_prompt})

        # 2. 최근 N턴 대화
        recent = self._get_recent_turns()
        messages.extend(recent)

        return messages

    # ─── 내부 메서드 ───

    def _get_recent_turns(self):
        """최근 N턴 대화 반환 (user + assistant 쌍 기준)"""
        max_messages = self.max_recent_turns * 2  # user + assistant
        if len(self.current_step_history) <= max_messages:
            return list(self.current_step_history)
        return list(self.current_step_history[-max_messages:])

    def _check_mid_step_summary(self):
        """현재 스텝 대화가 길면 중간 요약 생성"""
        max_messages = self.max_recent_turns * 2
        total = len(self.current_step_history)

        # 최근 N턴을 넘는 이전 대화가 있으면 중간 요약
        if total > max_messages + 2:  # +2는 여유분
            early_messages = self.current_step_history[:total - max_messages]
            self.current_step_early_summary = self._summarize_conversation(
                early_messages, MID_STEP_SUMMARY_PROMPT
            )

    def _summarize_step(self, step_name):
        """스텝 전체 대화 요약"""
        if not self.current_step_history:
            return f"{step_name} 단계: 대화 없음"

        conversation_text = self._format_conversation(self.current_step_history)

        # LLM으로 요약
        if self.lora_switcher:
            prompt = STEP_SUMMARY_PROMPT.format(
                step_name=step_name,
                conversation=conversation_text[:2000],  # 너무 길면 자르기
            )
            messages = [{"role": "user", "content": prompt}]
            summary = self.lora_switcher.generate(
                messages, max_new_tokens=200, temperature=0.3
            )
            return summary

        # LLM 없으면 간단 요약 (키워드 기반)
        return self._simple_summary(step_name)

    def _summarize_conversation(self, messages, prompt_template):
        """대화 목록을 요약"""
        conversation_text = self._format_conversation(messages)

        if self.lora_switcher:
            prompt = prompt_template.format(conversation=conversation_text[:1500])
            msgs = [{"role": "user", "content": prompt}]
            return self.lora_switcher.generate(
                msgs, max_new_tokens=150, temperature=0.3
            )

        # LLM 없으면 마지막 발화 기반 간단 요약
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        if user_msgs:
            return f"내담자 주요 발언: {user_msgs[-1][:100]}..."
        return "이전 대화 내용"

    def _simple_summary(self, step_name):
        """LLM 없이 간단 요약 (테스트용)"""
        user_msgs = [
            m["content"]
            for m in self.current_step_history
            if m["role"] == "user"
        ]
        turn_count = len(user_msgs)

        if not user_msgs:
            return f"{step_name}: 대화 없음"

        # 첫 번째와 마지막 사용자 발화로 요약
        first = user_msgs[0][:80]
        last = user_msgs[-1][:80] if len(user_msgs) > 1 else ""

        summary = f"{step_name} ({turn_count}턴): "
        summary += f"내담자 첫 발화 - \"{first}\""
        if last:
            summary += f" / 마지막 발화 - \"{last}\""

        return summary

    def _format_conversation(self, messages):
        """대화 메시지를 텍스트로 변환"""
        lines = []
        for msg in messages:
            role = "내담자" if msg["role"] == "user" else "상담사"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def _format_previous_summaries(self):
        """이전 스텝 요약들을 텍스트로 포맷"""
        lines = []
        for step_num in sorted(self.step_summaries.keys()):
            info = self.step_summaries[step_num]
            lines.append(
                f"[Step {info['step_num']}. {info['step_name']}] "
                f"{info['summary']}"
            )
        return "\n".join(lines)

    # ─── 상태 조회 ───

    def get_full_history(self):
        """전체 원본 히스토리 (리포트용)"""
        return list(self.full_history)

    def get_step_summaries(self):
        """스텝별 요약 반환"""
        return dict(self.step_summaries)

    def get_status(self):
        """현재 상태"""
        return {
            "completed_steps": len(self.step_summaries),
            "current_step_turns": len(self.current_step_history) // 2,
            "total_turns": len(self.full_history) // 2,
            "has_mid_summary": self.current_step_early_summary is not None,
        }


# ============ 테스트 코드 (GPU 없이) ============
if __name__ == "__main__":
    manager = HistoryManager(max_recent_turns=3, lora_switcher=None)

    print("=== Step 1: 공감 형성 ===")
    manager.add_assistant_message("안녕하세요. 직장에서 많이 힘드셨나 봐요.")
    manager.add_user_message("네, 상사가 계속 부당하게 대해서 정말 힘들어요.")
    manager.add_assistant_message("정말 속상하셨겠어요. 더 이야기해 주실 수 있나요?")
    manager.add_user_message("매일 야근하는데 인정도 안 해주고...")
    manager.add_assistant_message("충분히 화가 나실 만한 상황이네요.")
    manager.add_user_message("네, 그래서 요즘 잠도 못 자요.")

    # 컨텍스트 확인
    context = manager.build_context_messages("당신은 CBT 상담사입니다.")
    print(f"  컨텍스트 메시지 수: {len(context)}")
    for msg in context:
        preview = msg["content"][:80]
        print(f"  [{msg['role']}] {preview}...")

    # 스텝 전환
    print("\n=== Step 1 → Step 2 전환 ===")
    manager.on_step_transition(1, "공감 형성")
    print(f"  Step 1 요약: {manager.step_summaries[1]['summary']}")

    print("\n=== Step 2: 문제 탐색 ===")
    manager.add_assistant_message("조금 더 구체적으로 살펴볼게요.")
    manager.add_user_message("상사가 제 보고서를 자기 이름으로 제출했어요.")
    manager.add_assistant_message("그때 어떤 생각이 드셨어요?")
    manager.add_user_message("나는 무능하다는 생각이 들었어요.")

    # Step 2 컨텍스트 - 이전 스텝 요약이 포함됨
    context = manager.build_context_messages("당신은 CBT 상담사입니다. 문제를 탐색하세요.")
    print(f"\n  컨텍스트 메시지 수: {len(context)}")
    print(f"  시스템 프롬프트에 이전 요약 포함: {'이전 상담 요약' in context[0]['content']}")

    # 스텝 전환
    print("\n=== Step 2 → Step 3 전환 ===")
    manager.on_step_transition(2, "문제 탐색")
    print(f"  Step 2 요약: {manager.step_summaries[2]['summary']}")

    # 최종 상태
    print(f"\n=== 상태 ===")
    status = manager.get_status()
    print(f"  완료 스텝: {status['completed_steps']}")
    print(f"  전체 턴: {status['total_turns']}")
    print(f"  스텝별 요약:")
    for num, info in manager.get_step_summaries().items():
        print(f"    Step {num} ({info['step_name']}): {info['summary'][:60]}...")