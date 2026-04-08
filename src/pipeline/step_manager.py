"""
StepManager - CBT 5-Step 플래닝 관리
- 동적 플랜 지원 (DynamicPlanGenerator에서 생성한 맞춤형 플랜)
- 고정 템플릿 폴백 지원
- 턴 수 관리 (최소/최대)
- 스텝 전환 판단 (LLM 판단 + 사용자 확인)
- 상담사 첫 발화 생성

변경점 (기존 대비):
- __init__에서 DynamicPlanGenerator를 통해 플랜 생성 (LLM 맞춤형)
- get_current_system_prompt()가 동적 플랜의 goal/focus/key_questions 반영
- 기존 고정 템플릿은 폴백으로 유지
"""

from dynamic_plan import DynamicPlanGenerator, build_dynamic_system_prompt


# ============ 스텝 시작 시 상담사 첫 발화 프롬프트 ============

STEP_OPENING_PROMPTS = {
    1: (
        "내담자가 '{topic}' 주제로 상담을 시작합니다. "
        "내담자의 현재 감정은 {emotion}이고, 상세 고민은 '{detail}'입니다. "
        "따뜻하게 인사하고, 편하게 이야기할 수 있도록 안내하세요. "
        "짧고 자연스럽게 말하세요."
    ),
    2: (
        "이제 문제 탐색 단계로 넘어갑니다. "
        "지금까지의 이야기를 간단히 수용하면서, "
        "더 구체적으로 살펴보겠다고 자연스럽게 전환하세요. "
        "짧고 자연스럽게 말하세요."
    ),
    3: (
        "이제 사고 전환 단계로 넘어갑니다. "
        "앞서 나눈 이야기를 바탕으로, "
        "생각 패턴을 함께 살펴보자고 자연스럽게 전환하세요. "
        "짧고 자연스럽게 말하세요."
    ),
    4: (
        "이제 행동 계획 단계로 넘어갑니다. "
        "새로운 관점을 찾은 것을 격려하고, "
        "실천할 수 있는 것을 함께 생각해보자고 자연스럽게 전환하세요. "
        "짧고 자연스럽게 말하세요."
    ),
    5: (
        "이제 마무리 단계로 넘어갑니다. "
        "오늘 의미 있는 이야기를 나눴다고 말하며, "
        "함께 정리하자고 자연스럽게 전환하세요. "
        "짧고 자연스럽게 말하세요."
    ),
}


class StepManager:
    """CBT 5-Step 플래닝 관리"""

    def __init__(self, topic, emotion, detail, api_key=None, lora_switcher=None):
        """
        Args:
            topic: 상담 주제
            emotion: 초기 감정
            detail: 상세 고민
            api_key: OpenAI API 키 (GPT-4o-mini 플랜 생성용, 권장)
            lora_switcher: 로컬 LLM (API 없을 때 폴백)
        """
        self.topic = topic
        self.current_step = 1
        self.transition_proposed = False

        # 동적 플랜 생성 (우선순위: API → 로컬 LLM → 폴백)
        generator = DynamicPlanGenerator(api_key=api_key, lora_switcher=lora_switcher)
        self.plan = generator.generate(topic, emotion, detail)

        self.is_dynamic = self.plan.get("is_dynamic", False)

        if self.is_dynamic:
            print(f"[StepManager] 동적 플랜 생성 완료")
            print(f"  핵심 문제: {self.plan['analysis']['core_problem']}")
            print(f"  인지 패턴: {self.plan['analysis']['cognitive_pattern']}")
        else:
            print(f"[StepManager] 폴백 플랜 사용")

    # ─── 정보 조회 ───

    def get_current_step_info(self):
        """현재 스텝 정보 반환"""
        return self.plan['steps'][self.current_step - 1]

    def get_plan_for_frontend(self):
        """프론트 사이드바용 플랜 데이터 반환"""
        result = {
            "topic": self.plan['topic'],
            "initial_emotion": self.plan['initial_emotion'],
            "steps": [
                {
                    "step": s['step'],
                    "name": s['name'],
                    "status": s['status'],
                    "goal": s.get('goal', ''),
                }
                for s in self.plan['steps']
            ],
        }

        if 'analysis' in self.plan:
            result['analysis'] = self.plan['analysis']

        return result

    # ─── 턴 관리 ───

    def increment_turn(self):
        """현재 스텝의 턴 수 증가"""
        step = self.get_current_step_info()
        step['current_turns'] += 1

    def should_suggest_transition(self):
        """LLM에게 전환 제안을 유도할지 판단"""
        step = self.get_current_step_info()
        if step['current_turns'] < step['min_turns']:
            return False
        if step['current_turns'] >= step['max_turns']:
            return True
        return None

    # ─── 시스템 프롬프트 ───

    def get_current_system_prompt(self, current_emotion):
        """현재 스텝의 시스템 프롬프트 (동적 플랜의 goal/focus/질문 반영)"""
        step = self.get_current_step_info()

        return build_dynamic_system_prompt(
            step_data=step,
            topic=self.topic,
            current_emotion=current_emotion,
        )

    def get_step_opening_prompt(self, step_num, current_emotion):
        """스텝 시작 시 상담사 첫 발화용 프롬프트"""
        return STEP_OPENING_PROMPTS[step_num].format(
            topic=self.topic,
            emotion=current_emotion,
            detail=self.plan.get('detail', ''),
        )

    # ─── 스텝 전환 ───

    def check_user_response_for_transition(self, user_text):
        """사용자 응답에서 스텝 전환 동의 확인"""
        if not self.transition_proposed:
            return False

        positive_keywords = ['네', '좋아요', '넘어가요', '괜찮아요', '응', '그래요', '좋습니다']
        negative_keywords = ['아직', '아니요', '더', '잠깐', '아니', '좀 더']

        user_lower = user_text.strip()

        for kw in positive_keywords:
            if kw in user_lower:
                self.transition_proposed = False
                return True

        for kw in negative_keywords:
            if kw in user_lower:
                self.transition_proposed = False
                return False

        self.transition_proposed = False
        return False

    def detect_transition_proposal(self, response):
        """LLM 응답에서 스텝 전환 제안 감지"""
        transition_keywords = ['다음 단계로', '넘어가볼까요', '마무리 단계로', '살펴볼까요']
        for kw in transition_keywords:
            if kw in response:
                self.transition_proposed = True
                return True
        return False

    def advance_step(self):
        """다음 스텝으로 전환"""
        if self.current_step < 5:
            self.plan['steps'][self.current_step - 1]['status'] = 'completed'
            self.current_step += 1
            self.plan['steps'][self.current_step - 1]['status'] = 'active'
            return True
        return False

    def complete_session(self):
        """마지막 스텝 완료 처리"""
        self.plan['steps'][4]['status'] = 'completed'

    def force_next_step(self):
        """사용자가 버튼으로 강제 전환"""
        return self.advance_step()

    def is_completed(self):
        """모든 스텝 완료 여부"""
        return self.current_step == 5 and \
               self.plan['steps'][4]['status'] == 'completed'


# ============ 테스트 코드 ============
if __name__ == "__main__":
    print("=" * 60)
    print("StepManager 테스트 (동적 플랜 - 폴백 모드)")
    print("=" * 60)

    manager = StepManager(
        topic="직장 스트레스",
        emotion="angry",
        detail="상사가 제 성과를 자기 것처럼 가져가요",
        lora_switcher=None,
    )

    # 플랜 확인
    print(f"\n=== 플랜 ===")
    plan = manager.get_plan_for_frontend()
    if 'analysis' in plan:
        print(f"  분석: {plan['analysis']['core_problem']}")
    for step in plan['steps']:
        status = "●" if step['status'] == 'active' else "○"
        print(f"  {status} {step['step']}. {step['name']} — {step.get('goal', '')[:40]}")

    # 시스템 프롬프트 확인
    print(f"\n=== Step 1 시스템 프롬프트 ===")
    prompt = manager.get_current_system_prompt("angry")
    print(prompt)

    # 턴 진행
    print(f"\n=== 턴 진행 시뮬레이션 ===")
    manager.increment_turn()
    print(f"  턴 1: 전환? {manager.should_suggest_transition()}")

    manager.increment_turn()
    print(f"  턴 2: 전환? {manager.should_suggest_transition()}")

    # 전환
    manager.detect_transition_proposal("다음 단계로 넘어가볼까요?")
    agreed = manager.check_user_response_for_transition("네, 좋아요")
    if agreed:
        manager.advance_step()
        print(f"  → Step {manager.current_step}: {manager.get_current_step_info()['name']}")

    # Step 2 프롬프트
    print(f"\n=== Step 2 시스템 프롬프트 ===")
    print(manager.get_current_system_prompt("sad"))