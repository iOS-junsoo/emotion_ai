# 4-1. CBT 5-Step 플래닝 설계

## 5-Step 구조 (CACTUS 기반)

| Step | 이름 | 목적 | 최소 턴 | 최대 턴 | 시스템 프롬프트 핵심 |
|------|------|------|--------|--------|-------------------|
| 1 | Rapport Building | 신뢰 형성, 공감 | 2 | 5 | 내담자의 이야기를 경청하고 공감하세요. 판단하지 마세요. |
| 2 | Assessment | 문제/감정/상황 파악 | 3 | 7 | 구체적 상황, 감정, 생각을 탐색하세요. |
| 3 | Cognitive Restructuring | 부정적 사고 재구조화 | 3 | 8 | 자동적 사고를 발견하고, 대안적 사고를 유도하세요. |
| 4 | Behavioral Activation | 행동 변화 계획 | 2 | 5 | 실천 가능한 구체적 행동 계획을 함께 세우세요. |
| 5 | Wrap-up | 요약, 마무리 | 1 | 3 | 오늘 상담을 요약하고, 행동 지침을 정리하세요. |

**전체 상담: 최소 11턴 ~ 최대 28턴**

---

## 초기 플랜 생성

사용자가 상담 시작 시 입력하는 정보:
- **상담 주제**: 직장 스트레스, 대인관계, 이별과 상실, 등 (STEP 1/4 화면)
- **초기 감정**: 첫 발화에서 멀티모달 감정 인식으로 파악

이 정보를 기반으로 **고정 템플릿**에 채워넣어 플랜 생성:

```python
CBT_PLAN_TEMPLATE = {
    "topic": "",        # 사용자 선택
    "initial_emotion": "",  # 첫 발화 감정 인식 결과
    "steps": [
        {
            "step": 1,
            "name": "Rapport Building",
            "min_turns": 2,
            "max_turns": 5,
            "current_turns": 0,
            "system_prompt": (
                "당신은 CBT 기반 심리 상담사입니다. "
                "내담자가 '{topic}' 주제로 상담을 시작했습니다. "
                "내담자의 현재 감정: {emotion}. "
                "먼저 내담자의 이야기를 충분히 경청하고 공감해주세요. "
                "판단하거나 조언하지 말고, 안전한 공간을 만들어주세요. "
                "내담자가 충분히 이야기했다고 판단되면 "
                "'다음 단계로 넘어가볼까요?'라고 자연스럽게 물어보세요."
            ),
            "status": "pending",
        },
        {
            "step": 2,
            "name": "Assessment",
            "min_turns": 3,
            "max_turns": 7,
            "current_turns": 0,
            "system_prompt": (
                "당신은 CBT 기반 심리 상담사입니다. "
                "내담자가 '{topic}' 주제로 상담 중입니다. "
                "내담자의 현재 감정: {emotion}. "
                "이제 내담자의 구체적인 상황, 감정, 생각을 탐색하세요. "
                "'그때 어떤 생각이 드셨어요?', '어떤 감정이 느껴지셨나요?' 등 "
                "구체적 질문으로 문제를 파악하세요. "
                "충분히 파악되었다고 판단되면 "
                "'다음 단계로 넘어가볼까요?'라고 자연스럽게 물어보세요."
            ),
            "status": "pending",
        },
        {
            "step": 3,
            "name": "Cognitive Restructuring",
            "min_turns": 3,
            "max_turns": 8,
            "current_turns": 0,
            "system_prompt": (
                "당신은 CBT 기반 심리 상담사입니다. "
                "내담자가 '{topic}' 주제로 상담 중입니다. "
                "내담자의 현재 감정: {emotion}. "
                "내담자의 자동적 사고(부정적 사고 패턴)를 발견하도록 도와주세요. "
                "'정말 그렇게 생각하세요?', '다른 가능성은 없을까요?' 등 "
                "소크라테스식 질문으로 인지 재구조화를 유도하세요. "
                "내담자가 새로운 관점을 발견했다고 판단되면 "
                "'다음 단계로 넘어가볼까요?'라고 자연스럽게 물어보세요."
            ),
            "status": "pending",
        },
        {
            "step": 4,
            "name": "Behavioral Activation",
            "min_turns": 2,
            "max_turns": 5,
            "current_turns": 0,
            "system_prompt": (
                "당신은 CBT 기반 심리 상담사입니다. "
                "내담자가 '{topic}' 주제로 상담 중입니다. "
                "내담자의 현재 감정: {emotion}. "
                "이제 내담자와 함께 실천 가능한 구체적 행동 계획을 세우세요. "
                "'이번 주에 해볼 수 있는 작은 것이 있을까요?' 등 "
                "작고 구체적인 행동 목표를 설정하세요. "
                "행동 계획이 충분히 세워졌다면 "
                "'마무리 단계로 넘어가볼까요?'라고 자연스럽게 물어보세요."
            ),
            "status": "pending",
        },
        {
            "step": 5,
            "name": "Wrap-up",
            "min_turns": 1,
            "max_turns": 3,
            "current_turns": 0,
            "system_prompt": (
                "당신은 CBT 기반 심리 상담사입니다. "
                "내담자가 '{topic}' 주제로 상담 중입니다. "
                "내담자의 현재 감정: {emotion}. "
                "오늘 상담 내용을 요약해주세요: "
                "1) 내담자의 주요 고민 "
                "2) 발견한 자동적 사고 패턴 "
                "3) 새로운 관점 "
                "4) 행동 계획 "
                "따뜻하게 마무리하고 격려해주세요."
            ),
            "status": "pending",
        },
    ],
}
```

---

## 스텝 전환 메커니즘

### 전환 방식: LLM 판단 + 사용자 확인

```
[현재 스텝 진행 중]
        │
        ├─ LLM이 충분하다고 판단
        │       │
        │       ├─ LLM: "다음 단계로 넘어가볼까요?"
        │       │
        │       ├─ 사용자: "네" → 다음 스텝으로 전환
        │       │
        │       └─ 사용자: "아직요" / 다른 이야기 → 현재 스텝 계속
        │
        └─ 사용자가 "다음" 버튼 클릭 → 다음 스텝으로 전환
```

### 구현 방식

LLM 응답에 **스텝 전환 제안 여부**를 포함시킴:

```python
class StepManager:
    def __init__(self, topic, initial_emotion):
        self.topic = topic
        self.current_step = 1
        self.plan = self._create_plan(topic, initial_emotion)
        self.transition_proposed = False  # 전환 제안 했는지

    def _create_plan(self, topic, emotion):
        """고정 템플릿에 주제/감정 채우기"""
        plan = copy.deepcopy(CBT_PLAN_TEMPLATE)
        plan['topic'] = topic
        plan['initial_emotion'] = emotion
        for step in plan['steps']:
            step['system_prompt'] = step['system_prompt'].format(
                topic=topic, emotion=emotion
            )
        return plan

    def get_current_step_info(self):
        """현재 스텝 정보 반환"""
        return self.plan['steps'][self.current_step - 1]

    def increment_turn(self):
        """현재 스텝의 턴 수 증가"""
        step = self.get_current_step_info()
        step['current_turns'] += 1

    def should_suggest_transition(self):
        """LLM에게 전환 제안을 유도할지 판단"""
        step = self.get_current_step_info()
        # 최소 턴 미달 → 전환 제안 금지
        if step['current_turns'] < step['min_turns']:
            return False
        # 최대 턴 도달 → 자동 전환 제안
        if step['current_turns'] >= step['max_turns']:
            return True
        # 최소~최대 사이 → LLM이 자유롭게 판단
        return None  # LLM 판단에 맡김

    def get_current_system_prompt(self, current_emotion):
        """현재 스텝의 시스템 프롬프트 (감정 + 턴 수 반영)"""
        step = self.get_current_step_info()
        # 감정을 실시간으로 업데이트
        prompt = step['system_prompt'].replace(
            f"내담자의 현재 감정: {self.plan['initial_emotion']}",
            f"내담자의 현재 감정: {current_emotion}"
        )

        # 턴 수에 따라 전환 지시 추가
        transition_hint = self.should_suggest_transition()
        if transition_hint is False:
            # 최소 턴 미달 → 전환 금지
            prompt += (
                f" 현재 이 단계에서 {step['current_turns']}/{step['min_turns']}턴 진행했습니다. "
                f"아직 최소 턴에 도달하지 않았으니, 전환을 제안하지 마세요."
            )
        elif transition_hint is True:
            # 최대 턴 도달 → 자동 전환 제안
            prompt += (
                f" 현재 이 단계에서 {step['current_turns']}/{step['max_turns']}턴 진행했습니다. "
                f"최대 턴에 도달했으니, 반드시 다음 단계로 넘어가자고 제안하세요."
            )
        # None이면 LLM 자유 판단 (프롬프트 추가 안 함)

        return prompt

    def check_user_response_for_transition(self, user_text):
        """사용자 응답에서 스텝 전환 동의 확인"""
        if not self.transition_proposed:
            return False

        positive_keywords = ['네', '좋아요', '넘어가요', '괜찮아요', '응', '그래요']
        negative_keywords = ['아직', '아니요', '더', '잠깐', '아니']

        for kw in positive_keywords:
            if kw in user_text:
                self.transition_proposed = False
                return True

        for kw in negative_keywords:
            if kw in user_text:
                self.transition_proposed = False
                return False

        return False

    def advance_step(self):
        """다음 스텝으로 전환"""
        if self.current_step < 5:
            self.plan['steps'][self.current_step - 1]['status'] = 'completed'
            self.current_step += 1
            self.plan['steps'][self.current_step - 1]['status'] = 'active'
            return True
        return False  # 이미 마지막 스텝

    def is_completed(self):
        """모든 스텝 완료 여부"""
        return self.current_step == 5 and \
               self.plan['steps'][4]['status'] == 'completed'

    def force_next_step(self):
        """사용자가 버튼으로 강제 전환"""
        return self.advance_step()
```

---

## LLM 응답 생성 흐름

```python
async def generate_response(self, user_text, current_emotion):
    """LLM 응답 생성 (스텝 + 턴 수 관리 포함)"""

    # 1. 스텝 전환 확인 (LLM이 이전에 제안했으면)
    if self.step_manager.check_user_response_for_transition(user_text):
        self.step_manager.advance_step()

    # 2. 턴 수 증가
    self.step_manager.increment_turn()

    # 3. 현재 스텝의 시스템 프롬프트 가져오기 (턴 수 반영)
    system_prompt = self.step_manager.get_current_system_prompt(current_emotion)

    # 4. 대화 히스토리 구성
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(self.chat_history)
    messages.append({"role": "user", "content": user_text})

    # 5. LLM 응답 생성
    response = self.llm.generate(messages)

    # 6. 스텝 전환 제안 감지
    transition_keywords = ['다음 단계로', '넘어가볼까요', '마무리 단계로']
    for kw in transition_keywords:
        if kw in response:
            self.step_manager.transition_proposed = True
            break

    # 7. 히스토리 업데이트
    self.chat_history.append({"role": "user", "content": user_text})
    self.chat_history.append({"role": "assistant", "content": response})

    step_info = self.step_manager.get_current_step_info()
    return {
        "response": response,
        "current_step": self.step_manager.current_step,
        "step_name": step_info['name'],
        "step_turns": f"{step_info['current_turns']}/{step_info['max_turns']}",
    }
```

---

## 전체 상담 흐름 예시

```
[사용자 입력]
  상담 주제: 직장 스트레스
  초기 감정: angry (멀티모달 감정 인식)

[Step 1: Rapport Building]
  상담사: "안녕하세요. 직장에서 힘든 일이 있으셨나 봐요.
          편하게 말씀해 주세요."
  사용자: "네, 상사가 계속 부당하게 대해서..."
  상담사: "정말 속상하셨겠어요. 더 이야기해 주실 수 있나요?"
  사용자: "매일 야근하는데 인정도 안 해주고..."
  상담사: "충분히 화가 나실 만한 상황이네요.
          이야기를 더 자세히 나눠볼까요?
          다음 단계로 넘어가볼까요?"
  사용자: "네, 좋아요." → Step 2로 전환

[Step 2: Assessment]
  상담사: "상사가 부당하게 대할 때, 어떤 생각이 드세요?"
  사용자: "나는 무능하다는 생각이 들어요..."
  ...

[Step 3: Cognitive Restructuring]
  상담사: "정말 무능하다고 생각하세요?
          다른 가능성은 없을까요?"
  ...

[Step 4: Behavioral Activation]
  상담사: "이번 주에 해볼 수 있는 작은 것이 있을까요?"
  ...

[Step 5: Wrap-up]
  상담사: "오늘 상담을 정리해볼게요.
          1) 상사와의 갈등으로 인한 스트레스
          2) '나는 무능하다'는 자동적 사고 발견
          3) '상사의 태도는 나의 능력과 별개'라는 관점
          4) 이번 주 퇴근 시간 지키기 실천
          오늘 정말 잘하셨어요."

[상담 완료 → 리포트 생성]
```

---

## 리포트 생성 (Step 5 완료 후)

```python
def generate_report(self):
    """상담 완료 후 리포트 생성"""
    return {
        "topic": self.step_manager.topic,
        "emotion_timeline": self.emotion_history,  # 턴별 감정 변화
        "summary": {
            "main_concern": "...",  # LLM이 생성
            "automatic_thoughts": "...",  # Step 3에서 발견한 것
            "new_perspectives": "...",  # Step 3에서 재구조화한 것
            "action_plan": "...",  # Step 4에서 세운 계획
        },
        "emotion_change": {
            "start": self.emotion_history[0]['emotion'],
            "end": self.emotion_history[-1]['emotion'],
        },
        "total_turns": len(self.chat_history) // 2,
    }
```
