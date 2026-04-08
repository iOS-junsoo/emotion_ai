"""
DynamicPlanGenerator - 초기 입력 기반 동적 5-Step 플랜 생성
============================================================
GPT-4o-mini API로 맞춤형 5-Step 플랜 생성 (세션당 1회 호출)
상담 대화 자체는 로컬 모델(Qwen + LoRA)이 담당

우선순위:
1. GPT-4o-mini API (고품질 JSON 생성)
2. 로컬 LLM 폴백 (API 키 없을 때)
3. 고정 템플릿 폴백 (LLM도 없을 때)

사용 흐름:
    # API 키로 초기화 (권장)
    generator = DynamicPlanGenerator(api_key="sk-...")
    plan = generator.generate(topic, emotion, detail)

    # 또는 로컬 LLM으로 초기화
    generator = DynamicPlanGenerator(lora_switcher=switcher)
    plan = generator.generate(topic, emotion, detail)

    # StepManager에서 사용:
    prompt = build_dynamic_system_prompt(plan['steps'][0], topic, emotion)

위치: src/pipeline/dynamic_plan.py
"""

import json
import re
import os


# ============================================================
# 플랜 생성 프롬프트
# ============================================================

PLAN_GENERATION_PROMPT = """당신은 CBT(인지행동치료) 전문 상담사입니다.
아래 내담자 정보를 바탕으로 맞춤형 5단계 상담 계획을 JSON으로 생성하세요.

[내담자 정보]
상담 주제: {topic}
현재 감정: {emotion}
상세 고민: {detail}

[지시사항]
1. 내담자의 핵심 문제와 예상되는 인지 왜곡 패턴을 분석하세요 (과잉일반화, 흑백사고, 재난화, 개인화, 독심술, 감정적 추론 등)
2. 각 단계별로 이 내담자의 구체적 상황에 맞는 목표와 질문을 설계하세요
3. 질문은 한국어 상담사가 실제로 쓰는 자연스러운 구어체로 작성하세요
4. 반드시 한국어로만 작성하세요

[출력 형식 - 반드시 아래 JSON만 출력. 다른 텍스트 금지]
{{
    "analysis": {{
        "core_problem": "핵심 문제 한 줄 요약",
        "cognitive_pattern": "예상되는 인지 왜곡 패턴명과 설명"
    }},
    "steps": [
        {{
            "step": 1,
            "name": "공감 형성",
            "goal": "이 내담자의 상황에 맞는 구체적 목표",
            "key_questions": [
                "이 내담자에게 물어볼 구체적인 질문 1",
                "구체적인 질문 2"
            ],
            "focus": "이 스텝에서 상담사가 집중할 포인트"
        }},
        {{
            "step": 2,
            "name": "문제 탐색",
            "goal": "구체적 목표",
            "key_questions": ["질문 1", "질문 2", "질문 3"],
            "focus": "집중 포인트"
        }},
        {{
            "step": 3,
            "name": "사고 전환",
            "goal": "구체적 목표",
            "key_questions": ["소크라테스식 질문 1", "질문 2", "질문 3"],
            "focus": "집중 포인트"
        }},
        {{
            "step": 4,
            "name": "행동 계획",
            "goal": "구체적 목표",
            "key_questions": ["행동 관련 질문 1", "질문 2"],
            "focus": "집중 포인트"
        }},
        {{
            "step": 5,
            "name": "마무리",
            "goal": "구체적 목표",
            "key_questions": ["마무리 질문 1"],
            "focus": "집중 포인트"
        }}
    ]
}}"""


# ============================================================
# 동적 시스템 프롬프트 템플릿
# ============================================================

DYNAMIC_SYSTEM_PROMPT_TEMPLATE = (
    "당신은 한국어로만 대화하는 CBT 기반 심리 상담사입니다. "
    "절대로 중국어, 영어, 일본어를 사용하지 마세요. 한국어만 사용하세요. "
    "내담자가 '{topic}' 주제로 상담 중입니다. "
    "내담자의 현재 감정: {emotion}. "
    "\n\n[현재 단계] {step_name}"
    "\n[목표] {goal}"
    "\n[집중 포인트] {focus}"
    "\n[핵심 질문 가이드]\n{questions_text}"
    "\n\n위 핵심 질문을 참고하되, 대화 흐름에 맞게 자연스럽게 질문하세요. "
    "기계적으로 질문 목록을 읽지 마세요. "
    "짧고 자연스럽게 말하세요. 3~4문장을 넘기지 마세요. "
    "{transition_hint}"
)


# ============================================================
# DynamicPlanGenerator
# ============================================================

class DynamicPlanGenerator:
    """초기 입력(주제/감정/상세) → 맞춤형 5-Step 플랜 생성"""

    TURN_CONFIG = {
        1: (2, 5),   # 공감 형성
        2: (3, 7),   # 문제 탐색
        3: (3, 8),   # 사고 전환
        4: (2, 5),   # 행동 계획
        5: (1, 3),   # 마무리
    }

    def __init__(self, api_key=None, lora_switcher=None):
        """
        Args:
            api_key: OpenAI API 키 (GPT-4o-mini용, 권장)
            lora_switcher: 로컬 LLM (API 없을 때 폴백)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.lora_switcher = lora_switcher
        self.client = None

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                print("[경고] openai 패키지 미설치. pip install openai")
                self.client = None

    def generate(self, topic, emotion, detail):
        """
        주제/감정/상세 고민 → 5-Step 플랜 생성

        우선순위: GPT-4o-mini API → 로컬 LLM → 고정 폴백
        """
        prompt = PLAN_GENERATION_PROMPT.format(
            topic=topic, emotion=emotion, detail=detail
        )

        # 1순위: GPT-4o-mini API
        if self.client:
            print("[PlanGenerator] GPT-4o-mini로 플랜 생성 중...")
            raw_output = self._call_api(prompt)
            if raw_output:
                plan = self._parse_plan(raw_output, topic, emotion, detail)
                if plan.get("is_dynamic"):
                    return plan

        # 2순위: 로컬 LLM
        if self.lora_switcher:
            print("[PlanGenerator] 로컬 LLM으로 플랜 생성 중...")
            messages = [{"role": "user", "content": prompt}]
            raw_output = self.lora_switcher.generate(
                messages, max_new_tokens=800, temperature=0.3
            )
            plan = self._parse_plan(raw_output, topic, emotion, detail)
            if plan.get("is_dynamic"):
                return plan

        # 3순위: 고정 폴백
        print("[PlanGenerator] 폴백 플랜 사용")
        return self._fallback_plan(topic, emotion, detail)

    def _call_api(self, prompt):
        """GPT-4o-mini API 호출"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 CBT 전문 상담사입니다. "
                            "반드시 요청된 JSON 형식으로만 응답하세요. "
                            "한국어로만 작성하세요."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[경고] API 호출 실패: {e}")
            return None

    def _parse_plan(self, raw_output, topic, emotion, detail):
        """LLM 출력 → 플랜 딕셔너리"""
        try:
            cleaned = raw_output.strip()
            cleaned = re.sub(r"```json\s*", "", cleaned)
            cleaned = re.sub(r"```\s*$", "", cleaned)

            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                cleaned = cleaned[json_start:json_end]

            plan_data = json.loads(cleaned)

            # 검증
            if "steps" not in plan_data or len(plan_data["steps"]) != 5:
                print("[경고] 플랜 구조 불일치.")
                return {"is_dynamic": False}

            # 메타 정보
            plan_data["topic"] = topic
            plan_data["initial_emotion"] = emotion
            plan_data["detail"] = detail
            plan_data["is_dynamic"] = True

            # 각 스텝에 턴 수/상태 기본값
            for step in plan_data["steps"]:
                step_num = step["step"]
                min_t, max_t = self.TURN_CONFIG.get(step_num, (2, 5))
                step.setdefault("min_turns", min_t)
                step.setdefault("max_turns", max_t)
                step.setdefault("current_turns", 0)
                step.setdefault("status", "pending")
                step.setdefault("key_questions", [])
                step.setdefault("goal", "")
                step.setdefault("focus", "")

            plan_data["steps"][0]["status"] = "active"
            return plan_data

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"[경고] 플랜 파싱 실패: {e}")
            return {"is_dynamic": False}

    def _fallback_plan(self, topic, emotion, detail):
        """고정 폴백 플랜"""
        return {
            "topic": topic,
            "initial_emotion": emotion,
            "detail": detail,
            "is_dynamic": False,
            "analysis": {
                "core_problem": f"'{topic}' 관련 어려움. 상세: {detail[:50]}",
                "cognitive_pattern": "상담 진행 중 파악 예정",
            },
            "steps": [
                {
                    "step": 1, "name": "공감 형성",
                    "goal": f"'{topic}'에 대한 내담자의 고충을 충분히 경청",
                    "key_questions": [
                        f"이 상황에 대해 좀 더 이야기해 주실 수 있나요?",
                        "그때 어떤 기분이 드셨어요?",
                    ],
                    "focus": "판단 없이 경청, 공감 표현",
                    "min_turns": 2, "max_turns": 5,
                    "current_turns": 0, "status": "active",
                },
                {
                    "step": 2, "name": "문제 탐색",
                    "goal": "구체적인 상황, 감정, 생각의 연결고리 파악",
                    "key_questions": [
                        "구체적으로 어떤 상황이었나요?",
                        "그때 어떤 생각이 드셨어요?",
                        "그 상황이 일상에 어떤 영향을 미치고 있나요?",
                    ],
                    "focus": "상황-감정-생각 연결고리",
                    "min_turns": 3, "max_turns": 7,
                    "current_turns": 0, "status": "pending",
                },
                {
                    "step": 3, "name": "사고 전환",
                    "goal": "자동적 사고 패턴 발견 및 인지 재구조화",
                    "key_questions": [
                        "정말 그렇게 생각하세요?",
                        "다른 가능성은 없을까요?",
                        "친한 친구가 같은 상황이라면 뭐라고 말해주겠어요?",
                    ],
                    "focus": "소크라테스식 질문으로 인지 왜곡 도전",
                    "min_turns": 3, "max_turns": 8,
                    "current_turns": 0, "status": "pending",
                },
                {
                    "step": 4, "name": "행동 계획",
                    "goal": "작고 구체적인 실천 계획 수립",
                    "key_questions": [
                        "이번 주에 해볼 수 있는 작은 것이 있을까요?",
                        "그걸 실천하려면 어떤 준비가 필요할까요?",
                    ],
                    "focus": "실현 가능한 구체적 행동 목표",
                    "min_turns": 2, "max_turns": 5,
                    "current_turns": 0, "status": "pending",
                },
                {
                    "step": 5, "name": "마무리",
                    "goal": "상담 내용 요약 및 격려",
                    "key_questions": [
                        "오늘 이야기하면서 어떤 점이 도움이 되셨나요?",
                    ],
                    "focus": "핵심 요약, 변화 인정, 격려",
                    "min_turns": 1, "max_turns": 3,
                    "current_turns": 0, "status": "pending",
                },
            ],
        }


# ============================================================
# 시스템 프롬프트 빌더
# ============================================================

def build_dynamic_system_prompt(step_data, topic, current_emotion):
    """동적 플랜의 스텝 데이터 → 시스템 프롬프트"""
    questions = step_data.get("key_questions", [])
    questions_text = "\n".join(f"- {q}" for q in questions) if questions else "- 내담자의 이야기를 경청하세요"

    current = step_data.get("current_turns", 0)
    min_t = step_data.get("min_turns", 2)
    max_t = step_data.get("max_turns", 5)

    if current < min_t:
        transition_hint = (
            f"현재 {current}/{min_t}턴 진행했습니다. "
            "아직 최소 턴에 도달하지 않았으니, 전환을 제안하지 마세요."
        )
    elif current >= max_t:
        transition_hint = (
            f"현재 {current}/{max_t}턴 진행했습니다. "
            "최대 턴에 도달했으니, 반드시 다음 단계로 넘어가자고 제안하세요."
        )
    else:
        transition_hint = (
            "내담자가 충분히 이야기했다고 판단되면 "
            "'다음 단계로 넘어가볼까요?'라고 자연스럽게 물어보세요."
        )

    return DYNAMIC_SYSTEM_PROMPT_TEMPLATE.format(
        topic=topic,
        emotion=current_emotion,
        step_name=step_data.get("name", ""),
        goal=step_data.get("goal", ""),
        focus=step_data.get("focus", ""),
        questions_text=questions_text,
        transition_hint=transition_hint,
    )


# ============================================================
# 테스트
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DynamicPlanGenerator 테스트")
    print("=" * 60)

    # API 키가 있으면 GPT-4o-mini, 없으면 폴백
    api_key = os.environ.get("OPENAI_API_KEY")

    generator = DynamicPlanGenerator(api_key=api_key)

    plan = generator.generate(
        topic="직장 스트레스",
        emotion="angry",
        detail="상사가 제 성과를 자기 것처럼 가져가고, 매일 야근하는데 인정도 안 해줘요."
    )

    print(f"\n주제: {plan['topic']}")
    print(f"감정: {plan['initial_emotion']}")
    print(f"동적 플랜: {plan['is_dynamic']}")
    print(f"\n분석:")
    print(f"  핵심 문제: {plan['analysis']['core_problem']}")
    print(f"  인지 패턴: {plan['analysis']['cognitive_pattern']}")

    print(f"\n5-Step 플랜:")
    for step in plan['steps']:
        status = "●" if step['status'] == 'active' else "○"
        print(f"\n  {status} Step {step['step']}: {step['name']}")
        print(f"    목표: {step['goal']}")
        print(f"    집중: {step['focus']}")
        print(f"    핵심 질문:")
        for q in step['key_questions']:
            print(f"      - {q}")

    # 시스템 프롬프트 테스트
    print(f"\n{'=' * 60}")
    print("시스템 프롬프트 (Step 1)")
    print(f"{'=' * 60}")
    prompt = build_dynamic_system_prompt(plan['steps'][0], plan['topic'], "angry")
    print(prompt)