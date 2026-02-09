import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_core.service.orchestrator import TurnOrchestrator
from ai_core.io.schemas import TurnInput


def main() -> None:
    orch = TurnOrchestrator()
    ti = TurnInput(utterance="요즘 스트레스를 많이 받는 것 같아요.")
    to = orch.run(ti)
    print("Response:", to.response)
    print("Emotion:", to.emotion.aggregated)
    print("Meta:", to.meta)


if __name__ == "__main__":
    main()


