## Counseling-AI

실시간 멀티모달 감정 인식과 LLM 응답 생성을 조합한 코어 서비스/학습 레포입니다.

- `ai_core/`: 실시간 감정 추정, LLM 호출, 톤 라우팅, 서비스 오케스트레이션
- `training/`: 데이터 정규화, SFT(QLoRA) 학습, LoRA 병합
- `contracts/`: 프론트/백엔드와의 API 계약 (OpenAPI / gRPC)
- `scripts/`: 데모 스크립트 및 vLLM 서버 실행 스크립트
- `docker/`: 배포용 Dockerfile
- `data/`: 데이터 파이프라인 입력/중간/출력 디렉터리

### 빠른 시작

#### **방법 1: Mock 모드 (GPU 없이 테스트)** 🎭 ⭐
```bash
# 최소 패키지 설치
pip install numpy pydantic opencv-python soundfile

# Mock 모드로 실행
export MOCK_MODE=true
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0 \
    --auto
```

#### **방법 2: 실제 AI 모델 (GPU 환경)**
1) 의존성 설치
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
2) 환경변수 템플릿 복사
```bash
cp .env.example .env
```
3) 데모 실행
```bash
python scripts/demo_full_counseling_workflow.py \
    --video test_video/sample.mp4 \
    --backend transformers \
    --gpu 0
```

### vLLM 서버 실행(옵션)
```bash
bash scripts/run_vllm.sh
```

### 디렉터리 개요
- `ai_core/configs/`: 스트리밍 파라미터, LLM 설정, 경로 설정
- `ai_core/emotion/`: 얼굴/오디오/텍스트 기반 감정 추정 및 평활화
- `ai_core/llm/`: 로컬 Transformers 혹은 OpenAI 호환(vLLM) 클라이언트
- `ai_core/router/`: 감정→응답 톤/전략 매핑
- `ai_core/io/`: Pydantic 스키마, Redis 상태 저장, Kafka 퍼블리시
- `ai_core/service/`: 한 턴 오케스트레이션


