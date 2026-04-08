# 프로젝트 구조 및 파일별 기능 정리

## 프로젝트명: AI를 통한 상담 - 비언어적 특성을 반영한 CBT 기반 상담 시스템

> 마지막 업데이트: 2026-03-08

---

## 폴더 구조 요약

```
capstone-ai-counselin/
├── configs/                    # 설정 파일 (미사용)
├── counseling_server/          # 백엔드 서버 (팀원 담당, Git 클론)
│   ├── ai_modules/             # AI 모듈 인터페이스 + 실제 구현
│   └── app/                    # FastAPI 앱 + WebSocket
├── data/                       # 데이터셋
│   ├── processed/              # 전처리된 학습 데이터
│   └── emotion_labels/         # 감정 라벨 (AI Hub)
├── models/                     # 학습된 모델 저장소
│   ├── base/                   # 베이스 모델 (CBT, 텍스트감정, 음성감정)
│   └── lora/                   # 감정별 LoRA 어댑터 (7개)
├── notebooks/                  # 분석/테스트 스크립트
├── outputs/                    # 출력 결과물
├── pipeline_design/            # 파이프라인 설계 문서
├── results/                    # 평가 결과/로그
├── src/                        # AI 파트 소스 코드
│   ├── emotion/                # 감정 인식 모듈
│   ├── pipeline/               # 파이프라인 핵심 모듈
│   ├── train/                  # 학습 스크립트
│   └── utils/                  # 유틸리티 (비어있음)
└── folder_structure.md         # 이 문서
```

---

## data/ — 데이터셋

| 경로 | 설명 |
|------|------|
| `data/processed/cactus_qwen_format.json` | CACTUS 데이터셋을 Qwen chat format으로 전처리한 전체 데이터 (31,577개) |
| `data/processed/cactus_train.json` | CACTUS 학습용 (28,418개) |
| `data/processed/cactus_val.json` | CACTUS 검증용 (3,159개) |
| `data/processed/cactus_korean/cactus_korean.json` | CACTUS 한국어 번역본 (🔄 1,870/5,000 진행 중) |
| `data/processed/emotion_lora/all_emotions.json` | 감정 7개 통합 데이터 (2,800개 = 7감정 × 400개) |
| `data/processed/emotion_lora/{감정}.json` | 감정별 개별 데이터 (happy, sad, angry, surprise, fear, disgust, neutral) |
| `data/emotion_labels/` | 감정 라벨 관련 (AI Hub 5차년도_2차.csv 등) |

---

## models/ — 학습된 모델

### models/base/ — 베이스 모델

| 경로 | 모델 | 정확도 | 설명 |
|------|------|--------|------|
| `models/base/cbt-counselor/` | Qwen 2.5 3B + QLoRA | loss 0.518 | CACTUS로 학습한 CBT 상담 LoRA 어댑터 |
| `models/base/text-emotion/` | klue/bert-base 파인튜닝 | 93% | AI Hub 데이터로 학습한 텍스트 감정 분류기 |
| `models/base/voice-emotion/` | wav2vec2-large-xlsr-53 파인튜닝 | 86% | AI Hub 데이터로 학습한 음성 감정 분류기 |

### models/lora/ — 감정별 LoRA (400개 데이터 기준)

| 경로 | 감정 | 상태 |
|------|------|------|
| `models/lora/happy/` | 기쁨/행복 | ✅ 학습 완료 |
| `models/lora/sad/` | 슬픔/우울 | ✅ 학습 완료 |
| `models/lora/angry/` | 분노/화남 | ✅ 학습 완료 |
| `models/lora/surprise/` | 놀람 | ✅ 학습 완료 |
| `models/lora/fear/` | 두려움/불안 | ✅ 학습 완료 |
| `models/lora/disgust/` | 혐오/불쾌 | ✅ 학습 완료 |
| `models/lora/neutral/` | 평온/무감정 | ✅ 학습 완료 |

※ 7개 전부 Google Drive에 저장. 로컬에 happy, sad, angry, neutral만 다운로드된 상태. surprise, fear, disgust 로컬 다운로드 필요.

---

## src/emotion/ — 감정 인식 모듈

| 파일 | 기능 | 사용 모델 |
|------|------|----------|
| `text.py` | 텍스트 감정 인식 테스트 | klue/bert-base (93%) |
| `voice.py` | 음성 감정 인식 테스트 | wav2vec2-large-xlsr-53 (86%) |
| `face.py` | 얼굴 감정 인식 (웹캠 실시간) | DeepFace (84%) |
| `fusion.py` | 멀티모달 감정 융합 | 가중치 기반 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25) |
| `accuracy.md` | 감정 인식 모델별 정확도 기록 | - |

---

## src/pipeline/ — 파이프라인 핵심 모듈

| 파일 | 기능 | 상태 |
|------|------|------|
| `lora_switcher.py` | 감정별 LoRA 동적 스위칭. Multi-adapter 방식으로 7개 LoRA를 미리 로드하고 `set_adapter()`로 즉시 전환 (~0ms). CBT 베이스 merge 후 감정별 LoRA 로드, LLM 응답 생성 | ✅ 구현 + GPU 테스트 완료 (VRAM 2.03GB) |
| `step_manager.py` | CBT 5-Step 플래닝 관리. 플랜 생성, 턴 수 관리 (최소/최대), 스텝 전환 판단 (LLM 키워드 감지 + 사용자 확인), 시스템 프롬프트 생성 | ✅ 구현 완료 |
| `counseling_session.py` | 전체 상담 세션 통합 관리. StepManager + LoRASwitcher + EmotionFusion + 감정 버퍼 통합. 세션 시작, 발화 완료 처리(`on_user_done`), 스텝 전환, 강제 전환, 상태 관리 | ✅ 구현 완료 (통합 테스트 필요) |
| `report_generator.py` | 상담 완료 후 리포트 생성. 감정 변화 타임라인, 감정 valence 분석, 스텝별 요약, LLM 자연어 요약, 마크다운 변환 | ✅ 구현 완료 (LLM 연결 후 품질 테스트 필요) |
| `chat_sim.py` | 멀티턴 상담 시뮬레이션 (초기 테스트용, 영어 CACTUS 기반) | ✅ |
| `LoRA apapt pipeline.md` | LoRA 적용 파이프라인 설계 메모 | 📄 |

---

## src/train/ — 학습 스크립트

| 파일 | 기능 | 실행 환경 |
|------|------|----------|
| `train_base.py` | CACTUS 데이터셋으로 Qwen 2.5 3B QLoRA 학습 | 코랩 (A100/L4) |
| `train_text_emotion.py` | klue/bert 텍스트 감정 분류기 학습 | 로컬 (3060 Ti) |
| `generate_emotion_data.py` | GPT-4o-mini로 감정별 상담 데이터 합성 (7감정 × 400개 = 2,800개 완료) | 로컬 (API) |
| `translate_cactus.py` | CACTUS 데이터셋 한국어 번역 (GPT-4o-mini, 🔄 1,870/5,000 진행 중) | 로컬 (API) |

※ 코랩 노트북: `train_base_colab_v2.ipynb`, `train_voice_wav2vec2_colab.ipynb`, `train_emotion_lora_colab.ipynb`는 Google Drive에 저장

---

## counseling_server/ — 백엔드 서버 (팀원 담당, Git 클론)

> ⚠️ 이 폴더는 백엔드 팀원이 관리하는 별도 Git 리포지토리를 로컬에 클론한 것

### counseling_server/ai_modules/ — AI 모듈 인터페이스

| 파일 | 기능 |
|------|------|
| `interfaces.py` | AI 모듈 추상 인터페이스 (BaseVADModel, BaseSTTModel, BaseEmotionModel, BaseLLMModel) + Dummy 구현체 |
| `models.py` | 실제 AI 모델 구현체 (SileroVAD, WhisperSTT, Wav2VecEmotion, TextEmotion, DeepFaceEmotion, CBTLLM, EmotionFusion) |
| `schemas.py` | Pydantic 데이터 스키마 (VADInput/Output, STTInput/Output, EmotionResult, LLMContext/Response, FaceInput) |
| `AI_MODULE_GUIDE.md` | AI 모듈 가이드 문서 |

### counseling_server/app/ — FastAPI 앱

| 파일 | 기능 |
|------|------|
| `main.py` | FastAPI 앱 생성, WebSocket 엔드포인트 (`/ws/counseling/{client_id}`) |
| `schemas.py` | WebSocket 입출력 스키마 (InputTest, ServerResponse) |
| `core/container.py` | AI 모델 싱글톤 컨테이너 (현재 Dummy 모델로 연결) |
| `services/session_manager.py` | WebSocket 연결 관리, 오디오/비디오/컨트롤 데이터 분기 처리 |

※ `models.py`의 `CBTLLMModel`이 아직 기존 메서드명(`switch_lora`, `load_base_model`)을 사용 중 → 백엔드 팀원에게 `switch()`, `load_all()`로 변경 요청 필요

---

## pipeline_design/ — 파이프라인 설계 문서

| 파일 | 내용 |
|------|------|
| `감정 인식 파이프라인.md` | 실시간 감정 인식 파이프라인 설계 (v2). 웹캠/음성/텍스트 처리 흐름, 병렬 처리, 기술 스택 |
| `CBT 5STEP 파이프라인.md` | CBT 5-Step 플래닝 설계. 스텝 구조, 턴 수 관리, 전환 메커니즘, 리포트 생성 |
| `전체 파이프라인.md` | 초기 입력 → 턴별 감정 추출 → LoRA 적용 → 리포트까지 전체 흐름도. 처리 시간 분석 (~4초) |

---

## notebooks/ — 분석/테스트

| 파일 | 기능 |
|------|------|
| `01_data_analysis.py` | CACTUS 데이터셋 분석 (대화 수, 턴 수 분포 등) |
| `02_model_test.py` | 모델 테스트 스크립트 |

---

## 감정 카테고리 (7개 통일)

happy, sad, angry, surprise, fear, disgust, neutral

---

## 기술 스택

| 구성요소 | 기술 |
|---------|------|
| LLM | Qwen 2.5 3B-Instruct + QLoRA |
| LoRA 스위칭 | PEFT multi-adapter (7개 미리 로드, set_adapter로 즉시 전환) |
| 텍스트 감정 | klue/bert-base 파인튜닝 (93%) |
| 음성 감정 | wav2vec2-large-xlsr-53 파인튜닝 (86%) |
| 얼굴 감정 | DeepFace (84%) |
| STT | Whisper small (로컬) |
| 감정 융합 | 가중치 기반 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25) |
| LLM 서빙 | vLLM (예정) |
| 백엔드 | FastAPI + WebSocket (팀원 담당) |
| 프론트엔드 | HTML/JS + MediaStream API (팀원 담당) |

---

## 현재 진행 상태 요약

| 항목 | 상태 |
|------|------|
| CBT LLM 학습 (영어) | ✅ 완료 |
| 텍스트 감정 인식 (93%) | ✅ 완료 |
| 음성 감정 인식 (86%) | ✅ 완료 |
| 얼굴 감정 인식 (84%) | ✅ 완료 |
| 멀티모달 감정 융합 | ✅ 완료 |
| 감정별 LoRA 학습 (400개) | ✅ 완료 |
| LoRA 스위칭 메커니즘 (multi-adapter) | ✅ 완료 + GPU 테스트 통과 |
| 파이프라인 설계 문서 | ✅ 완료 |
| 파이프라인 핵심 모듈 코드 | ✅ 완료 (통합 테스트 필요) |
| 리포트 생성 모듈 | ✅ 완료 (LLM 연결 후 품질 테스트 필요) |
| CACTUS 한국어 번역 | 🔄 1,870/5,000 진행 중 |
| 번역 데이터 후처리 (이름 제거 등) | ⬜ 번역 완료 후 진행 |
| LLM 한국어 재학습 | ⬜ 미착수 |
| 대화 히스토리 요약 관리 | ⬜ 미착수 |
| 실시간 감정 인식 파이프라인 구현 | ⬜ 미착수 |
| 엔드투엔드 통합 테스트 | ⬜ 미착수 |