from typing import Dict
import torch
from transformers import AutoConfig


class TextEmotionEstimator:
    """
    KLUE-BERT 기반 텍스트 감정 추정.
    Hugging Face pipeline을 사용하여 더 안정적으로 동작.
    """

    def __init__(
        self,
        model_name: str = "M1NJ1/klue-bert-emotion",
        use_pretrained: bool = True,
        gpu_id: int = 0
    ) -> None:
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.gpu_id = gpu_id
        self.classifier = None
        # Hugging Face pipeline: GPU는 번호, CPU는 -1
        self.device = gpu_id if torch.cuda.is_available() else -1
        self.model_ready = False
        
        # 표준 감정 라벨 (7개)
        self.standard_labels = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
        
        # LABEL 번호를 실제 감정으로 매핑 (KOTE 44개 감정 기준)
        # 실제 매핑은 모델 문서나 데이터셋에서 확인 필요
        # 현재는 패턴 기반 임시 매핑
        self.label_id_to_emotion = {}

    def warmup(self) -> None:
        """모델 로드 및 워밍업"""
        try:
            if self.use_pretrained and "klue-bert-emotion" in self.model_name:
                # Config 로드 (조용히)
                try:
                    config = AutoConfig.from_pretrained(self.model_name)
                except:
                    pass
                
                # Hugging Face pipeline 사용
                from transformers import pipeline
                
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=self.device
                )
                
                # 조용히 로드
                pass
            else:
                # KLUE-BERT-base를 직접 사용하는 경우 (폴백)
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "klue/bert-base",
                    num_labels=7
                )
                self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                self.model.eval()
                print(f"KLUE-BERT-base 모델 로드 완료 (7개 감정 분류)")
            
            self.model_ready = True
        except Exception as e:
            print(f"Warning: KLUE-BERT 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            print("더미 모델 사용")
            self.classifier = None
            self.model = None
            self.model_ready = True

    def infer(self, text: str) -> Dict[str, float]:
        """
        텍스트에서 감정 추정.
        """
        if not self.model_ready:
            self.warmup()
        
        if not text or not text.strip():
            return {label: 0.0 if label != "neutral" else 1.0 for label in self.standard_labels}
        
        # Pipeline 사용
        if self.classifier is not None:
            try:
                # 모든 감정 점수 가져오기
                results = self.classifier(text, top_k=None)
                
                # 결과 처리
                if isinstance(results, list) and len(results) > 0:
                    emotion_dict = {}
                    
                    # results는 딕셔너리 리스트
                    for item in results:
                        if isinstance(item, dict):
                            label = item.get('label', '')
                            score = item.get('score', 0.0)
                            
                            # LABEL_24 형식을 실제 감정으로 매핑
                            standard_label = self._map_label_id_to_emotion(label)
                            
                            if standard_label in emotion_dict:
                                emotion_dict[standard_label] += float(score)
                            else:
                                emotion_dict[standard_label] = float(score)
                    
                    # 표준 라벨로 정규화
                    result = {label: 0.0 for label in self.standard_labels}
                    for label, prob in emotion_dict.items():
                        if label in result:
                            result[label] += prob
                    
                    # 정규화
                    total = sum(result.values())
                    if total > 0:
                        for key in result:
                            result[key] /= total
                    else:
                        result["neutral"] = 1.0
                    
                    return result
                
            except Exception as e:
                print(f"Pipeline 추론 에러: {e}")
                import traceback
                traceback.print_exc()
                return self._dummy_infer(text)
        
        # 직접 모델 사용 (폴백)
        if hasattr(self, 'model') and self.model is not None:
            try:
                from transformers import AutoTokenizer
                if not hasattr(self, 'tokenizer') or self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
                
                result = {label: 0.0 for label in self.standard_labels}
                for i, prob in enumerate(probs):
                    if i < len(self.standard_labels):
                        result[self.standard_labels[i]] = float(prob)
                
                return result
            except Exception as e:
                print(f"직접 모델 추론 에러: {e}")
                return self._dummy_infer(text)
        
        # 더미 모델
        return self._dummy_infer(text)

    def _map_label_id_to_emotion(self, label: str) -> str:
        """
        LABEL_24 같은 형식을 실제 감정으로 매핑.
        KOTE 데이터셋의 44개 감정을 7개 표준 감정으로 매핑.
        패턴 분석 기반 매핑 (2025-01-11 업데이트)
        """
        if not label or not label.startswith("LABEL_"):
            return "neutral"
        
        try:
            label_id = int(label.replace("LABEL_", ""))
        except:
            return "neutral"
        
        # 패턴 분석 결과 기반 매핑
        emotion_groups = {
            "happy": [13, 28, 40, 42],
            "sad": [5, 10, 19, 27, 36, 38, 41],
            "angry": [0, 6, 22, 31],
            "fear": [2, 5, 12, 16, 18, 21, 33, 35, 41],
            "surprise": [2, 6, 15, 16, 33, 34, 39],
            "disgust": [0, 21, 22, 31],
            "neutral": [10, 14, 24, 26, 27, 37, 40, 42],
        }
        
        # 중복이 있는 경우 우선순위: sad > fear > angry > surprise > disgust > happy > neutral
        # (더 구체적인 감정을 우선)
        priority_order = ["sad", "fear", "angry", "surprise", "disgust", "happy", "neutral"]
        
        for emotion in priority_order:
            if label_id in emotion_groups[emotion]:
                return emotion
        
        # 기본값
        return "neutral"

    def _map_label_to_standard(self, label: str) -> str:
        """
        모델의 라벨을 표준 감정 라벨로 매핑 (텍스트 기반).
        """
        if not label:
            return "neutral"
        
        # LABEL_ 형식이면 _map_label_id_to_emotion 사용
        if label.startswith("LABEL_"):
            return self._map_label_id_to_emotion(label)
        
        label_lower = label.lower()
        
        # Happy 계열
        if any(word in label_lower for word in ["기쁨", "행복", "즐거움", "만족", "신남", "설렘", "뿌듯", 
                                                 "happy", "joy", "pleasure", "satisfaction", "excitement"]):
            return "happy"
        
        # Sad 계열
        if any(word in label_lower for word in ["슬픔", "우울", "절망", "외로움", "실망", "아픔", "그리움",
                                                 "sad", "sorrow", "depression", "loneliness", "disappointment"]):
            return "sad"
        
        # Angry 계열
        if any(word in label_lower for word in ["분노", "화남", "짜증", "불만", "angry", "anger", "rage", "frustration"]):
            return "angry"
        
        # Fear 계열
        if any(word in label_lower for word in ["두려움", "불안", "걱정", "공포", "fear", "anxiety", "worry", "panic"]):
            return "fear"
        
        # Surprise 계열
        if any(word in label_lower for word in ["놀람", "당황", "충격", "surprise", "shock", "amazement"]):
            return "surprise"
        
        # Disgust 계열
        if any(word in label_lower for word in ["혐오", "역겨움", "disgust", "revulsion"]):
            return "disgust"
        
        # Neutral 계열
        if any(word in label_lower for word in ["중립", "평온", "무감정", "neutral", "calm"]):
            return "neutral"
        
        # 기본값
        return "neutral"

    def _dummy_infer(self, text: str) -> Dict[str, float]:
        """더미 감정 추정 (모델 로드 실패 시)"""
        text_lower = (text or "").lower()
        if "고마워" in text_lower or "좋아" in text_lower or "행복" in text_lower:
            return {"happy": 0.7, "neutral": 0.3, "sad": 0.0, "angry": 0.0, 
                   "fear": 0.0, "surprise": 0.0, "disgust": 0.0}
        if "슬퍼" in text_lower or "우울" in text_lower or "힘들" in text_lower:
            return {"sad": 0.7, "neutral": 0.3, "happy": 0.0, "angry": 0.0, 
                   "fear": 0.0, "surprise": 0.0, "disgust": 0.0}
        if "화나" in text_lower or "짜증" in text_lower or "분노" in text_lower:
            return {"angry": 0.7, "neutral": 0.3, "happy": 0.0, "sad": 0.0, 
                   "fear": 0.0, "surprise": 0.0, "disgust": 0.0}
        return {"neutral": 0.9, "happy": 0.05, "sad": 0.03, "angry": 0.02, 
               "fear": 0.0, "surprise": 0.0, "disgust": 0.0}