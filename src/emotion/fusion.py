import numpy as np

class EmotionFusion:
    """멀티모달 감정 융합 모듈"""

    def __init__(self):
        self.emotion_names = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']

        # 가중치 (텍스트 0.40 + 음성 0.35 + 얼굴 0.25)
        self.weights = {
            'text': 0.40,
            'voice': 0.35,
            'face': 0.25,
        }

        # DeepFace 라벨 → 통일 라벨 매핑
        self.face_label_map = {
            'happy': 'happy', 'sad': 'sad', 'angry': 'angry',
            'surprise': 'surprise', 'fear': 'fear', 'disgust': 'disgust',
            'neutral': 'neutral'
        }

    def _normalize_probs(self, probs):
        """확률 합이 1이 되도록 정규화"""
        total = sum(probs.values())
        if total == 0:
            return {e: 1/7 for e in self.emotion_names}
        return {e: p / total for e, p in probs.items()}

    def _text_to_probs(self, text_output):
        """텍스트 모델 출력 → 7개 감정 확률"""
        probs = {}
        for i, emo in enumerate(self.emotion_names):
            probs[emo] = text_output[i]
        return probs

    def _voice_to_probs(self, voice_output):
        """음성 모델 출력 → 7개 감정 확률"""
        probs = {}
        for i, emo in enumerate(self.emotion_names):
            probs[emo] = voice_output[i]
        return probs

    def _face_to_probs(self, face_output):
        """DeepFace 출력 → 7개 감정 확률"""
        probs = {}
        for emo in self.emotion_names:
            probs[emo] = face_output.get(emo, 0.0) / 100.0  # DeepFace는 0~100 스케일
        return self._normalize_probs(probs)

    def fuse(self, text_probs=None, voice_probs=None, face_probs=None):
        """
        멀티모달 감정 융합

        Args:
            text_probs: 텍스트 모델 softmax 출력 (list, 7개)
            voice_probs: 음성 모델 softmax 출력 (list, 7개)
            face_probs: DeepFace emotion dict (0~100 스케일)

        Returns:
            dict: {
                'emotion': 최종 감정,
                'confidence': 최종 확신도,
                'scores': 7개 감정별 점수,
                'modality_emotions': 각 모달리티별 예측 감정,
                'conflict': 충돌 여부
            }
        """
        fused = {emo: 0.0 for emo in self.emotion_names}
        active_weight_sum = 0.0
        modality_emotions = {}

        # 텍스트
        if text_probs is not None:
            t_probs = self._text_to_probs(text_probs)
            t_emotion = max(t_probs, key=t_probs.get)
            modality_emotions['text'] = t_emotion
            for emo in self.emotion_names:
                fused[emo] += self.weights['text'] * t_probs[emo]
            active_weight_sum += self.weights['text']

        # 음성
        if voice_probs is not None:
            v_probs = self._voice_to_probs(voice_probs)
            v_emotion = max(v_probs, key=v_probs.get)
            modality_emotions['voice'] = v_emotion
            for emo in self.emotion_names:
                fused[emo] += self.weights['voice'] * v_probs[emo]
            active_weight_sum += self.weights['voice']

        # 얼굴
        if face_probs is not None:
            f_probs = self._face_to_probs(face_probs)
            f_emotion = max(f_probs, key=f_probs.get)
            modality_emotions['face'] = f_emotion
            for emo in self.emotion_names:
                fused[emo] += self.weights['face'] * f_probs[emo]
            active_weight_sum += self.weights['face']

        # 가중치 재정규화 (일부 모달리티가 없는 경우)
        if active_weight_sum > 0:
            fused = {emo: score / active_weight_sum for emo, score in fused.items()}

        # 최종 감정
        final_emotion = max(fused, key=fused.get)
        confidence = fused[final_emotion]

        # 충돌 감지 (모달리티별 예측이 다른 경우)
        unique_emotions = set(modality_emotions.values())
        conflict = len(unique_emotions) > 1

        return {
            'emotion': final_emotion,
            'confidence': confidence,
            'scores': fused,
            'modality_emotions': modality_emotions,
            'conflict': conflict,
        }


# ============ 테스트 ============
if __name__ == "__main__":
    fusion = EmotionFusion()

    print("=" * 60)
    print("멀티모달 감정 융합 테스트")
    print("=" * 60)

    # 테스트 1: 모든 모달리티 일치 (happy)
    print("\n[테스트 1] 모든 모달리티 일치 - happy")
    result = fusion.fuse(
        text_probs=[0.9, 0.02, 0.01, 0.02, 0.01, 0.02, 0.02],
        voice_probs=[0.85, 0.03, 0.02, 0.03, 0.02, 0.02, 0.03],
        face_probs={'happy': 80, 'sad': 5, 'angry': 3, 'surprise': 2, 'fear': 2, 'disgust': 3, 'neutral': 5},
    )
    print(f"  최종 감정: {result['emotion']} ({result['confidence']:.1%})")
    print(f"  모달리티별: {result['modality_emotions']}")
    print(f"  충돌: {result['conflict']}")

    # 테스트 2: 모달리티 간 충돌 (텍스트=angry, 음성=sad, 얼굴=angry)
    print("\n[테스트 2] 모달리티 충돌 - 텍스트:angry, 음성:sad, 얼굴:angry")
    result = fusion.fuse(
        text_probs=[0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05],
        voice_probs=[0.05, 0.7, 0.1, 0.03, 0.05, 0.02, 0.05],
        face_probs={'happy': 5, 'sad': 10, 'angry': 60, 'surprise': 5, 'fear': 5, 'disgust': 10, 'neutral': 5},
    )
    print(f"  최종 감정: {result['emotion']} ({result['confidence']:.1%})")
    print(f"  모달리티별: {result['modality_emotions']}")
    print(f"  충돌: {result['conflict']}")

    # 테스트 3: 비언어적 특성 반영 (텍스트=happy, 음성=angry, 얼굴=angry)
    # "웃기다"를 분노 상태에서 말하는 경우 → 실제는 어이없음(angry)
    print("\n[테스트 3] 비언어적 특성 - '웃기다'를 화난 상태에서 말함")
    print("  → 텍스트:happy, 음성:angry, 얼굴:angry")
    result = fusion.fuse(
        text_probs=[0.6, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05],
        voice_probs=[0.05, 0.05, 0.75, 0.03, 0.02, 0.05, 0.05],
        face_probs={'happy': 10, 'sad': 5, 'angry': 60, 'surprise': 10, 'fear': 5, 'disgust': 5, 'neutral': 5},
    )
    print(f"  최종 감정: {result['emotion']} ({result['confidence']:.1%})")
    print(f"  모달리티별: {result['modality_emotions']}")
    print(f"  충돌: {result['conflict']}")
    print(f"  → 텍스트는 happy지만 음성+얼굴이 angry → 최종: {result['emotion']}")

    # 테스트 4: 일부 모달리티 없는 경우 (텍스트만)
    print("\n[테스트 4] 텍스트만 있는 경우")
    result = fusion.fuse(
        text_probs=[0.1, 0.7, 0.05, 0.03, 0.05, 0.02, 0.05],
    )
    print(f"  최종 감정: {result['emotion']} ({result['confidence']:.1%})")
    print(f"  모달리티별: {result['modality_emotions']}")

    # 전체 점수 출력
    print("\n[테스트 3 상세 점수]")
    for emo, score in sorted(result['scores'].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 30)
        print(f"  {emo:10s}: {score:5.1%} {bar}")