# scripts/analyze_label_patterns.py
import sys
from pathlib import Path
from collections import defaultdict
from transformers import pipeline

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_label_patterns():
    """다양한 감정 텍스트에 대해 모델이 반환하는 LABEL 패턴 분석"""
    
    # 다양한 감정의 테스트 케이스
    test_cases = {
        "happy": [
            "오늘 정말 기분이 좋아요!",
            "너무 행복해요",
            "기쁘고 즐거워요",
            "만족스럽고 뿌듯해요",
            "신나고 설레요",
            "즐겁고 재미있어요",
        ],
        "sad": [
            "요즘 너무 우울하고 힘들어요",
            "슬프고 외로워요",
            "절망적이고 힘들어요",
            "실망스럽고 아파요",
            "우울하고 침울해요",
            "슬퍼서 눈물이 나요",
        ],
        "angry": [
            "화가 나서 참을 수가 없어요",
            "짜증나고 분노가 치밀어요",
            "화나고 불만이 많아요",
            "분노하고 화가 나요",
            "짜증스럽고 답답해요",
            "화가 치밀어 올라요",
        ],
        "fear": [
            "무서워서 잠을 못 자겠어요",
            "두려워하고 불안해요",
            "걱정되고 불안해요",
            "공포스럽고 무서워요",
            "불안하고 걱정돼요",
            "두려움에 떨어요",
        ],
        "surprise": [
            "놀랍고 신기해요",
            "충격적이고 놀라워요",
            "당황스럽고 놀라워요",
            "예상치 못해서 놀라요",
            "깜짝 놀랐어요",
            "놀라운 일이에요",
        ],
        "disgust": [
            "역겨워서 못 보겠어요",
            "혐오스럽고 역겨워요",
            "징그럽고 역겨워요",
            "혐오감이 들어요",
            "역겨운 느낌이에요",
        ],
        "neutral": [
            "오늘 날씨가 좋네요",
            "그냥 평범한 하루예요",
            "특별한 일은 없어요",
            "보통이에요",
            "평온하고 조용해요",
        ],
    }
    
    # 모델 로드
    print("모델 로딩 중...")
    classifier = pipeline(
        "text-classification",
        model="M1NJ1/klue-bert-emotion",
        device=0 if __import__('torch').cuda.is_available() else -1
    )
    print("모델 로드 완료!\n")
    
    # 각 감정별로 상위 LABEL 수집
    emotion_to_labels = defaultdict(lambda: defaultdict(int))
    
    print("=" * 80)
    print("패턴 분석 시작")
    print("=" * 80)
    
    for expected_emotion, texts in test_cases.items():
        print(f"\n[{expected_emotion.upper()}] 감정 그룹 분석")
        print("-" * 80)
        
        for text in texts:
            # 상위 10개 LABEL 가져오기
            results = classifier(text, top_k=10)
            
            # 상위 3개만 출력
            top_labels = []
            for i, item in enumerate(results[:3]):
                label = item['label']
                score = item['score']
                top_labels.append((label, score))
                emotion_to_labels[expected_emotion][label] += score
            
            print(f"  텍스트: {text}")
            print(f"    상위 LABEL: {', '.join([f'{l}({s*100:.1f}%)' for l, s in top_labels])}")
    
    # 패턴 요약
    print("\n" + "=" * 80)
    print("패턴 분석 결과 요약")
    print("=" * 80)
    
    for emotion, label_scores in emotion_to_labels.items():
        print(f"\n[{emotion.upper()}] 감정에 자주 나타나는 LABEL들:")
        # 점수 순으로 정렬
        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        for label, total_score in sorted_labels[:15]:  # 상위 15개
            avg_score = total_score / len(test_cases[emotion])
            print(f"  {label}: {avg_score*100:.2f}% (총 {total_score*100:.2f}%)")
    
    # 매핑 제안
    print("\n" + "=" * 80)
    print("매핑 제안 (emotion_groups 딕셔너리용)")
    print("=" * 80)
    
    # 각 감정별로 상위 LABEL들 추출
    suggested_mapping = {}
    for emotion in ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]:
        label_scores = emotion_to_labels[emotion]
        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 LABEL들의 ID 추출
        label_ids = []
        for label, score in sorted_labels[:10]:  # 상위 10개
            if label.startswith("LABEL_"):
                try:
                    label_id = int(label.replace("LABEL_", ""))
                    label_ids.append(label_id)
                except:
                    pass
        
        suggested_mapping[emotion] = sorted(label_ids)
    
    print("\n# 제안된 매핑:")
    print("emotion_groups = {")
    for emotion, label_ids in suggested_mapping.items():
        print(f'    "{emotion}": {label_ids},')
    print("}")

if __name__ == "__main__":
    analyze_label_patterns()