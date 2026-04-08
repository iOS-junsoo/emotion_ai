# 얼굴 감정 인식 테스트 코드

from deepface import DeepFace
import cv2
import time

print("웹캠 감정 인식 테스트")
print("종료하려면 'q' 키를 누르세요")

cap = cv2.VideoCapture(0)
time.sleep(2)  # 카메라 초기화 대기

if not cap.isOpened():
    print("카메라를 열 수 없습니다!")
    exit()

print("카메라 연결 성공!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 재시도 중...")
        time.sleep(0.5)
        continue

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
        emotion = result[0]['dominant_emotion']
        scores = result[0]['emotion']

        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        y = 60
        for emo, score in sorted(scores.items(), key=lambda x: -x[1]):
            cv2.putText(frame, f"{emo}: {score:.1f}%", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

    except Exception as e:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Emotion Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("테스트 종료")