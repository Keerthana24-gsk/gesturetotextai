import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
gesture_name = input("Enter gesture label (e.g., A, B, Hello): ")

file = open('gesture_dataset.csv', 'a', newline='')
writer = csv.writer(file)

print("Press 's' to save data for current frame. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y])  # Collect x, y only

            if cv2.waitKey(1) & 0xFF == ord('s'):
                data.append(gesture_name)
                writer.writerow(data)
                print(f"Saved: {gesture_name}")

    cv2.imshow("Collecting Gesture Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
