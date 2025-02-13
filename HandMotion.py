import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

class TrainedGestureRecognition:
    def __init__(self, model_path, max_history=15):
        self.model = joblib.load(model_path)
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_gesture(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        gesture = None
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                self.position_history.append((np.mean([lm.x for lm in hand_landmarks.landmark]),
                                              np.mean([lm.y for lm in hand_landmarks.landmark])))

                # Use the classifier to predict probabilities.
                # Reshape landmarks into the expected input shape.
                features = np.array(landmarks).reshape(1, -1)
                proba = self.model.predict_proba(features)[0]
                predicted_index = np.argmax(proba)
                predicted_label = self.model.classes_[predicted_index]
                confidence = proba[predicted_index]
                # You can set a threshold here if needed.
                if confidence >= 0.5:
                    gesture = predicted_label
                else:
                    gesture = None
        else:
            self.position_history.clear()

        return gesture, confidence, image

def main():
    cap = cv2.VideoCapture(0)
    gesture_recognizer = TrainedGestureRecognition(
    "/Users/granthenderson/Desktop/Projects/Hand Detection Project/gesture_classifier.pkl",
    max_history=15
)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gesture, conf, annotated_frame = gesture_recognizer.detect_gesture(frame)
        if gesture:
            cv2.putText(annotated_frame, f"Gesture: {gesture} ({conf*100:.0f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, "No Gesture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
