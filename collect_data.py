import cv2
import mediapipe as mp
import csv
import time

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Define a mapping from key to gesture label.
gesture_keys = {
    ord('1'): "Thumbs Up",
    ord('2'): "Thumbs Down",
    ord('3'): "Wave",
    ord('4'): "Up",
    ord('5'): "Down"
}

# CSV file where data will be saved.
csv_file = "hand_gesture_data.csv"

# Open CSV file for appending data.
with open(csv_file, mode='a', newline='') as f:
    csv_writer = csv.writer(f)
    # Optionally, write header if file is empty.
    # header: gesture label followed by 42 landmark values (x and y for 21 landmarks)
    # csv_writer.writerow(["label"] + [f"{i}_{coord}" for i in range(21) for coord in ["x", "y"]])

    cap = cv2.VideoCapture(0)
    print("Press keys 1-5 to label the gesture. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Prepare landmark data: we'll store x and y of each landmark.
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                # Display the landmarks on screen (for debugging).
                cv2.putText(frame, f"Collected: {len(landmarks)} values", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key to exit.
            break
        elif key in gesture_keys and results.multi_hand_landmarks:
            # Save the landmarks with the corresponding gesture label.
            label = gesture_keys[key]
            # Using the first detected hand.
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            # Write the label and landmark data to CSV.
            csv_writer.writerow([label] + landmarks)
            print(f"Saved gesture: {label}")
            # Optional: add a short delay so you have time to reposition your hand.
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
