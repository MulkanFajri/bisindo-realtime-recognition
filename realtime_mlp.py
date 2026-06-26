import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model
model = joblib.load("bisindo_mlp.pkl")
encoder = joblib.load("label_encoder.pkl")

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Webcam
cap = cv2.VideoCapture(0)

last_label = ""
counter = 0
final_label = ""

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        features = []

        for lm in hand.landmark:
            features.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        features = np.array(features).reshape(1, -1)

        pred = model.predict(features)

        label = encoder.inverse_transform(pred)[0]

        # stabilisasi prediksi
        if label == last_label:
            counter += 1
        else:
            counter = 0

        if counter >= 5:
            final_label = label

        last_label = label

        # gambar landmark
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

    # UI
    cv2.rectangle(
        frame,
        (0, 0),
        (400, 80),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"HASIL: {final_label}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow(
        "BISINDO MLP Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()