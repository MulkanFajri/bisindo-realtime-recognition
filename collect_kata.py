import cv2
import mediapipe as mp
import os
import time

# =====================
# LABEL
# =====================

label = "Semoga Beruntung"

# =====================
# TARGET DATA
# =====================

TARGET = 100

# =====================
# INTERVAL SIMPAN
# =====================

CAPTURE_INTERVAL = 0.2

# =====================
# FOLDER
# =====================

save_dir = os.path.join(
    "dataset_kata",
    label
)

os.makedirs(
    save_dir,
    exist_ok=True
)

# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# CAMERA
# =====================

cap = cv2.VideoCapture(0)

count = len(os.listdir(save_dir))

capturing = False

last_capture_time = 0

# =====================
# LOOP
# =====================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    display = frame.copy()

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    hand_detected = False

    if results.multi_hand_landmarks:

        hand_detected = True

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # =====================
    # AUTO CAPTURE
    # =====================

    if capturing and hand_detected:

        current_time = time.time()

        if (
            current_time - last_capture_time
            >= CAPTURE_INTERVAL
        ):

            filename = os.path.join(
                save_dir,
                f"{count + 1}.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            count += 1

            last_capture_time = current_time

            print(
                f"[{count}] {filename}"
            )

    # =====================
    # STATUS
    # =====================

    status = (
        "Tangan Terdeteksi"
        if hand_detected
        else
        "Tidak Ada Tangan"
    )

    mode = (
        "CAPTURING"
        if capturing
        else
        "READY"
    )

    cv2.putText(
        display,
        f"Label : {label}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        display,
        f"Jumlah : {count}/{TARGET}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.putText(
        display,
        f"Status : {status}",
        (10,115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.putText(
        display,
        f"Mode : {mode}",
        (10,155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.putText(
        display,
        "[SPACE] Mulai Capture",
        (10,195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )

    cv2.putText(
        display,
        "[Q] Keluar",
        (10,235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

    cv2.imshow(
        f"Dataset {label}",
        display
    )

    key = cv2.waitKey(1)

    # =====================
    # START CAPTURE
    # =====================

    if key == 32 and not capturing:

        capturing = True

        last_capture_time = time.time()

        print(
            f"Mulai capture {label}"
        )

    # =====================
    # QUIT
    # =====================

    elif key == ord("q"):

        break

    # =====================
    # SELESAI
    # =====================

    if count >= TARGET:

        print(
            "Dataset selesai!"
        )

        break

cap.release()
cv2.destroyAllWindows()