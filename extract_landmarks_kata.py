import os
import cv2
import mediapipe as mp
import pandas as pd

# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands

# =====================
# DATASET
# =====================

dataset_path = "dataset_kata"

data = []

# =====================
# EXTRACT
# =====================

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    for label in os.listdir(dataset_path):

        label_folder = os.path.join(
            dataset_path,
            label
        )

        if not os.path.isdir(label_folder):
            continue

        print(f"Processing {label} ...")

        for image_name in os.listdir(label_folder):

            image_path = os.path.join(
                label_folder,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(
                image_rgb
            )

            if results.multi_hand_landmarks:

                left_hand = [0] * 63
                right_hand = [0] * 63

                for hand_index, hand_landmarks in enumerate(
                    results.multi_hand_landmarks
                ):

                    temp = []

                    for lm in hand_landmarks.landmark:

                        temp.extend([
                            lm.x,
                            lm.y,
                            lm.z
                        ])

                    if hand_index == 0:

                        left_hand = temp

                    elif hand_index == 1:

                        right_hand = temp

                row = (
                    left_hand +
                    right_hand
                )

                row.append(
                    label
                )

                data.append(
                    row
                )

# =====================
# COLUMN
# =====================

columns = []

for i in range(21):

    columns.extend([
        f"left_x{i}",
        f"left_y{i}",
        f"left_z{i}"
    ])

for i in range(21):

    columns.extend([
        f"right_x{i}",
        f"right_y{i}",
        f"right_z{i}"
    ])

columns.append(
    "label"
)

# =====================
# SAVE CSV
# =====================

df = pd.DataFrame(
    data,
    columns=columns
)

df.to_csv(
    "kata_landmarks.csv",
    index=False
)

print("\nSelesai!")
print(
    "File tersimpan: kata_landmarks.csv"
)
print(
    "Jumlah data:",
    len(df)
)
print(
    "Shape:",
    df.shape
)