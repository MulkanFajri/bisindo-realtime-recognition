from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify
)
import cv2
import mediapipe as mp
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==========================
# MODEL HURUF & ANGKA (MLP)
# ==========================
model_huruf = joblib.load("models/bisindo_mlp.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# ==========================
# MODEL KATA (RANDOM FOREST)
# ==========================
model_kata = joblib.load(
    "models/kata_model.pkl"
)

actions = [
    "mendengar",
    "tersenyum",
    "semoga_beruntung",
    "ragu",
    "damai"
]

# ==========================
# MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================
# CAMERA
# ==========================
camera = cv2.VideoCapture(0)

# ==========================
# BUFFER KATA
# ==========================

last_label = "-"
last_confidence = 0

current_mode = "huruf"

def gen_frames():

    global current_mode 
    global last_label
    global last_confidence

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb)

        status = "Tidak Ada Tangan"
        
        label = "-"
        confidence = 0.0

        
        # =====================================
        # MODE HURUF & ANGKA (MLP)
        # =====================================
        if current_mode == "huruf":

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

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                features = left_hand + right_hand

                features = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)

                pred = model_huruf.predict(
                    features
                )

                prob = model_huruf.predict_proba(
                    features
                )

                confidence = float(
                    np.max(prob) * 100
                )

                label = encoder.inverse_transform(
                    pred
                )[0]

                last_label = label
                last_confidence = confidence

                print(
                    "API UPDATE:",
                    label,
                    confidence
                )

                status = "Tangan Terdeteksi"

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # =====================================
        # MODE KATA (RANDOM FOREST)
        # =====================================
        else:

            if results.multi_hand_landmarks:

                left_hand = [0] * 63
                right_hand = [0] * 63

                status = "Tangan Terdeteksi"

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

                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                features = np.array(
                    left_hand + right_hand,
                    dtype=np.float32
                ).reshape(1, -1)

                pred = model_kata.predict(
                    features
                )[0]

                prob = model_kata.predict_proba(
                    features
                )

                confidence = float(
                    np.max(prob) * 100
                )

                label = pred

                last_label = label
                last_confidence = confidence

                print(
                    "API UPDATE:",
                    label,
                    confidence
                )

        # =====================================
        # PANEL INFORMASI
        # =====================================
        overlay = frame.copy()
        
        cv2.rectangle(
            frame,
            (0, 0),
            (500, 160),
            (0, 0, 0),
            -1
        )
        
        alpha = 0.4

        cv2.addWeighted(
            overlay,
            alpha,
            frame,
            1 - alpha,
            0,
            frame
        )
        
        cv2.putText(
            frame,
            f"HASIL : {label}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"CONF : {confidence:.2f}%",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"STATUS : {status}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"MODE : {current_mode.upper()}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

@app.route("/set_mode", methods=["POST"])
def set_mode():

    global current_mode

    mode = request.json.get("mode")

    if mode in ["huruf", "kata"]:

        current_mode = mode


    print("MODE AKTIF =", current_mode)

    return jsonify({
        "mode": current_mode
    })

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/status")
def api_status():

    return jsonify({
        "status": "online",
        "mode": current_mode,
        "model_huruf": "MLP",
        "model_kata": "Random Forest"
    })

@app.route("/api/predict")
def api_predict():

    return jsonify({
        "mode": current_mode,
        "label": last_label,
        "confidence": round(last_confidence, 2)
    })
    
    
if __name__ == "__main__":
    app.run(debug=True)