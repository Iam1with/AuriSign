import os
import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load trained model
model = tf.keras.models.load_model("word_model.keras", compile=False)

# Class labels (same order as training)
labels = ["hello", "yes", "no", "thankyou", "name"]

# Track last spoken prediction
last_prediction = ""

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def predict_sign(frame):
    global last_prediction

    if frame is None:
        return "Waiting...", None

    # Flip webcam
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize landmarks
        landmarks = landmarks - np.min(landmarks)
        landmarks = landmarks / (np.max(landmarks) + 1e-6)

        landmarks = landmarks.reshape(1, -1)

        # Predict
        pred = model.predict(landmarks, verbose=0)

        confidence = np.max(pred)
        class_id = np.argmax(pred)

        if confidence < 0.85:
            return "Detecting...", None

        label = labels[class_id]

        audio_file = None

        # Only generate speech if word changed
        if label != last_prediction:

            tts = gTTS(text=label, lang="en")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(tmp.name)

            audio_file = tmp.name

            last_prediction = label

        return label, audio_file

    return "No hand detected", None


# Build UI
with gr.Blocks() as demo:

    gr.Markdown("# ✋ AuriSign Real-Time Translator")

    with gr.Row():

        webcam = gr.Image(
            sources="webcam",
            streaming=True,
            type="numpy",
            label="Webcam"
        )

        with gr.Column():

            text_output = gr.Textbox(label="Detected Sign")

            audio_output = gr.Audio(
                label="Speech Output",
                autoplay=True
            )

    webcam.stream(
        fn=predict_sign,
        inputs=webcam,
        outputs=[text_output, audio_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
