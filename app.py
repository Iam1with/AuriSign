import os
import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load model
model = tf.keras.models.load_model("word_model.keras")
labels = ["hello", "yes", "no", "thankyou", "name"]

# Initialize MediaPipe Hands (Legacy API support enabled by pinning version 0.10.9)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def predict_sign(frame):
    if frame is None:
        return "Waiting...", None

    # Process frame
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Prepare for model
        landmarks = np.array(landmarks).reshape(1, -1)
        pred = model.predict(landmarks, verbose=0)
        label = labels[np.argmax(pred)]

        # Generate audio
        tts = gTTS(label)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        
        return label, tmp.name

    return "No hand detected", None

# Build UI with streaming enabled
with gr.Blocks() as demo:
    gr.Markdown("# AuriSign Real-Time Translator")
    with gr.Row():
        input_img = gr.Image(sources="webcam", streaming=True, type="numpy")
        with gr.Column():
            label_out = gr.Textbox(label="Detected Sign")
            audio_out = gr.Audio(label="Audio Output", autoplay=True)

    # Stream the input to the prediction function
    input_img.stream(fn=predict_sign, inputs=input_img, outputs=[label_out, audio_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
