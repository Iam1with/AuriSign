import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile

model = tf.keras.models.load_model("word_model.keras")

labels = ["hello","yes","no","thankyou","name"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def predict_sign(frame):

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x,lm.y,lm.z])

        landmarks = np.array(landmarks).reshape(1,-1)

        pred = model.predict(landmarks)

        label = labels[np.argmax(pred)]

        tts = gTTS(label)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)

        return label, tmp.name

    return "No hand detected", None

iface = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(source="webcam", type="numpy"),
    outputs=[gr.Text(), gr.Audio()],
    live=True,
)

iface.launch(server_name="0.0.0.0", server_port=10000)


