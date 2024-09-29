#works but returns everything as neutral

import os
import queue
import sys
import sounddevice as sd
import vosk
import json
import numpy as np
from speechbrain.inference import Tacotron2

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\Emotion_in_Voice\\vosk-model-small-en-us-0.15"

# Load the Vosk model
if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

# Initialize the emotion detection model (you may need to change this based on the model you choose)
# This is just an example; make sure to load your trained model
emotion_model = None  # Load your emotion detection model here

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Function to detect emotion from audio
def detect_emotion(audio_data):
    # Preprocess the audio data as needed
    # Make sure your emotion model accepts the input format
        # audio_array = np.frombuffer(audio_data, dtype=np.float32)
        # emotion = emotion_model.predict(audio_array)  # Use the model to predict emotion
        # return emotion
    return "Neutral"  # Placeholder return

# Configure the audio stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print('#' * 80)
    print('Press Ctrl+C to stop the recording')
    print('#' * 80)

    rec = vosk.KaldiRecognizer(model, 16000)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result['text']
            emotion = detect_emotion(data)  # Detect emotion from the raw audio data
            print(f"Text: {text}, Emotion: {emotion}")
        else:
            result = json.loads(rec.PartialResult())
            print(result['partial'])
