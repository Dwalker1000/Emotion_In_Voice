import os
import queue
import sys
import sounddevice as sd
import vosk
import json
import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.sequence import pad_sequences

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\AI_Therapist2_2024\\vosk-model-small-en-us-0.15"

# Load the Vosk model
if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

# Load the emotion detection model
emotion_model_path = "Emotion_Voice_Detection_Model.h5"
try:
    emotion_model = load_model(emotion_model_path)
except Exception as e:
    print(f"Error loading emotion detection model: {e}")
    sys.exit(1)

# Print the model summary to understand the input shape
print(emotion_model.summary())

# Define the emotions
emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Function to detect emotion from audio
def detect_emotion(audio_data):
    # Preprocess the audio data as needed
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize the audio data
    audio_array = pad_sequences([audio_array], maxlen=16000, padding='post')  # Pad the sequence
    audio_array = np.expand_dims(audio_array, axis=-1)  # Add channel dimension if needed
    audio_array = np.expand_dims(audio_array, axis=0)  # Add batch dimension if needed
    audio_array = np.expand_dims(audio_array, axis=-1)  # Add another dimension if needed
    emotion_prediction = emotion_model.predict(audio_array)
    emotion = emotions[np.argmax(emotion_prediction)]
    return emotion

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
