import os
import queue
import sys
import sounddevice as sd
import vosk
import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input
from tensorflow.keras.layers import GlobalAveragePooling2D

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\AI_Therapist2_2024\\vosk-model-small-en-us-0.15"
emotion_model_path = "Emotion_Voice_Detection_Model.h5"

# Load the Vosk model
if not os.path.exists(model_path):
    print(
        "Please download the Vosk model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

# Load the pre-trained emotion detection model
emotion_model = load_model(emotion_model_path)

# Build the emotion detection model
def build_emotion_model(input_shape=(40, 40, 1)):  # Adjust this shape as necessary
    model = Sequential()
    model.add(Input(shape=input_shape))  # Define the input layer here
    model.add(Conv2D(64, (5, 1), activation='relu'))  # Example Conv2D layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(8, activation='softmax'))  # Assuming 8 emotion classes
    return model

# Audio preprocessing to extract MFCC features
def preprocess_audio(audio_data, sample_rate=16000):
    # Convert byte data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize the audio data

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)

    # Transpose MFCCs to match the input shape: (timesteps, features, 1)
    mfccs = np.expand_dims(mfccs.T, axis=-1)  # Shape: (timesteps, 40, 1)

    return mfccs

# Function to detect emotion from audio
def detect_emotion(audio_data):
    # Preprocess the audio data
    mfccs = preprocess_audio(audio_data)

    # Reshape the data to fit the model's expected input (batch_size, timesteps, 40, 1)
    mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension: (1, timesteps, 40, 1)

    # Predict emotion using the pre-trained model
    emotion_prediction = emotion_model.predict(mfccs)

    # Convert prediction to label (assuming a softmax output with categorical labels)
    emotion_label = np.argmax(emotion_prediction)

    # Return the predicted emotion label
    return emotion_label

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Configure the audio stream
with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
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
