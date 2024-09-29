import os
import queue
import sys
import sounddevice as sd
import vosk
import json

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\Emotion_in_Voice\\vosk-model-small-en-us-0.15"

# Load the Vosk model
if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

# Function to detect emotion from the transcribed text using simple keyword matching
def detect_emotion(text):
    # Keywords for each emotion
    keywords = {
        "happy": ["happy", "joy", "excited", "awesome", "great", "good"],
        "sad": ["sad", "unhappy", "depressed", "cry", "down"],
        "angry": ["angry", "mad", "furious", "upset", "annoyed"],
        "fearful": ["scared", "afraid", "fear", "terrified", "worried"],
        "calm": ["calm", "relaxed", "peaceful", "chill", "okay"],
        "surprised": ["surprised", "shocked", "amazed", "wow"],
        "disgust": ["disgusted", "gross", "nasty", "horrible"],
        "frustrated": ["frustrated", "annoyed", "irritated", "stressed", "fed up"]
    }

    # Check the text for any of the keywords
    for emotion, words in keywords.items():
        for word in words:
            if word in text.lower():
                return emotion.capitalize()

    return "Neutral"

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
            emotion = detect_emotion(text)  # Detect emotion from the transcribed text
            print(f"Text: {text}, Emotion: {emotion}")
        else:
            result = json.loads(rec.PartialResult())
            print(result['partial'])
