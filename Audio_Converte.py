import os
import queue
import sys
import sounddevice as sd
import vosk
import json

# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\AI_Therapist2_2024\\vosk-model-small-en-us-0.15"

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
            print(result['text'])
        else:
            result = json.loads(rec.PartialResult())
            print(result['partial'])
