import os
import queue
import sys
import sounddevice as sd
import vosk
import json
import time
import wave
import tempfile
from speechbrain.inference import EncoderClassifier
import warnings

#ignore warnings
warnings.filterwarnings("ignore")


# Path to the Vosk model
model_path = "C:\\Users\\25wal\\OneDrive\\Desktop\\Vanderbilt\\2025\\CS Module\\AI_Therapist2_2024\\vosk-model-small-en-us-0.15"

# Load the Vosk model
if not os.path.exists(model_path):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

model = vosk.Model(model_path)
q = queue.Queue()

# Function to save audio data to a temporary file
def save_audio_to_file(data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        with wave.open(tmpfile, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(16000)
            wf.writeframes(data)
        return tmpfile.name

# Load the emotion recognition model
emotion_model = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmp")
print(4)

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
    last_speech_time = time.time()
    silence_threshold = 1.0  # 1 second of silence
    speech_data = b''

    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result['text']
            print('Transcription: ', text)

            # Reset speech data and track time of last speech detected
            speech_data += data
            last_speech_time = time.time()

        else:
            result = json.loads(rec.PartialResult())
            print('Partial Transcription: ', result['partial'])

        # Check for silence (no transcription) for the specified threshold
        if time.time() - last_speech_time > silence_threshold and speech_data:
            print("Silence detected for 1 second, saving audio...")
            file_path = save_audio_to_file(speech_data)
            print(f"Audio saved to {file_path}")
            file_path = file_path.replace("\\", "\\\\")
            print(f"Audio saved to {file_path}")


            # Emotion recognition
            emotion = emotion_model.classify_file(file_path)
            print('Emotion: ', emotion)

            # Reset speech data after emotion classification
            speech_data = b''
