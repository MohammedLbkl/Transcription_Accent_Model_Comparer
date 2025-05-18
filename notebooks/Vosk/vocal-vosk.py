import sounddevice as sd
import queue
import sys
import json
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))
    
model = Model(lang="en-us")
rec = KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                       channels=1, callback=callback):
    print("Parlez dans le micro (Ctrl+C pour arrêter).")
    try:
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = rec.Result()
                print(json.loads(result)["text"])
            else:
                partial = rec.PartialResult()
                
    except KeyboardInterrupt:
        print("\n Arrêté par l'utilisateur")
