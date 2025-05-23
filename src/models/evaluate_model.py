import pandas as pd
import whisper
from jiwer import wer
from vosk import Model, KaldiRecognizer
import wave
import json
import re
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

metadata = pd.read_csv('../../data/processed/Common Voice.csv')

def Whisper_evaluate(model_name = "tiny", data = metadata):
    print(f"Model: {model_name}")
    list_score = []
    list_duration = []
    for i in range(30):

        model = whisper.load_model(model_name)
        
        start = time.time()
        
        
        audio = whisper.load_audio(f'../../clips.wav/{data["wav_path"].iloc[i]}')
        audio = whisper.pad_or_trim(audio)

        
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        
        
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        end = time.time()

        duration = end - start
        
        reference = metadata["sentence"].iloc[i].lower()  # conversion en minuscules
        reference = re.sub(r'[^\w\s]', '', reference)  # supprime tout sauf lettres/chiffres/espaces
        

        hypothesis = result.text.lower()  # conversion en minuscules
        hypothesis = re.sub(r'[^\w\s]', '', hypothesis)  # supprime tout sauf lettres/chiffres/espaces

        score = wer(reference, hypothesis)

        list_score.append(score)
        list_duration.append(duration)
        
    moyenne_duration = sum(list_duration) / len(list_duration)
    moyenne_score = sum(list_score) / len(list_score)
    print(f"WER moyen: {moyenne_score:.2%}")
    print(f"Durée moyenne de transcription : {moyenne_duration:.2f} secondes")
    
    
    
    
    
    
    
def vosk_evaluation(data=metadata):
    list_score = []
    list_duration = []
    for i in range(30):
        wf = wave.open(f'../../clips.wav/{data["wav_path"].iloc[i]}', 'rb')
        model = Model(lang="en-us")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        full_text = ""

        start = time.time()

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                
        
        end = time.time()

        duration = end - start
        
        full_text = json.loads(rec.Result()).get("text")
        if full_text=="":
            full_text = result.get("text")
            
        full_text = re.sub(r'[^\w\s]', '', full_text)  # supprime tout sauf lettres/chiffres/espaces
        hypothesis = full_text.lower()  # conversion en minuscules


        reference = metadata["sentence"].iloc[i].lower()  # conversion en minuscules
        reference = re.sub(r'[^\w\s]', '', reference)  # supprime tout sauf lettres/chiffres/espaces
        
        score = wer(reference, hypothesis)

        list_score.append(score)
        list_duration.append(duration)
            
    moyenne_duration = sum(list_duration) / len(list_duration)
    moyenne = sum(list_score) / len(list_score)

    print(f"WER moyen: {moyenne:.2%}")
    print(f"Durée moyenne de transcription : {moyenne_duration:.2f} secondes")
    
    
    
    
    
    
df = pd.read_csv('../../data/processed/spectrogram_final.csv') 
df = df.drop("wav_path", axis=1) 
y = df["accents"] 

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)


def evaluation_accent(X_train, y_train, X_test, y_test,model=LogisticRegression(solver='lbfgs', max_iter=1000)):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(confusion_matrix(y_test, y_pred))