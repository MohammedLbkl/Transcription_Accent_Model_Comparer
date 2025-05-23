from vosk import Model, KaldiRecognizer
import wave
import json
import whisper

def whisper_transcription(file_path, model_name="tiny"):
    model = whisper.load_model(model_name)


    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)


    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    result = result.text
    return result


def vosk_transcription(file_path):
    wf = wave.open(file_path, 'rb')
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    full_text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            
    full_text = json.loads(rec.Result()).get("text")
    if full_text=="":
        full_text = result.get("text")

    return full_text
