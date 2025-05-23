import whisper
import pandas as pd
import numpy as np
import torch.nn as nn



metadata = pd.read_csv('../../data/processed/Common Voice.csv')

model = whisper.load_model("turbo")

def data_spectrograme(data=metadata , kernel_size=100, stride=100):
    l=[]
    for i in range(len(data)):
        audio = whisper.load_audio(f'../../clips.wav/{data["wav_path"].iloc[i]}')
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        mel_pooled = pool(mel).squeeze(1)
        mel_numpy = mel_pooled.cpu().numpy()
        flat_array = mel_numpy.flatten()
        l.append(flat_array)
    
    # Crée une ligne avec 3839 colonnes
    row = np.random.rand((mel.shape[0]*mel.shape[1])/kernel_size)
    df = pd.DataFrame([row])
    df = df.drop(0)
    
    new_row = pd.DataFrame(l)
    df = pd.concat([df, new_row], ignore_index=True)
    return df