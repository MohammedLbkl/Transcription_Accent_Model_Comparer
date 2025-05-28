import streamlit as st
import pyaudio
import wave
import whisper
import time
import os
import tempfile 
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib  


st.set_page_config(page_title="Audio Transcriber", layout="wide")

# --- Charger le modèle et le label encoder ---
@st.cache_resource
def load_accent_model_and_encoder():
    modele = joblib.load("../outputs/models/modele_logistique.pkl")  # à adapter à ton chemin
    label_encoder = joblib.load("../outputs/label_encoder/label_encoder.pkl")
    return modele, label_encoder

modele_accent, label_encoder = load_accent_model_and_encoder()

# --- Fonction pour prédire l'accent (ta fonction, légèrement modifiée pour intégrer le label_encoder) ---
def pred_accent(file_path, modele, label_encoder):
    model = whisper.load_model("turbo")  

    kernel_size = 100
    stride = 100

    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
    mel_pooled = pool(mel).squeeze(1)
    mel_numpy = mel_pooled.cpu().numpy()
    flat_array = mel_numpy.flatten()

    df = pd.DataFrame([flat_array])

    y_pred = modele.predict(df)
    y_pred_proba = modele.predict_proba(df)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    return y_pred_labels[0], np.max(y_pred_proba)

# --- Votre fonction de transcription Whisper (inchangée) ---
def whisper_transcription(file_path, model_name="tiny"):

    with st.spinner(f"Chargement du modèle Whisper '{model_name}'... (peut prendre du temps la première fois)"):
        model = whisper.load_model(model_name)

    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    options = whisper.DecodingOptions(fp16=False if model.device.type == 'cpu' else True, language="en") # Spécifier la langue peut aider
    result = whisper.decode(model, mel, options)


# --- Fonction pour enregistrer l'audio (légèrement modifiée pour Streamlit) ---
def record_audio(filename_prefix="temp_audio_streamlit", record_seconds=5, rate=16000, chunk_size=1024, channels=1, audio_format=pyaudio.paInt16):
    p = pyaudio.PyAudio()

    # Crée un fichier temporaire pour l'enregistrement
    temp_file = tempfile.NamedTemporaryFile(prefix=filename_prefix, suffix=".wav", delete=False)
    output_filename = temp_file.name
    temp_file.close() # Fermer le handle pour que wave puisse l'ouvrir

    # Message dans Streamlit
    status_placeholder = st.empty()
    status_placeholder.info(f"Enregistrement en cours pendant {record_seconds} secondes... Parlez maintenant !")

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    frames = []
    for i in range(0, int(rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)


    status_placeholder.success(f"✅ Enregistrement terminé. Sauvegarde en cours...")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    status_placeholder.empty() # Effacer le message de statut
    return output_filename




# --- Interface Streamlit ---
st.title("🎤 Enregistreur et Transcripteur Audio avec OpenAI Whisper")
st.markdown("Enregistrez un court message audio et obtenez sa transcription.")


# Initialiser l'état de session pour stocker le chemin du fichier et la transcription
if "audio_file_path" not in st.session_state:
    st.session_state.audio_file_path = None
if "transcription_text" not in st.session_state:
    st.session_state.transcription_text = ""
    
    
# Options pour l'utilisateur
col1, col2 = st.columns(2)
with col1:
    duration_seconds = st.slider(
        "Durée de l'enregistrement (secondes) :",
        min_value=3,
        max_value=30,
        value=5,
        step=1
    )
with col2:
    model_options = ["tiny", "base", "small", "medium", "turbo", "large"]
    selected_model = st.selectbox(
        "Choisissez le modèle Whisper :",
        options=model_options,
        index=model_options.index("base"),
        help="Les modèles plus grands sont plus précis mais plus lents et gourmands en ressources."
    )



# Bouton d'enregistrement et de transcription
if st.button("Démarrer l'enregistrement et Transcrire", type="primary", use_container_width=True):
    # Supprime l'ancien fichier audio s'il existe
    if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):      
        os.remove(st.session_state.audio_file_path)
    
    st.session_state.audio_file_path = None
    st.session_state.transcription_text = ""


    # Étape 1: Enregistrer l'audio
    with st.spinner("Préparation de l'enregistrement..."):
        recorded_file_path = record_audio(
            record_seconds=duration_seconds,
            rate=16000,
            channels=1
        )
    st.session_state.audio_file_path = recorded_file_path
    st.success(f"Audio enregistré : {os.path.basename(recorded_file_path)}")

    # Étape 2: Transcrire l'audio enregistré
    with st.spinner(f"Transcription avec le modèle '{selected_model}' en cours... Ceci peut prendre un moment."):
        transcription = whisper_transcription(st.session_state.audio_file_path, model_name=selected_model)

    if transcription:
        st.session_state.transcription_text = transcription
    else:
        st.session_state.transcription_text = "La transcription a échoué ou n'a rien retourné."



# Afficher le lecteur audio et la transcription s'ils existent
if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
    st.subheader("Audio Enregistré :")
    with open(st.session_state.audio_file_path, 'rb') as audio_file:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

if st.session_state.transcription_text:
    st.subheader("📝 Transcription :")
    st.text_area("Texte transcrit", st.session_state.transcription_text, height=200,
                 help="Vous pouvez copier ce texte.")


# --- Ajouter la prédiction d'accent après transcription ---
if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):

    with st.spinner("🔎 Prédiction de l'accent en cours..."):
        predicted_accent, confidence = pred_accent(
            st.session_state.audio_file_path, modele_accent, label_encoder
        )

    st.subheader("🌍 Accent détecté :")
    st.markdown(f"**Accent :** {predicted_accent}  \n**Confiance :** {confidence:.2%}")


st.markdown("---")
st.caption("Application développée avec Streamlit et OpenAI Whisper.")

# Note sur le nettoyage :
# Les fichiers temporaires créés avec delete=False ne sont pas automatiquement supprimés
# à la fermeture du handle. Nous les supprimons au début d'un nouvel enregistrement.
# Le dernier fichier créé pourrait rester jusqu'à ce que le système d'exploitation nettoie /tmp
# ou que l'application soit redémarrée et qu'un nouvel enregistrement soit fait.
# Pour une application en production, une stratégie de nettoyage plus robuste serait nécessaire.