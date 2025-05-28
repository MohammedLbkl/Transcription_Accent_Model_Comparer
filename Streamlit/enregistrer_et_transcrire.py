import pyaudio
import wave
import whisper
import time # Pour la durée d'enregistrement

# --- Votre fonction de transcription Whisper ---
def whisper_transcription(file_path, model_name="tiny"):
    """
    Transcrit un fichier audio en utilisant OpenAI Whisper.

    Args:
        file_path (str): Chemin vers le fichier audio.
        model_name (str): Nom du modèle Whisper à utiliser (ex: "tiny", "base", "small", "medium", "large").
                          Les modèles plus grands sont plus précis mais plus lents et gourmands en ressources.
    Returns:
        str: Le texte transcrit.
    """
    try:
        model = whisper.load_model(model_name)

        # Charger l'audio et le préparer pour Whisper
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio) # S'assure que l'audio a la bonne longueur (30s)

        # Calculer le spectrogramme log-Mel
        # model.dims.n_mels donne le nombre de bandes Mel attendu par le modèle
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

        # Définir les options de décodage
        # fp16=False peut être nécessaire si vous n'avez pas de GPU compatible ou si vous rencontrez des erreurs.
        options = whisper.DecodingOptions(fp16=False if model.device.type == 'cpu' else True) # fp16=False sur CPU

        # Décoder l'audio
        result = whisper.decode(model, mel, options)

        return result.text
    except Exception as e:
        return f"Erreur lors de la transcription : {e}"

# --- Fonction pour enregistrer l'audio ---
def record_audio(filename="recorded_audio.wav", record_seconds=5, rate=16000, chunk_size=1024, channels=1, audio_format=pyaudio.paInt16):
    """
    Enregistre l'audio du microphone et le sauvegarde dans un fichier WAV.

    Args:
        filename (str): Nom du fichier WAV de sortie.
        record_seconds (int): Durée de l'enregistrement en secondes.
        rate (int): Taux d'échantillonnage (Hz). Whisper préfère 16000 Hz.
        chunk_size (int): Nombre de trames par buffer.
        channels (int): Nombre de canaux audio (1 pour mono, 2 pour stéréo). Mono est suffisant pour la parole.
        audio_format (pyaudio.paFormat): Format des échantillons audio.
    """
    p = pyaudio.PyAudio()

    print(f"Enregistrement audio pendant {record_seconds} secondes...")

    stream = p.open(format=audio_format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    frames = []

    for i in range(0, int(rate / chunk_size * record_seconds)):
        try:
            data = stream.read(chunk_size)
            frames.append(data)
        except IOError as e:
            if e.errno == pyaudio.paInputOverflowed:
                print("Avertissement : Input overflowed. Des données audio ont pu être perdues.")
                # On peut choisir de continuer ou d'ignorer ce chunk
            else:
                raise # Renvoyer d'autres IOErrors

    print("Enregistrement terminé.")

    # Arrêter et fermer le flux
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Sauvegarder les données audio dans un fichier WAV
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(audio_format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f"Audio sauvegardé sous : {filename}")
    return filename

# --- Programme principal ---
if __name__ == "__main__":
    output_filename = "mon_enregistrement.wav"
    duration_seconds = 10  # Vous pouvez changer cette durée ou la demander à l'utilisateur

    # Étape 1: Enregistrer l'audio
    try:
        recorded_file = record_audio(
            filename=output_filename,
            record_seconds=duration_seconds,
            rate=16000,       # Whisper est optimisé pour 16kHz
            channels=1        # Mono est généralement préférable pour la transcription vocale
        )
    except Exception as e:
        print(f"Une erreur est survenue lors de l'enregistrement : {e}")
        print("Vérifiez que vous avez un microphone connecté et que les permissions sont accordées.")
        exit()

    # Étape 2: Transcrire l'audio enregistré
    print("\nLancement de la transcription...")
    # Vous pouvez choisir un autre modèle si besoin : "base", "small", "medium", "large"
    # "tiny" est rapide mais moins précis. "base" est un bon compromis.
    model_to_use = "base" # Essayez "tiny" si "base" est trop lent, ou "small" pour plus de précision
    transcription = whisper_transcription(recorded_file, model_name=model_to_use)

    print("\n--- Transcription ---")
    print(transcription)
    print("---------------------")

    # Optionnel: Supprimer le fichier audio après transcription
    # import os
    # try:
    #     os.remove(recorded_file)
    #     print(f"Fichier {recorded_file} supprimé.")
    # except OSError as e:
    #     print(f"Erreur lors de la suppression du fichier {recorded_file}: {e}")