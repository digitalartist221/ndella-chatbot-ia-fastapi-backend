# simple_transcriber.py
import speech_recognition as sr
import tempfile
import os

def simple_transcribe(audio_file_path: str) -> str:
    """
    Transcription ultra-simple qui fonctionne uniquement avec WAV
    """
    recognizer = sr.Recognizer()
    
    try:
        # Vérification basique
        if not audio_file_path.lower().endswith('.wav'):
            return "NDELLA_ERROR: Format non supporté. Veuillez envoyer un fichier WAV ou installer FFmpeg pour plus de formats."
        
        if not os.path.exists(audio_file_path):
            return "NDELLA_ERROR: Fichier introuvable"
        
        # Transcription directe
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="fr-FR")
            return text
            
    except sr.UnknownValueError:
        return "NDELLA_ERROR: Impossible de comprendre l'audio"
    except sr.RequestError as e:
        return f"NDELLA_ERROR: Service de reconnaissance: {e}"
    except Exception as e:
        return f"NDELLA_ERROR: {str(e)}"