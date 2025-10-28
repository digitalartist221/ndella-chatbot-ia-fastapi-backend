# audio_transcriber.py
import speech_recognition as sr
import tempfile
import os
from pydub import AudioSegment

def transcribe_audio(audio_file_path: str) -> str:
    """
    Transcription audio robuste avec meilleure gestion d'erreurs
    """
    recognizer = sr.Recognizer()
    temp_wav_path = None
    
    try:
        print(f"🔍 Début transcription: {audio_file_path}")
        
        # Vérifications initiales
        if not os.path.exists(audio_file_path):
            return "NDELLA_ERROR: Fichier audio introuvable"
        
        file_size = os.path.getsize(audio_file_path)
        print(f"📏 Taille fichier: {file_size} bytes")
        
        if file_size == 0:
            return "NDELLA_ERROR: Fichier audio vide"
        
        if file_size < 1000:  # Moins de 1KB
            return "NDELLA_ERROR: Fichier audio trop court"
        
        # Conversion en WAV pour une meilleure compatibilité
        temp_wav_path = "temp_audio.wav"
        
        try:
            # Essayer de détecter le format automatiquement
            audio = AudioSegment.from_file(audio_file_path)
            print(f"🎵 Format détecté, durée: {len(audio)}ms")
            
            # Configurer pour une meilleure reconnaissance
            audio = audio.set_frame_rate(16000)  # 16kHz optimal pour reconnaissance
            audio = audio.set_channels(1)        # Mono
            audio = audio.set_sample_width(2)    # 16-bit
            
            audio.export(temp_wav_path, format="wav")
            print("✅ Conversion WAV réussie")
            
        except Exception as e:
            print(f"❌ Erreur conversion: {e}")
            # Si la conversion échoue, essayer avec le fichier original
            temp_wav_path = audio_file_path
        
        # Transcription avec SpeechRecognition
        with sr.AudioFile(temp_wav_path) as source:
            print("🎯 Début reconnaissance vocale...")
            
            # Ajustement du bruit ambiant avec timeout
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            # Enregistrement avec timeout
            audio_data = recognizer.record(source)
            
            # Vérifier qu'il y a des données audio
            if len(audio_data.frame_data) == 0:
                return "NDELLA_ERROR: Aucun son détecté dans l'audio"
            
            print(f"🎙️ Données audio: {len(audio_data.frame_data)} bytes")
            
            # Reconnaissance Google avec timeout
            text = recognizer.recognize_google(
                audio_data, 
                language="fr-FR",
                show_all=False
            )
            
            print(f"✅ Transcription réussie: {text}")
            return text
            
    except sr.UnknownValueError:
        print("❌ Audio incompréhensible")
        return "NDELLA_ERROR: Impossible de comprendre l'audio. Parlez plus clairement et vérifiez votre micro."
    
    except sr.RequestError as e:
        print(f"❌ Erreur service Google: {e}")
        return f"NDELLA_ERROR: Service de reconnaissance vocale indisponible: {e}"
    
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return f"NDELLA_ERROR: Erreur lors du traitement: {str(e)}"
    
    finally:
        # Nettoyage
        if temp_wav_path and temp_wav_path != audio_file_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                print("🧹 Fichier temporaire nettoyé")
            except:
                pass