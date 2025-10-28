from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import os

from nlu_model import ndella_chatbot
from audio_transcriber import transcribe_audio

app = FastAPI(
    title="Ndella Chatbot API",
    description="Backend NLU pour le Centre de Ressources de Dakar (CRD) avec support Audio.",
)

# Configuration CORS pour le frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # À adapter pour la prod : ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    input_type: str
    transcription: str = None

# --- Endpoint 1: Traitement de Texte ---
@app.post("/chat", response_model=ChatResponse)
async def chat_text(request: MessageRequest):
    if not ndella_chatbot:
        raise HTTPException(status_code=503, detail="Le modèle NLU n'est pas initialisé.")
        
    try:
        response = ndella_chatbot.get_response(request.message)
        return ChatResponse(response=response, input_type="text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du texte: {e}")
# main.py - Endpoint audio amélioré
@app.post("/chat/audio")
async def chat_audio(audio_file: UploadFile = File(...)):
    """
    Endpoint amélioré pour le traitement audio
    """
    print(f"🎯 Requête audio reçue: {audio_file.filename}")
    
    temp_path = None
    
    try:
        # Validation du fichier
        if not audio_file.content_type.startswith('audio/'):
            return {
                "input_type": "audio_error",
                "transcription": "Format de fichier non supporté",
                "response": "Veuillez envoyer un fichier audio valide."
            }
        
        # Sauvegarder le fichier temporairement
        file_extension = os.path.splitext(audio_file.filename or "audio")[1]
        temp_path = f"temp_audio_{uuid.uuid4()}{file_extension}"
        
        with open(temp_path, "wb") as f:
            content = await audio_file.read()
            print(f"📥 Fichier reçu: {len(content)} bytes")
            
            if len(content) < 1000:
                return {
                    "input_type": "audio_error", 
                    "transcription": "Fichier audio trop court",
                    "response": "L'audio est trop court. Parlez pendant au moins 2 secondes."
                }
                
            f.write(content)
        
        # Transcription
        transcription = transcribe_audio(temp_path)
        print(f"📝 Résultat transcription: {transcription}")
        
        # Gestion des erreurs de transcription
        if transcription.startswith("NDELLA_ERROR:"):
            error_msg = transcription.replace("NDELLA_ERROR:", "").strip()
            return {
                "input_type": "audio_error",
                "transcription": error_msg,
                "response": "Je n'ai pas pu comprendre votre message audio. " + get_friendly_audio_advice(error_msg)
            }
        
        # Traitement normal avec le LLM
        llm_response = await generate_llm_response(transcription)
        
        return {
            "input_type": "audio",
            "transcription": transcription,
            "response": llm_response
        }
        
    except Exception as e:
        print(f"💥 Erreur endpoint audio: {e}")
        return {
            "input_type": "audio_error",
            "transcription": f"Erreur technique: {str(e)}",
            "response": "Une erreur technique s'est produite. Veuillez réessayer."
        }
    finally:
        # Nettoyage
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

def get_friendly_audio_advice(error_msg: str) -> str:
    """
    Retourne des conseils amicaux selon l'erreur
    """
    advice_map = {
        "trop court": "Assurez-vous de parler pendant au moins 2-3 secondes.",
        "incompréhensible": "Parlez clairement, dans un environnement calme, et vérifiez votre microphone.",
        "aucun son": "Vérifiez que votre microphone fonctionne et n'est pas muet.",
        "indisponible": "Vérifiez votre connexion internet.",
    }
    
    for key, advice in advice_map.items():
        if key in error_msg.lower():
            return advice
    
    return "Essayez de parler plus clairement dans un environnement calme."

async def generate_llm_response(message: str) -> str:
    """
    Votre logique LLM habituelle
    """
    # Exemple - remplacez par votre vrai code
    return f"J'ai bien reçu votre message audio : '{message}'. Comment puis-je vous aider?"
    
@app.get("/health")
async def health_check():
    return {"status": "healthy", "transcriber": "active"}