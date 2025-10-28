import nltk
import pickle
import numpy as np
import json
import random
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from typing import List, Dict, Any

# Définition des chemins de fichiers (à ajuster si besoin)
MODEL_PATH = 'checkpoints/chatbot_model.h5'
WORDS_PATH = 'checkpoints/words.pkl'
CLASSES_PATH = 'checkpoints/classes.pkl'
INTENTS_PATH = 'data/intention_crd_bambey.json'
COHERENCE_THRESHOLD = 0.50 # Seuil de confiance pour une réponse non-générique

class NdellaNLU:
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.model = load_model(MODEL_PATH)
            # Utilisation du nom de fichier fourni par l'utilisateur
            with open(INTENTS_PATH, encoding='utf-8') as f:
                self.intents = json.load(f)
            self.words = pickle.load(open(WORDS_PATH, 'rb'))
            self.classes = pickle.load(open(CLASSES_PATH, 'rb'))
            print("Ndella NLU Model loaded successfully.")
        except Exception as e:
            print(f"Error loading NLU components: {e}")
            raise

    def clean_up_sentence(self, sentence: str) -> List[str]:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence: str) -> np.ndarray:
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)  
        for s in sentence_words:
            for i, word in enumerate(self.words):
                if word == s:  
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence: str) -> List[Dict[str, Any]]:
        p = self.bag_of_words(sentence)
        # Validation d'entrée pour le modèle
        if p.sum() == 0:
            return [] # Aucune correspondance de mots
            
        # Le modèle Keras attend un batch
        res = self.model.predict(np.array([p]))[0]
        
        results = [[i, r] for i, r in enumerate(res)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": float(r[1])})
        return return_list

    def get_response(self, sentence: str) -> str:
        ints = self.predict_class(sentence)
        
        if not ints or ints[0]['probability'] < COHERENCE_THRESHOLD:
            # Réponse de secours si la confiance est faible
            return "Désolé, je suis Ndella, l'assistante du CRD. Je n'ai pas compris votre question. Pouvez-vous reformuler s'il vous plaît ?"
        
        tag = ints[0]['intent']
        list_of_intents = self.intents['intentions']
        
        for i in list_of_intents:
            if i['tag'] == tag:
                # Réponse aléatoire choisie parmi celles du tag
                return random.choice(i['réponses'])
                
        return "Je suis désolée, une erreur interne s'est produite."

# Initialisation globale
try:
    ndella_chatbot = NdellaNLU()
except Exception:
    print("FATAL: NLU Chatbot failed to initialize. Check your model and data files.")
    ndella_chatbot = None # Pour ne pas bloquer l'API mais empêcher l'utilisation du chat