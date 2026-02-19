from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Documents stockés simplement sans vectordb
documents = [
    {
        "content": """FICHE DÉPANNAGE MSV-1000 - Écran noir
SYMPTÔME : Écran totalement noir, pas de lumière, aucun bip au démarrage.
CAUSE FRÉQUENTE : Batterie complètement déchargée.
SOLUTION : Brancher sur secteur 4 heures minimum.
PIÈCE : Batterie lithium-ion 14.4V, référence BAT-MSV1000.
Code erreur associé : ERR 01""",
        "metadata": {
            "source": "Manuel MSV-1000 v1.0",
            "section": "Dépannage Écran Noir",
            "code_erreur": "ERR 01"
        }
    }
]

# LLM Groq uniquement
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.1)

template = """
Tu es BioAssist, assistant expert en dispositifs médicaux.
Voici la documentation technique : {context}
Question du technicien : {question}
Réponds de façon structurée et professionnelle.
RÉPONSE :
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def bioassist_repondre(question):
    # Sans vectordb : on envoie tous les documents directement
    contexte = "\n\n".join([d["content"] for d in documents])
    message = prompt.format(context=contexte, question=question)
    reponse = llm.invoke(message)
    return reponse.content, documents[0]["metadata"]

@app.route('/')
def home():
    return "BioAssist API ✅"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    question = data.get('message', '')

    if not question.strip():
        return jsonify({'response': 'Posez une question svp', 'source': ''})

    try:
        reponse, metadata = bioassist_repondre(question)
        source = f"{metadata['source']} | {metadata['section']}"
        return jsonify({'response': reponse, 'source': source})
    except Exception as e:
        return jsonify({'response': f'Erreur: {str(e)}', 'source': ''})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

---

## Remplace aussi `requirements.txt` par ceci :
```
flask
flask-cors
langchain-core
langchain-groq
gunicorn
```

---

## Sur Render, remet le Start Command simple :
```
gunicorn app:app --bind 0.0.0.0:10000

