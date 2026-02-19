from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import threading

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Variables globales
vectordb = None
llm = None
prompt = None

def initialiser():
    """Initialisation en arrière-plan pour ne pas bloquer le démarrage"""
    global vectordb, llm, prompt

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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_texts(
        texts=[d["content"] for d in documents],
        embedding=embeddings,
        metadatas=[d["metadata"] for d in documents]
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.1)

    template = """
Tu es BioAssist, assistant expert en dispositifs médicaux.
CONTEXTE : {context}
QUESTION : {question}
RÉPONSE :
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    print("✅ Initialisation terminée !")

# Lancer initialisation en arrière-plan
threading.Thread(target=initialiser, daemon=True).start()

@app.route('/')
def home():
    return "BioAssist API ✅"

@app.route('/chat', methods=['POST'])
def chat():
    if vectordb is None:
        return jsonify({'response': '⏳ Système en cours de démarrage, réessayez dans 30 secondes...', 'source': ''})

    data = request.get_json(force=True)
    question = data.get('message', '')

    try:
        docs = vectordb.similarity_search(question, k=1)
        contexte = docs[0].page_content
        message = prompt.format(context=contexte, question=question)
        reponse = llm.invoke(message)
        source = f"{docs[0].metadata['source']} | {docs[0].metadata['section']}"
        return jsonify({'response': reponse.content, 'source': source})
    except Exception as e:
        return jsonify({'response': f'Erreur: {str(e)}', 'source': ''})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

