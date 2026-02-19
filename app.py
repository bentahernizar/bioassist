from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os

app = Flask(__name__)
CORS(app)

# Clé API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Documents
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

# Initialisation embeddings + vectordb
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_texts(
    texts=[d["content"] for d in documents],
    embedding=embeddings,
    metadatas=[d["metadata"] for d in documents]
)

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.1)

template = """
Tu es BioAssist, assistant expert en dispositifs médicaux.
CONTEXTE : {context}
QUESTION : {question}
RÉPONSE :
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def bioassist_repondre(question):
    docs = vectordb.similarity_search(question, k=1)
    contexte = docs[0].page_content
    message = prompt.format(context=contexte, question=question)
    reponse = llm.invoke(message)
    return reponse.content, docs[0].metadata

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True)
    question = data.get('message', '')
    try:
        reponse, metadata = bioassist_repondre(question)
        source = f"{metadata['source']} | {metadata['section']}"
        return jsonify({'response': reponse, 'source': source})
    except Exception as e:
        return jsonify({'response': f'Erreur: {str(e)}', 'source': ''})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
