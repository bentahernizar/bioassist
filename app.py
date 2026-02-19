from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Tes documents techniques
DOCUMENTS = """
FICHE DÉPANNAGE MSV-1000 - Écran noir
SYMPTÔME : Écran totalement noir, pas de lumière, aucun bip au démarrage.
CAUSE FRÉQUENTE : Batterie complètement déchargée.
SOLUTION : Brancher sur secteur 4 heures minimum.
PIÈCE : Batterie lithium-ion 14.4V, référence BAT-MSV1000.
Code erreur associé : ERR 01
"""

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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es BioAssist, assistant expert en dispositifs médicaux. Voici la documentation : {DOCUMENTS}"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.1
        )

        reponse = response.choices[0].message.content
        return jsonify({
            'response': reponse,
            'source': 'Manuel MSV-1000 v1.0 | Dépannage Écran Noir'
        })

    except Exception as e:
        return jsonify({'response': f'Erreur: {str(e)}', 'source': ''})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

---

## Remplace `requirements.txt` par :
```
flask
flask-cors
groq
gunicorn
```

---

## Sur Render → Settings, Start Command :
```
gunicorn app:app --bind 0.0.0.0:10000 --timeout 120 --workers 1


