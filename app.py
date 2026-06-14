from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    messages = data.get("messages", [])

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "system", "content": "Kamu adalah asisten AI yang ramah dan membantu. Jawab dalam bahasa yang sama dengan pengguna."}] + messages
    )
    return jsonify({"reply": response.choices[0].message.content})

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    target_lang = data.get("target_lang", "English")

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "Kamu adalah penerjemah profesional. Terjemahkan teks yang diberikan ke bahasa target. Balas HANYA dengan terjemahannya saja, tanpa penjelasan tambahan."},
            {"role": "user", "content": f"Terjemahkan ke {target_lang}:\n\n{text}"}
        ]
    )
    return jsonify({"result": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
