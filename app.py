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
    try:
        data = request.json
        messages = data.get("messages", [])
        response = client.chat.completions.create(
            model="compound-beta",
            messages=[{"role": "system", "content": "Kamu adalah asisten AI yang ramah dan membantu. Jawab dalam bahasa yang sama dengan pengguna."}] + messages
        )
        # compound-beta kadang reply ada di content, kadang di tool
        reply = ""
        for block in response.choices[0].message.content if isinstance(response.choices[0].message.content, list) else [response.choices[0].message]:
            if hasattr(block, 'text'):
                reply += block.text
            elif hasattr(block, 'content') and isinstance(block.content, str):
                reply += block.content
            elif isinstance(block, str):
                reply += block
        if not reply:
            reply = str(response.choices[0].message.content)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        text = data.get("text", "")
        target_lang = data.get("target_lang", "English")
        response = client.chat.completions.create(
            model="compound-beta",
            messages=[
                {"role": "system", "content": "Kamu adalah penerjemah profesional. Terjemahkan teks yang diberikan ke bahasa target. Balas HANYA dengan terjemahannya saja, tanpa penjelasan tambahan."},
                {"role": "user", "content": f"Terjemahkan ke {target_lang}:\n\n{text}"}
            ]
        )
        result = str(response.choices[0].message.content)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)
