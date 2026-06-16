from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("ZHIPU_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/"
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])
        messages = messages[-10:]
        response = client.chat.completions.create(
            model="glm-4.7-flash",
            messages=[{"role": "system", "content": "Kamu adalah asisten AI yang ramah dan membantu. Jawab dalam bahasa yang sama dengan pengguna."}] + messages
        )
        reply = response.choices[0].message.content
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
            model="glm-4.7-flash",
            messages=[
                {"role": "system", "content": "Kamu adalah penerjemah profesional. Terjemahkan teks ke bahasa target. Balas HANYA terjemahannya saja."},
                {"role": "user", "content": f"Terjemahkan ke {target_lang}:\n\n{text}"}
            ]
        )
        result = response.choices[0].message.content
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
