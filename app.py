from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])
        messages = messages[-10:]

        # Convert format ke Gemini
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction="Kamu adalah asisten AI yang ramah dan membantu. Jawab dalam bahasa yang sama dengan pengguna.",
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
        reply = response.text
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.json
        text = data.get("text", "")
        target_lang = data.get("target_lang", "English")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Terjemahkan ke {target_lang}:\n\n{text}",
            config=types.GenerateContentConfig(
                system_instruction="Kamu adalah penerjemah profesional. Balas HANYA dengan terjemahannya saja, tanpa penjelasan tambahan."
            )
        )
        result = response.text
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
