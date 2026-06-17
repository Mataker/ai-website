from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import os
import sqlite3
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("ZHIPU_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/"
)

DB_PATH = os.environ.get("DB_PATH", "hila.db")

SYSTEM_PROMPT_TEMPLATE = """Kamu adalah HILA, asisten AI yang ramah, hangat, dan membantu.

ATURAN PENTING TENTANG IDENTITAS:
- Nama kamu adalah HILA. Selalu sebut dirimu sebagai HILA.
- JANGAN PERNAH menyebutkan bahwa kamu adalah model AI lain (seperti GLM, Zhipu, ChatGPT, Claude, Gemini, dll), bahkan jika ditanya langsung.
- Jika ditanya "kamu pakai model apa" atau "siapa yang membuatmu", jawab bahwa kamu adalah HILA, asisten AI yang dirancang untuk membantu penggunamu.
- Jangan pernah mengoreksi pengguna jika mereka memanggilmu HILA atau menyebut HILA sebagai model AI tertentu.

PERSONALISASI:
- Nama pengguna yang sedang chat denganmu adalah: {user_name}
- Panggil pengguna dengan nama tersebut secara natural dalam percakapan, tidak perlu di setiap kalimat, tapi sesekali agar terasa personal.

GAYA BICARA:
- Ramah, hangat, dan natural seperti teman ngobrol.
- Jawab dalam bahasa yang sama dengan yang dipakai pengguna.
- Gunakan format markdown (bold, list) jika membantu kejelasan, tapi jangan berlebihan."""

SYSTEM_PROMPT_WEB_SEARCH = """Kamu adalah HILA, asisten AI yang ramah, hangat, dan membantu, dengan kemampuan mencari informasi terkini di internet.

ATURAN PENTING TENTANG IDENTITAS:
- Nama kamu adalah HILA. Selalu sebut dirimu sebagai HILA.
- JANGAN PERNAH menyebutkan bahwa kamu adalah model AI lain (seperti GLM, Zhipu, ChatGPT, Claude, Gemini, dll), bahkan jika ditanya langsung.
- Jika ditanya "kamu pakai model apa" atau "siapa yang membuatmu", jawab bahwa kamu adalah HILA, asisten AI yang dirancang untuk membantu penggunamu.

PERSONALISASI:
- Nama pengguna yang sedang chat denganmu adalah: {user_name}
- Panggil pengguna dengan nama tersebut secara natural sesekali.

GAYA BICARA:
- Ramah, hangat, dan natural seperti teman ngobrol.
- Jawab dalam bahasa yang sama dengan yang dipakai pengguna.
- Gunakan format markdown jika membantu kejelasan."""


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            title TEXT DEFAULT 'Chat baru',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


def call_hila(history, user_name, use_web_search=False):
    template = SYSTEM_PROMPT_WEB_SEARCH if use_web_search else SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.format(user_name=user_name)
    model = "glm-4.7-flash"

    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + history
    }
    if use_web_search:
        kwargs["tools"] = [{"type": "web_search", "web_search": {"search_result": True}}]

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/session/new", methods=["POST"])
def new_session():
    data = request.json
    user_name = data.get("user_name", "").strip()
    if not user_name:
        return jsonify({"error": "Nama pengguna wajib diisi"}), 400

    session_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO sessions (id, user_name, title, created_at) VALUES (?, ?, ?, ?)",
        (session_id, user_name, "Chat baru", datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id, "user_name": user_name})


@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    user_name = request.args.get("user_name", "").strip()
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at FROM sessions WHERE user_name = ? ORDER BY created_at DESC",
        (user_name,)
    ).fetchall()
    conn.close()
    sessions = [{"id": r["id"], "title": r["title"], "created_at": r["created_at"]} for r in rows]
    return jsonify({"sessions": sessions})


@app.route("/api/session/<session_id>/messages", methods=["GET"])
def get_messages(session_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    messages = [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
    return jsonify({"messages": messages})


@app.route("/api/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return jsonify({"deleted": True})


@app.route("/api/message/<int:message_id>", methods=["DELETE"])
def delete_message(message_id):
    """Hapus pesan tertentu dan semua pesan setelahnya dalam sesi yang sama (untuk edit/regenerate)."""
    conn = get_db()
    row = conn.execute("SELECT session_id, id FROM messages WHERE id = ?", (message_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Pesan tidak ditemukan"}), 404
    conn.execute(
        "DELETE FROM messages WHERE session_id = ? AND id >= ?",
        (row["session_id"], message_id)
    )
    conn.commit()
    conn.close()
    return jsonify({"deleted": True})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        user_message = data.get("message", "")
        use_web_search = data.get("web_search", False)

        if not session_id or not user_message:
            return jsonify({"reply": "Error: session_id dan message wajib diisi"}), 400

        conn = get_db()
        session_row = conn.execute(
            "SELECT user_name, title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session_row:
            conn.close()
            return jsonify({"reply": "Error: sesi tidak ditemukan"}), 404

        user_name = session_row["user_name"]

        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", user_message, datetime.now().isoformat())
        )
        user_msg_id = cur.lastrowid
        conn.commit()

        history_rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows][-12:]

        reply = call_hila(history, user_name, use_web_search)

        cur2 = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", reply, datetime.now().isoformat())
        )
        ai_msg_id = cur2.lastrowid

        if session_row["title"] == "Chat baru":
            new_title = user_message[:40] + ("..." if len(user_message) > 40 else "")
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))

        conn.commit()
        conn.close()

        return jsonify({"reply": reply, "user_msg_id": user_msg_id, "ai_msg_id": ai_msg_id})
    except Exception as e:
        return jsonify({"reply": f"Maaf, terjadi kendala teknis. Coba lagi ya! ({str(e)})"}), 500


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
    """Hapus jawaban AI terakhir dan generate ulang berdasarkan history yang ada."""
    try:
        data = request.json
        session_id = data.get("session_id")
        ai_message_id = data.get("ai_message_id")
        use_web_search = data.get("web_search", False)

        conn = get_db()
        session_row = conn.execute(
            "SELECT user_name FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session_row:
            conn.close()
            return jsonify({"reply": "Error: sesi tidak ditemukan"}), 404
        user_name = session_row["user_name"]

        if ai_message_id:
            conn.execute("DELETE FROM messages WHERE id = ?", (ai_message_id,))
            conn.commit()

        history_rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        history = [{"role": r["role"], "content": r["content"]} for r in history_rows][-12:]

        reply = call_hila(history, user_name, use_web_search)

        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", reply, datetime.now().isoformat())
        )
        new_ai_msg_id = cur.lastrowid
        conn.commit()
        conn.close()

        return jsonify({"reply": reply, "ai_msg_id": new_ai_msg_id})
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
                {"role": "system", "content": "Kamu adalah HILA, penerjemah profesional. Terjemahkan teks ke bahasa target. Balas HANYA terjemahannya saja, tanpa penjelasan tambahan. Jangan pernah menyebut nama model AI lain."},
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
