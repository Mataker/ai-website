from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import os
import sqlite3
import uuid
import base64
import requests
from datetime import datetime
from dotenv import load_dotenv
from pypdf import PdfReader
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    api_key=os.environ.get("ZHIPU_API_KEY"),
    base_url="https://api.z.ai/api/paas/v4/"
)

DB_PATH = os.environ.get("DB_PATH", "hila.db")
UPLOAD_DIR = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp"}
ALLOWED_PDF_EXT = {"pdf"}
MAX_FILE_SIZE = 8 * 1024 * 1024

SYSTEM_PROMPT_TEMPLATE = """Kamu adalah HILA, asisten AI yang ramah, hangat, dan membantu.

ATURAN PENTING TENTANG IDENTITAS:
- Nama kamu adalah HILA. Selalu sebut dirimu sebagai HILA.
- JANGAN PERNAH menyebutkan bahwa kamu adalah model AI lain (seperti GLM, Zhipu, ChatGPT, Claude, Gemini, dll), bahkan jika ditanya langsung.
- Jika ditanya "kamu pakai model apa" atau "siapa yang membuatmu", jawab bahwa kamu adalah HILA, asisten AI yang dirancang untuk membantu penggunamu.
- Jangan pernah mengoreksi pengguna jika mereka memanggilmu HILA atau menyebut HILA sebagai model AI tertentu.

PERSONALISASI:
- Nama pengguna yang sedang chat denganmu adalah: {user_name}
- Panggil pengguna dengan nama tersebut secara natural dalam percakapan, tidak perlu di setiap kalimat, tapi sesekali agar terasa personal.

KEMAMPUAN:
- Kamu bisa melihat dan menganalisa gambar yang diupload pengguna.
- Kamu bisa membaca isi dokumen PDF yang diupload pengguna (akan diberikan dalam bentuk teks yang sudah diekstrak).
- Kamu bisa membuatkan gambar jika pengguna minta (mereka akan melihat tombol/fitur generate gambar di aplikasi).

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
            attachment_url TEXT,
            attachment_type TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def call_hila(history, user_name, use_web_search=False, has_image=False):
    template = SYSTEM_PROMPT_WEB_SEARCH if use_web_search else SYSTEM_PROMPT_TEMPLATE
    system_prompt = template.format(user_name=user_name)
    model = "glm-4.6v-flash" if has_image else "glm-4.7-flash"

    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + history
    }
    if use_web_search and not has_image:
        kwargs["tools"] = [{"type": "web_search", "web_search": {"search_result": True}}]

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


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
        "SELECT id, role, content, attachment_url, attachment_type FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    messages = [{
        "id": r["id"], "role": r["role"], "content": r["content"],
        "attachment_url": r["attachment_url"], "attachment_type": r["attachment_type"]
    } for r in rows]
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


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """Upload gambar atau PDF, kembalikan info file untuk dipakai saat chat."""
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File terlalu besar (maksimal 8MB)"}), 400

    unique_name = f"{uuid.uuid4().hex}_{filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    if ext in ALLOWED_IMAGE_EXT:
        file.save(save_path)
        return jsonify({
            "type": "image",
            "url": f"/static/uploads/{unique_name}",
            "filename": filename
        })
    elif ext in ALLOWED_PDF_EXT:
        file.save(save_path)
        try:
            reader = PdfReader(save_path)
            text_parts = []
            for page in reader.pages[:30]:
                text_parts.append(page.extract_text() or "")
            extracted_text = "\n".join(text_parts).strip()
            if not extracted_text:
                extracted_text = "(Tidak ada teks yang bisa diekstrak dari PDF ini, mungkin berupa hasil scan gambar.)"
            extracted_text = extracted_text[:15000]
        except Exception as e:
            extracted_text = f"(Gagal membaca PDF: {str(e)})"
        return jsonify({
            "type": "pdf",
            "url": f"/static/uploads/{unique_name}",
            "filename": filename,
            "extracted_text": extracted_text
        })
    else:
        return jsonify({"error": "Tipe file tidak didukung. Gunakan gambar (jpg/png/webp/gif) atau PDF."}), 400


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        user_message = data.get("message", "")
        use_web_search = data.get("web_search", False)
        attachment_url = data.get("attachment_url")
        attachment_type = data.get("attachment_type")
        pdf_text = data.get("pdf_text")

        if not session_id or (not user_message and not attachment_url):
            return jsonify({"reply": "Error: session_id dan message wajib diisi"}), 400

        conn = get_db()
        session_row = conn.execute(
            "SELECT user_name, title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session_row:
            conn.close()
            return jsonify({"reply": "Error: sesi tidak ditemukan"}), 404

        user_name = session_row["user_name"]

        stored_content = user_message if user_message else (
            "[Gambar]" if attachment_type == "image" else "[Dokumen PDF]"
        )
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, attachment_url, attachment_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "user", stored_content, attachment_url, attachment_type, datetime.now().isoformat())
        )
        user_msg_id = cur.lastrowid
        conn.commit()

        history_rows = conn.execute(
            "SELECT role, content, attachment_url, attachment_type FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        recent_rows = history_rows[-12:]

        history = []
        for r in recent_rows:
            if r["role"] == "user" and r["attachment_type"] == "image" and r["attachment_url"]:
                image_path = os.path.join(app.root_path, r["attachment_url"].lstrip("/"))
                try:
                    with open(image_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                    ext = r["attachment_url"].rsplit(".", 1)[-1].lower()
                    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                    history.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": r["content"] if r["content"] != "[Gambar]" else "Tolong lihat dan jelaskan gambar ini."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
                        ]
                    })
                except Exception:
                    history.append({"role": "user", "content": r["content"]})
            else:
                history.append({"role": r["role"], "content": r["content"]})

        if pdf_text:
            last_user_idx = len(history) - 1
            if last_user_idx >= 0 and isinstance(history[last_user_idx]["content"], str):
                history[last_user_idx]["content"] = (
                    f"{user_message}\n\n[Isi dokumen PDF yang diupload]:\n{pdf_text}"
                )

        has_image = any(
            isinstance(h.get("content"), list) for h in history
        )

        reply = call_hila(history, user_name, use_web_search, has_image)

        cur2 = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "assistant", reply, datetime.now().isoformat())
        )
        ai_msg_id = cur2.lastrowid

        if session_row["title"] == "Chat baru":
            title_source = user_message or "Percakapan dengan lampiran"
            new_title = title_source[:40] + ("..." if len(title_source) > 40 else "")
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))

        conn.commit()
        conn.close()

        return jsonify({"reply": reply, "user_msg_id": user_msg_id, "ai_msg_id": ai_msg_id})
    except Exception as e:
        return jsonify({"reply": f"Maaf, terjadi kendala teknis. Coba lagi ya! ({str(e)})"}), 500


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
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


@app.route("/api/generate-image", methods=["POST"])
def generate_image():
    """Generate gambar AI lewat Pollinations.ai (gratis, tanpa API key)."""
    try:
        data = request.json
        session_id = data.get("session_id")
        prompt = data.get("prompt", "").strip()
        if not prompt or not session_id:
            return jsonify({"error": "Prompt dan session_id wajib diisi"}), 400

        seed = uuid.uuid4().int % 1000000
        image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=768&height=768&seed={seed}&nologo=true"

        resp = requests.get(image_url, timeout=60)
        if resp.status_code != 200:
            return jsonify({"error": "Gagal generate gambar, coba lagi ya"}), 500

        unique_name = f"{uuid.uuid4().hex}_generated.jpg"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(save_path, "wb") as f:
            f.write(resp.content)

        local_url = f"/static/uploads/{unique_name}"

        conn = get_db()
        session_row = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, attachment_url, attachment_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "user", f"Buatkan gambar: {prompt}", None, None, datetime.now().isoformat())
        )
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, attachment_url, attachment_type, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "assistant", f"Ini gambar yang HILA buatkan untuk: *{prompt}*", local_url, "generated_image", datetime.now().isoformat())
        )
        ai_msg_id = cur.lastrowid
        if session_row and session_row["title"] == "Chat baru":
            new_title = f"Gambar: {prompt[:30]}"
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (new_title, session_id))
        conn.commit()
        conn.close()

        return jsonify({"image_url": local_url, "ai_msg_id": ai_msg_id})
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500


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
