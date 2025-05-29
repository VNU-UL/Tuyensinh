from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os, openai, fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

def split_text(text, max_chars=1000):
    chunks, current = [], ""
    for line in text.split('\n'):
        if len(current) + len(line) < max_chars:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def get_relevant_chunks(question, docs_path="docs"):
    texts, filenames = [], []
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            try:
                path = os.path.join(docs_path, file)
                doc = fitz.open(path)
                raw = "\n".join(page.get_text() for page in doc)
                doc.close()
                chunks = split_text(raw)
                texts.extend(chunks)
                filenames.extend([file] * len(chunks))
            except Exception as e:
                print(f"❌ Lỗi đọc {file}: {e}")

    if not texts:
        return ""

    vectorizer = TfidfVectorizer().fit_transform(texts + [question])
    cosine = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    top_indices = cosine.argsort()[-3:][::-1]
    return "\n\n".join([texts[i] for i in top_indices])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        user_question = request.json.get("question", "")
        context = get_relevant_chunks(user_question)

        if not context.strip():
            return jsonify({"response": "Không tìm thấy thông tin phù hợp trong tài liệu."})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý tuyển sinh đại học và sau đại học."},
                {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi:\n{user_question}"}
            ]
        )
        answer = completion["choices"][0]["message"]["content"]
        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"response": f"Lỗi: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
