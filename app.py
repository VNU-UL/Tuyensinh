from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Khởi tạo OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

def read_all_pdfs(folder_path="docs"):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(folder_path, filename)
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
            except Exception as e:
                print(f"Lỗi đọc {filename}: {e}")
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        user_question = data.get("question", "")
        context = read_all_pdfs()

        response = client.chat.completions.create(
            model="gpt-4",  # Hoặc gpt-3.5-turbo nếu chưa cần GPT-4
            messages=[
                {"role": "system", "content": "Bạn là trợ lý tuyển sinh của Trường Đại học Luật, ĐHQGHN. Trả lời mạch lạc, chính xác và ngắn gọn dựa trên nội dung tài liệu sau."},
                {"role": "user", "content": f"Tài liệu:\n{context}\n\nCâu hỏi:\n{user_question}"}
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"response": answer})

    except Exception as e:
        print("LỖI GỌI GPT:", str(e))
        return jsonify({"response": f"Lỗi: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
