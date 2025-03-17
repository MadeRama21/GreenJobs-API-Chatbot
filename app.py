from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from rapidfuzz import process
from langdetect import detect
from deep_translator import GoogleTranslator

# Load Dataset
df = pd.read_csv("dataset/greenjobs.csv")

# Preprocessing & Training Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Question"])
y = df["Answer"]

model = MultinomialNB()
model.fit(X, y)  # Latih model

# Simpan model dan vectorizer
joblib.dump(model, "chatbot_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Setup Flask API
app = Flask(__name__)
CORS(app) 


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    question = data.get("question", "").strip().lower()

    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

    # Deteksi bahasa
    try:
        language = detect(question)
    except:
        language = "unknown"

    # Jika bahasa inggris, terjemahkan ke indonesia
    if language == "en":
        question = GoogleTranslator(source="en", target="id").translate(question)

    # Load model & vectorizer
    model = joblib.load("chatbot_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # Transform pertanyaan user
    question_vectorized = vectorizer.transform([question])

    # Dapatkan probabilitas prediksi
    probabilities = model.predict_proba(question_vectorized)

    # Ambil probabilitas tertinggi
    max_prob = np.max(probabilities)
    predicted_answer = model.predict(question_vectorized)[0]

    # Jika model tidak yakin, gunakan Fuzzy Matching
    if max_prob < 0.4:
        best_match, score, _ = process.extractOne(question, df["Question"])

        # Jika skor Fuzzy Matching cukup tinggi (misalnya > 70)
        if score > 70:
            matched_answer = df[df["Question"] == best_match]["Answer"].values[0]
        else:
            matched_answer = "Maaf, Aku hanya bisa jawab pertanyaan umum tentang Green Jobs & Green Economy :("
    else:
        matched_answer = predicted_answer

    # Jika pertanyaan awal dalam bahasa inggris, terjemahkan jawaban ke bahsa indonesia
    if language == "en":
        matched_answer = GoogleTranslator(source="id", target="en").translate(
            matched_answer
        )

    return jsonify({"answer": matched_answer})


if __name__ == "__main__":
    app.run(debug=True)
