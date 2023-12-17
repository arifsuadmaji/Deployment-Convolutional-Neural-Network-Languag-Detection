from flask import Flask, request, jsonify, render_template
import pickle, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load Model
model = load_model("projek_deteksi_bahasa.h5")

# Load Tokenizer
# Pastikan Tokenizer sama dengan yang digunakan saat melatih model
# Jika Anda menyimpan tokenizer, Anda dapat memuatnya jugaa
with open("tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open("le.pkl", "rb") as le_file:
    le = pickle.load(le_file)

# Maksimal panjang sequence yang diharapkan
max_len =  331

# Fungsi untuk preprocessing teks
def preprocess_text(text):
    text = [text.lower()]
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded_sequences

# Fungsi untuk mendeteksi bahasa dari teks
def detect_language(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Ubah kembali ke label asli menggunakan inverse_transform
    language = le.inverse_transform([predicted_class])[0]
    return language

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    language = detect_language(text)
    return render_template('index.html', language=language, text=text)

if __name__ == '__main__':
    app.run(debug=True)
