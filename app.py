# Streamlit APP 
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('next_word_lstm2.h5')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set max_sequence_len (same as during training)
max_sequence_len = 14  # Update as per your model


# Prediction Function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
    return output_word


# ---------- Streamlit UI ---------- #

st.set_page_config(page_title="Next Word Predictor", page_icon="🧠", layout="centered")

# Custom CSS for better visuals
st.markdown("""
    <style>
        .title {
            font-size:40px;
            font-weight:bold;
            text-align:center;
            color:#4B8BBE;
        }
        .subtitle {
            font-size:20px;
            text-align:center;
            color:#666;
        }
        .result {
            font-size:28px;
            font-weight:bold;
            color:#2E8B57;
            text-align:center;
            background-color:#F0F2F6;
            padding:15px;
            border-radius:10px;
            margin-top:20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔤 Next Word Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Type a sentence, and the model will predict the most likely next word.</div>', unsafe_allow_html=True)
st.markdown("""
    <div style="text-align:center; font-size:16px; color:#888;">
        📘 <em>Note:</em> This model was trained specifically on <strong>Shakespeare's Hamlet</strong>. Predictions are influenced by that text's vocabulary and style.
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("---")

user_input = st.text_area("✍️ Enter your sentence below", height=120)

if st.button("🚀 Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        next_word = predict_next_word(model, tokenizer, user_input, max_sequence_len)
        st.markdown(f'<div class="result">👉 Predicted Next Word: <span style="color:#1E90FF">{next_word}</span></div>', unsafe_allow_html=True)
