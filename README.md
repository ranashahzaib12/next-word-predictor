# 🔮 Next Word Prediction App (Shakespeare's Hamlet)

This is a Streamlit web app that predicts the **next word** in a sentence, trained specifically on the text of *Shakespeare's Hamlet* using a Recurrent Neural Network (RNN) with LSTM layers.

---

## 📌 Project Highlights

- 📘 Trained on *Hamlet* by William Shakespeare.
- 💡 Uses TensorFlow LSTM model for language modeling.
- 🧠 Predicts the most likely next word based on your input.
- 🖥️ Clean and interactive UI built with Streamlit.

---
.
├── app.py               # Streamlit app interface
├── model/
│   ├── next_word_model.h5   # Trained LSTM model
│   └── tokenizer.pkl        # Tokenizer used for preprocessing
├── requirements.txt
└── README.md


## 🚀 Live Demo

> 📍 You can run this app locally using the instructions below.

---

## 🧰 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: TensorFlow, Keras, NumPy
- **Model**: 2-layer LSTM trained on Hamlet text
- **Language**: Python

---

## 🛠️ Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/ranashahzaib12/next-word-predictor.git
cd next-word-predictor
pip install -r requirements.txt
"# next-word-predictor" 
