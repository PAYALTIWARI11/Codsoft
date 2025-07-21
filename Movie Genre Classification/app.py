import streamlit as st
import pickle
import re

# Load saved model, vectorizer, and label encoder
with open('genre_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Text preprocessing
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower()

# Streamlit interface
st.set_page_config(page_title="Movie Genre Classifier", layout="centered")
st.title("🎬 Movie Genre Classification")
st.write("Enter the movie plot below and get the predicted genre:")

plot_input = st.text_area("📜 Movie Plot", height=200)

if st.button("Predict Genre"):
    if plot_input.strip() == "":
        st.warning("Please enter a movie plot.")
    else:
        clean_plot = clean_text(plot_input)
        vectorized_input = vectorizer.transform([clean_plot])
        prediction = model.predict(vectorized_input)
        genre = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🎯 Predicted Genre: **{genre.upper()}**")

import pickle

with open("genre_classifier.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")
