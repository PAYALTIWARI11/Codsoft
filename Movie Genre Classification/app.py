import streamlit as st
import pickle
import os

# === File Paths ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "genre_classifier.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

# === Load Model and Artifacts ===
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === Streamlit App ===
st.set_page_config(page_title="Movie Genre Classifier 🎬", layout="centered")

st.title("🎞️ Movie Genre Classification App")
st.markdown("Predict the **genre** of a movie based on its plot summary.")

# === Input Box ===
plot_input = st.text_area("Enter movie plot/story here 👇", height=200)

# === Prediction Button ===
if st.button("Predict Genre"):
    if not plot_input.strip():
        st.warning("⚠️ Please enter a movie plot.")
    else:
        # Vectorize input and predict
        X_input = tfidf_vectorizer.transform([plot_input])
        prediction = model.predict(X_input)
        predicted_genre = label_encoder.inverse_transform(prediction)[0]

        st.success(f"🎬 Predicted Genre: **{predicted_genre.upper()}**")

# === Footer ===
st.markdown("---")
st.markdown("Built with ❤️ by Payal Tiwari | Codsoft Internship Project")
