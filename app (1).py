import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./sentiment_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

labels = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😊"
}

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return labels[pred]

st.title("Sentiment Analysis App")
st.write("Enter text and predict sentiment.")

user_text = st.text_area("Your Text:")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter text.")
    else:
        result = predict_sentiment(user_text)
        st.success(f"Prediction: {result}")
