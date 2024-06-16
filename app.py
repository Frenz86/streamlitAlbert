import streamlit as st
from huggingface_hub import hf_hub_download
import onnxruntime
import joblib
import numpy as np

@st.cache_resource
def load_model_and_tokenizer():
    model_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="sentiment-int8.onnx")
    tokenizer_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="tokenizer_sentiment.pkl")

    tokenizer = joblib.load(tokenizer_path) # load tokenizer
    ort_session = onnxruntime.InferenceSession(model_path) # load model quantized int8
    return tokenizer, ort_session

# Function for sentiment analysis
def analyze_sentimentinference(text, ort_session, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    probabilities = np.exp(ort_outs[0][0]) / np.exp(ort_outs[0][0]).sum(-1, keepdims=True)
    sentiment = "Positive" if probabilities[1] > probabilities[0] else "Negative"
    
    return sentiment, probabilities[1], probabilities[0]


def main():
    tokenizer, ort_session = load_model_and_tokenizer()
    st.title("Sentiment Analysis App")

    user_input = st.text_input("Enter a sentence:")
    if user_input:
        sentiment, pos_prob, neg_prob = analyze_sentimentinference(user_input, ort_session, tokenizer)
        st.write(f"Sentiment: {sentiment}")
        if sentiment == "Positive":
            st.write(f"Probability: {pos_prob:.2f}")
        else:
            st.write(f"Probability: {neg_prob:.2f}")

if __name__ == "__main__":
    main()