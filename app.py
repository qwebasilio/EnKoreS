import streamlit as st
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel
import torch

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTModel.from_pretrained("openai-gpt")

def translate_text_gpt(input_text, src_lang, tgt_lang):
    prompt = f"Translate this text from {src_lang} to {tgt_lang}: {input_text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    translated_text = tokenizer.decode(last_hidden_states[0], skip_special_tokens=True)
    return translated_text

st.title("Translation and Summarization with GPT")

lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

input_text = st.text_area("Enter text to translate:")

if input_text:
    if lang_direction == "EN to KO":
        translated_text = translate_text_gpt(input_text, "English", "Korean")
    elif lang_direction == "KO to EN":
        translated_text = translate_text_gpt(input_text, "Korean", "English")
    
    st.write("Translated Text:")
    st.write(translated_text)

    if translated_text:
        summarize_button = st.button("Summarize")
        if summarize_button:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            summary = summarizer(translated_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.write("Summarized Text:")
            st.write(summary)