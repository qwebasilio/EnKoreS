import streamlit as st
import fasttext

# Load FastText models (replace with actual file paths to the vectors)
en_model = fasttext.load_model('cc.en.300.bin')  # Replace with your English FastText model path
kr_model = fasttext.load_model('cc.ko.300.bin')  # Replace with your Korean FastText model path

def translate_text(input_text, lang_direction):
    if lang_direction == "EN to KR":
        model = en_model
        target_language = "Korean"
    else:
        model = kr_model
        target_language = "English"
    
    # Dummy translation (replace with real logic)
    # Here, use word vectors to perform translation logic
    translation = f"Translated {input_text} to {target_language}."
    return translation

def summarize_text(text):
    # Dummy summarization (replace with real logic)
    return f"Summary of the text: {text[:50]}..."

# Streamlit App
st.title("EnKoreS: English-Korean Translator & Summarizer")
st.markdown("Translate between **English** and **Korean**, and summarize translated text.")

# Language selection
lang_direction = st.radio("Select Translation Direction:", ("EN to KR", "KR to EN"))

# Text Input
input_text = st.text_area("Enter text to translate:")

# Translation and Summarization Buttons
if st.button("Translate"):
    if input_text.strip():
        translation = translate_text(input_text, lang_direction)
        st.subheader("Translated Text")
        st.text_area("Translation Output:", translation, height=200)
    else:
        st.warning("Please enter text to translate!")

if st.button("Translate & Summarize"):
    if input_text.strip():
        translation = translate_text(input_text, lang_direction)
        summary = summarize_text(translation)
        st.subheader("Translated Text")
        st.text_area("Translation Output:", translation, height=200)
        st.subheader("Summary")
        st.text_area("Summary Output:", summary, height=200)
    else:
        st.warning("Please enter text to translate!")
