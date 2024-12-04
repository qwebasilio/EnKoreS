import streamlit as st
import nltk
import os
import pandas as pd
from googletrans import Translator
from transformers import pipeline

summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

translator = Translator()

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

VALID_LANG_CODES = ['ko', 'en']

def translate_text(text, src_lang, tgt_lang):
    translated = translator.translate(text, src=src_lang, dest=tgt_lang)
    return translated.text

def summarize_text(text):
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return ""

def get_translation(input_text, src_lang, tgt_lang):
    if not input_text.strip():
        return ""
    translated_text = translate_text(input_text, src_lang, tgt_lang)
    return translated_text

st.title("EnKoreS")

st.sidebar.title("Settings")
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KR", "KR to EN"])
st.session_state.lang_direction = lang_direction

input_text = st.text_area("Enter text to translate:")
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data from file: {e}")
        return pd.DataFrame(columns=['question2_en', 'question2_ko'])

# Replace the file path with your local file path
data = load_data('path_to_your_local_file.csv')

if lang_direction == "EN to KR":
    source_col = "question2_en"
    target_col = "question2_ko"
else:
    source_col = "question2_ko"
    target_col = "question2_en"

# Buttons for translation and summarization
translate_button = st.button("Translate")
summarize_button = st.button("Translate and Summarize")

# If Translate button is clicked
if translate_button:
    if input_text:
        st.session_state.output_text = get_translation(input_text, 'EN', 'KO' if lang_direction == "EN to KR" else 'EN')
        st.subheader("Translated Text:")
        st.write(st.session_state.output_text)
    else:
        st.warning("Please enter text to translate.")

# If Translate and Summarize button is clicked
if summarize_button:
    if input_text:
        # Perform translation first
        translated_text = get_translation(input_text, 'EN', 'KO' if lang_direction == "EN to KR" else 'EN')
        st.subheader("Translated Text:")
        st.write(translated_text)

        # Perform summarization
        summarized_text = summarize_text(translated_text)
        st.subheader("Summarized Text:")
        st.write(summarized_text)
    else:
        st.warning("Please enter text to translate and summarize.")

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)
