import streamlit as st
import nltk
import os
from easynmt import EasyNMT
import pandas as pd

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

model = EasyNMT('opus-mt')

VALID_LANG_CODES = ['en', 'ko']

def translate_text(text, src_lang, tgt_lang):
    if src_lang not in VALID_LANG_CODES or tgt_lang not in VALID_LANG_CODES:
        raise ValueError(f"Invalid language codes: src_lang={src_lang}, tgt_lang={tgt_lang}")
    translated_text = model.translate(text, source_lang=src_lang, target_lang=tgt_lang)
    return translated_text

def get_translation(input_text, data, lang_column="question2_ko"):
    if not input_text.strip():
        return ""
    existing_translation = data[data['question1'] == input_text]
    if not existing_translation.empty:
        translated_text = existing_translation[lang_column].iloc[0]
        st.write(f"Found existing translation: {translated_text}")
    else:
        translated_text = translate_text(input_text, "en", "ko")
        st.write(f"Generated new translation: {translated_text}")
    return translated_text

st.title("Translation App")

st.sidebar.title("Settings")
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KR", "KR to EN"])
st.session_state.lang_direction = lang_direction

input_text = st.text_area("Enter text to translate:")
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if input_text != st.session_state.input_text:
    st.session_state.input_text = input_text
    if st.session_state.lang_direction == "EN to KR":
        st.session_state.output_text = get_translation(input_text, pd.DataFrame(), "question2_ko")
    else:
        st.session_state.output_text = translate_text(input_text, "ko", "en")

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)

