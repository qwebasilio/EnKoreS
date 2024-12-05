import streamlit as st
import nltk
import os
from easynmt import EasyNMT

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")):
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt")):
    nltk.download('punkt', download_dir=nltk_data_dir)

model = EasyNMT('m2m_100_418M')

VALID_LANG_CODES = ['ko', 'en']

def translate_text(text, src_lang, tgt_lang):
    if src_lang == "en" and tgt_lang == "ko":
        translated_text = model.translate(text, source_lang="en", target_lang="ko")
    elif src_lang == "ko" and tgt_lang == "en":
        translated_text = model.translate(text, source_lang="ko", target_lang="en")
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} to {tgt_lang}")
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

if input_text != st.session_state.input_text:
    st.session_state.input_text = input_text
    st.session_state.output_text = translate_text(input_text, "en", "ko" if lang_direction == "EN to KR" else "en")

translate_button = st.button("Translate")

if translate_button:
    if input_text:
        st.session_state.output_text = translate_text(input_text, "en", "ko" if lang_direction == "EN to KR" else "en")
        st.subheader("Translated Text:")
        st.write(st.session_state.output_text)
    else:
        st.warning("Please enter text to translate.")

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)