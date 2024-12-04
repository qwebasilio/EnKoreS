import streamlit as st
import nltk
import os
from easynmt import EasyNMT
import pandas as pd
from transformers import pipeline

# Initialize Hugging Face summarizer
summarizer = pipeline('summarization', model='facebook/m2m_100_418M')

# Initialize the EasyNMT model (m2m_100_418M)
model = EasyNMT('m2m_100_418M')

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
    if src_lang == "en" and tgt_lang == "ko":
        translated_text = model.translate(text, source_lang="en", target_lang="ko")
    elif src_lang == "ko" and tgt_lang == "en":
        translated_text = model.translate(text, source_lang="ko", target_lang="en")
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} to {tgt_lang}")
    
    return translated_text

def summarize_text(text):
    # Summarize the translated text
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def get_translation(input_text, data, source_column, target_column):
    if not input_text.strip():
        return ""
    if source_column not in data.columns or target_column not in data.columns:
        st.warning(f"The provided data does not contain the columns: {source_column} and {target_column}. Using live translation.")
        return translate_text(input_text, source_column.split("_")[1], target_column.split("_")[1])
    existing_translation = data[data[source_column] == input_text]
    if not existing_translation.empty:
        translated_text = existing_translation[target_column].iloc[0]
    else:
        src_lang = source_column.split("_")[1] 
        tgt_lang = target_column.split("_")[1]
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
def load_data():
    url = 'https://raw.githubusercontent.com/qwebasilio/EnKoreS/refs/heads/master/sample_dataset.csv'
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}")
        return pd.DataFrame(columns=['question2_en', 'question2_ko'])

data = load_data()

if lang_direction == "EN to KR":
    source_col = "question2_en"
    target_col = "question2_ko"
else:
    source_col = "question2_ko"
    target_col = "question2_en"

# Buttons for translation and summarization
translate_button = st.button("Translate")
summarize_button = st.button("Translate and Summarize")

if input_text != st.session_state.input_text:
    st.session_state.input_text = input_text
    st.session_state.output_text = get_translation(input_text, data, source_col, target_col)

# If Translate button is clicked
if translate_button:
    if input_text:
        # Perform translation
        st.session_state.output_text = get_translation(input_text, data, source_col, target_col)
        st.subheader("Translated Text:")
        st.write(st.session_state.output_text)
    else:
        st.warning("Please enter text to translate.")

# If Translate and Summarize button is clicked
if summarize_button:
    if input_text:
        # Perform translation first
        translated_text = get_translation(input_text, data, source_col, target_col)
        st.subheader("Translated Text:")
        st.write(translated_text)

        # Perform summarization
        summarized_text = summarize_text(translated_text)
        st.subheader("Summarized Text:")
        st.write(summarized_text)
    else:
        st.warning("Please enter text to translate and summarize.")

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)
