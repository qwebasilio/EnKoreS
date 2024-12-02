import streamlit as st
import fasttext
import os
import requests

# Define function to download files from Google Drive
def download_from_drive(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# File URLs from Google Drive (make sure the files are shared publicly)
en_url = "YOUR_ENGLISH_GZ_FILE_LINK"
ko_url = "YOUR_KOREAN_GZ_FILE_LINK"

# Local file paths
en_file = "cc.en.300.vec.gz"
ko_file = "cc.ko.300.vec.gz"

# Download and extract the files if they don't exist locally
if not os.path.exists(en_file):
    st.write("Downloading English vector file...")
    download_from_drive(en_url, en_file)

if not os.path.exists(ko_file):
    st.write("Downloading Korean vector file...")
    download_from_drive(ko_url, ko_file)

# Load FastText models
st.write("Loading FastText models...")
en_model = fasttext.load_model(en_file)
ko_model = fasttext.load_model(ko_file)

st.write("Models loaded successfully!")

# App UI
st.title("EnKoreS: English-Korean Translator with Summarization")

# Define translation function using FastText (word-level translation)
def translate(text, model, lang='en'):
    words = text.split()
    translated_words = []
    for word in words:
        if lang == 'en':
            # Get the closest word in the Korean model
            translated_word = ko_model.get_nearest_neighbors(word, k=1)
            if translated_word:
                translated_words.append(translated_word[0][1])  # Translate word
            else:
                translated_words.append(word)  # Keep original if no translation found
        else:
            # Get the closest word in the English model
            translated_word = en_model.get_nearest_neighbors(word, k=1)
            if translated_word:
                translated_words.append(translated_word[0][1])  # Translate word
            else:
                translated_words.append(word)  # Keep original if no translation found

    return " ".join(translated_words)

# Translation buttons
lang = 'en'  # Default language (English to Korean)
input_text = st.text_area("Enter text to translate", "")

# UI to switch between languages
col1, col2 = st.columns(2)

# Buttons to switch between translation direction
with col1:
    if st.button("Translate EN to KR"):
        lang = 'en'  # English to Korean
        st.write("Translating from English to Korean...")
        translated_text = translate(input_text, en_model, lang='en')
        st.text_area("Translated Text", translated_text, height=200)

with col2:
    if st.button("Translate KR to EN"):
        lang = 'ko'  # Korean to English
        st.write("Translating from Korean to English...")
        translated_text = translate(input_text, ko_model, lang='ko')
        st.text_area("Translated Text", translated_text, height=200)

# Summarization (placeholder for now)
if st.button("Summarize Text"):
    st.write("Summarizing the text...")
    # Replace with actual summarization logic (e.g., extract important sentences)
    summarized_text = "Summarized content here."  # Placeholder
    st.text_area("Summarized Text", summarized_text, height=150)

# Live translation (using fasttext for word-level translation)
if input_text:
    st.write("Live translation...")
    live_translation = translate(input_text, en_model, lang='en') if lang == 'en' else translate(input_text, ko_model, lang='ko')
    st.text_area("Live Translated Text", live_translation, height=150)