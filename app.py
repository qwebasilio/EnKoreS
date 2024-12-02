import streamlit as st
import fasttext
import os
import requests

def download_from_drive(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

en_url = "https://drive.google.com/file/d/1V9lwDZOSpaPLhk6wJ0EPoY2fwY2sGHfw/view?usp=drive_link"
ko_url = "https://drive.google.com/file/d/1--87G0NQFl33ewSJ1uj53I3G7L8wPJI6/view?usp=drive_link"

en_file = "cc.en.300.vec.gz"
ko_file = "cc.ko.300.vec.gz"

if not os.path.exists(en_file):
    st.write("Downloading English vector file...")
    download_from_drive(en_url, en_file)

if not os.path.exists(ko_file):
    st.write("Downloading Korean vector file...")
    download_from_drive(ko_url, ko_file)

st.write("Loading FastText models...")
en_model = fasttext.load_model(en_file)
ko_model = fasttext.load_model(ko_file)

st.write("Models loaded successfully!")
st.title("EnKoreS")

def translate(text, model, lang='en'):
    words = text.split()
    translated_words = []
    for word in words:
        if lang == 'en':
            translated_word = ko_model.get_nearest_neighbors(word, k=1)
            if translated_word:
                translated_words.append(translated_word[0][1])
            else:
                translated_words.append(word)
        else:
            translated_word = en_model.get_nearest_neighbors(word, k=1)
            if translated_word:
                translated_words.append(translated_word[0][1])
            else:
                translated_words.append(word)

    return " ".join(translated_words)

lang = 'en'
input_text = st.text_area("Start Typing", "")
col1, col2 = st.columns(2)
with col1:
    if st.button("Translate EN to KR"):
        lang = 'en' 
        st.write("Translating from English to Korean...")
        translated_text = translate(input_text, en_model, lang='en')
        st.text_area("Translated Text", translated_text, height=200)

with col2:
    if st.button("Translate KR to EN"):
        lang = 'ko'
        st.write("Translating from Korean to English...")
        translated_text = translate(input_text, ko_model, lang='ko')
        st.text_area("Translated Text", translated_text, height=200)

# Summarization placeholder
if st.button("Summarize Text"):
    st.write("Summarizing the text...")
    summarized_text = "Summarized content here."
    st.text_area("Summarized Text", summarized_text, height=150)

if input_text:
    st.write("...")
    live_translation = translate(input_text, en_model, lang='en') if lang == 'en' else translate(input_text, ko_model, lang='ko')
    st.text_area("Live Translated Text", live_translation, height=150)
