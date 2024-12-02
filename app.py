import streamlit as st
import fasttext
import os
import requests
import gzip
import io

def create_drive_download_link(google_drive_url):
    file_id = google_drive_url.split('/d/')[1].split('/')[0]
    return f"https://drive.google.com/uc?id={file_id}"

def download_from_drive(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def load_model_from_gz(filename):
    with gzip.open(filename, 'rb') as f:
        model_bytes = f.read()
    model_file = io.BytesIO(model_bytes)
    return fasttext.load_model(model_file)

en_url = "https://drive.google.com/file/d/14zlXmW3iUgx39jU6uwDIbcYLe4QK2FFi/view?usp=drive_link"
ko_url = "https://drive.google.com/file/d/1L4sjC9DjBqNlaCf6MDpXfHLbyjN9T_na/view?usp=drive_link"

en_download_url = create_drive_download_link(en_url)
ko_download_url = create_drive_download_link(ko_url)

en_file = "cc.en.50.vec"
ko_file = "cc.ko.50.vec"

if not os.path.exists(en_file):
    st.write("Downloading English vector file...")
    download_from_drive(en_download_url, en_file)

if not os.path.exists(ko_file):
    st.write("Downloading Korean vector file...")
    download_from_drive(ko_download_url, ko_file)

print(f"Current working directory: {os.getcwd()}")

st.write("Loading FastText models...")
en_model = fasttext.load_model(en_file)
ko_model = fasttext.load_model(ko_file)
st.write("Models loaded successfully!")

st.title("EnKoreS: English-Korean Translator with Summarization")

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"

def switch_languages():
    if st.session_state.lang_direction == "EN to KR":
        st.session_state.lang_direction = "KR to EN"
    else:
        st.session_state.lang_direction = "EN to KR"

col1, col2 = st.columns(2)

with col1:
    st.header("English" if st.session_state.lang_direction == "EN to KR" else "Korean")
    input_text = st.text_area("Input Text", key="input_text", height=200)

with col2:
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English")
    output_text = st.text_area("Translated Text", value="", height=200, disabled=True, key="output_text")

st.button("Switch Languages", on_click=switch_languages)

def translate_text(text, lang_direction):
    words = text.split()
    if lang_direction == "EN to KR":
        translated = " ".join([ko_model.get_word_vector(word).tolist() for word in words])
    else:
        translated = " ".join([en_model.get_word_vector(word).tolist() for word in words])
    return translated

if input_text:
    translated_text = translate_text(input_text, st.session_state.lang_direction)
    st.session_state.output_text = translated_text
