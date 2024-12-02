import streamlit as st
import numpy as np
import os
import requests

def download_from_drive(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path + ".gz", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def load_vec_file(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as f:
        num_words, dim = map(int, f.readline().split())
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            word_vectors[word] = vector
    return word_vectors, dim

en_url = "https://drive.google.com/file/d/14zlXmW3iUgx39jU6uwDIbcYLe4QK2FFi/view?usp=drive_link"
ko_url = "https://drive.google.com/file/d/1L4sjC9DjBqNlaCf6MDpXfHLbyjN9T_na/view?usp=drive_link"

en_file = "cc.en.50.vec"
ko_file = "cc.ko.50.vec"

if not os.path.exists(en_file):
    st.write("Downloading English vector file...")
    download_from_drive(en_url, en_file)

if not os.path.exists(ko_file):
    st.write("Downloading Korean vector file...")
    download_from_drive(ko_url, ko_file)

print(f"Current working directory: {os.getcwd()}")

st.write("Loading FastText models...")
en_word_vectors, en_dim = load_vec_file(en_file)
ko_word_vectors, ko_dim = load_vec_file(ko_file)
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
        translated = " ".join([str(ko_word_vectors.get(word, np.zeros(50))) for word in words])
    else:
        translated = " ".join([str(en_word_vectors.get(word, np.zeros(50))) for word in words])
    return translated

if input_text:
    translated_text = translate_text(input_text, st.session_state.lang_direction)
    st.session_state.output_text = translated_text
