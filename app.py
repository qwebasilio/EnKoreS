import streamlit as st
import fasttext
import os
import requests

def download_from_drive(url, save_path):
  with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open(save_path + ".gz", "wb") as f:
      for chunk in response.iter_content(chunk_size=8192):
        if chunk:
          f.write(chunk)

en_url = "https://drive.google.com/file/d/1KKLysauoldy9Elc_m6Om55Q_wFLwRC7U/view?usp=drive_link"
ko_url = "https://drive.google.com/file/d/1ammCe6kFeX7kbM2RsjU9nmXKNpPOp_AL/view?usp=drive_link"

en_file = "cc.en.300.vec.gz"
ko_file = "cc.ko.300.vec.gz" 

if not os.path.exists(en_file):
  st.write("Downloading English vector file...")
  download_from_drive(en_url, en_file)

if not os.path.exists(ko_file):
  st.write("Downloading Korean vector file...")
  download_from_drive(ko_url, ko_file)

st.write("Loading FastText models...")
en_model = fasttext.load_model(en_file, mmap=None)
ko_model = fasttext.load_model(ko_file, mmap=None)
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
