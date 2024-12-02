import streamlit as st
import os
import gzip
import requests
import fasttext

def download_from_drive(url, save_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

def decompress_gz(gz_path, vec_path):
    with gzip.open(gz_path, "rb") as gz_file:
        with open(vec_path, "wb") as vec_file:
            vec_file.write(gz_file.read())

en_url = "https://drive.google.com/file/d/1V9lwDZOSpaPLhk6wJ0EPoY2fwY2sGHfw/view?usp=drive_link"
ko_url = "https://drive.google.com/file/d/1--87G0NQFl33ewSJ1uj53I3G7L8wPJI6/view?usp=drive_link"

en_gz_path = "cc.en.300.vec.gz"
ko_gz_path = "cc.ko.300.vec.gz"
en_vec_path = "cc.en.300.vec"
ko_vec_path = "cc.ko.300.vec"

if not os.path.exists(en_gz_path):
    download_from_drive(en_url, en_gz_path)
if not os.path.exists(ko_gz_path):
    download_from_drive(ko_url, ko_gz_path)

if not os.path.exists(en_vec_path):
    decompress_gz(en_gz_path, en_vec_path)
if not os.path.exists(ko_vec_path):
    decompress_gz(ko_gz_path, ko_vec_path)

st.write("Loading FastText models, please wait...")
en_model = fasttext.load_model(en_vec_path)
ko_model = fasttext.load_model(ko_vec_path)
st.session_state["source_lang"] = "EN"
st.session_state["target_lang"] = "KR"
st.session_state["is_loaded"] = True

if st.session_state.get("is_loaded"):
    st.session_state.pop("is_loaded")
    st.experimental_rerun()

def translate(text, source_model, target_model):
    words = text.split()
    translated_words = []
    for word in words:
        try:
            word_vec = source_model.get_word_vector(word)
            closest_word = target_model.get_nearest_neighbors(word_vec, k=1)[0][1]
            translated_words.append(closest_word)
        except Exception:
            translated_words.append(word)
    return " ".join(translated_words)

st.title("EnKoreS")

col1, col2 = st.columns([1, 1])

with col1:
    st.text(f"{st.session_state['source_lang']} Input")
    input_text = st.text_area("", key="input_text")
with col2:
    st.text(f"{st.session_state['target_lang']} Output")
    translated_text = translate(input_text, en_model if st.session_state["source_lang"] == "EN" else ko_model,
                                ko_model if st.session_state["source_lang"] == "EN" else en_model)
    st.text_area("", translated_text, key="translated_text", disabled=True)

if st.button(f"Switch to {st.session_state['target_lang']} to {st.session_state['source_lang']}"):
    st.session_state["source_lang"], st.session_state["target_lang"] = st.session_state["target_lang"], st.session_state["source_lang"]
    st.experimental_rerun()

if st.button("Translate & Summarize"):
    summarized_text = "Summarized content goes here."
    st.text_area("Summary", summarized_text, height=150, disabled=True)
