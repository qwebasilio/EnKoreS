import streamlit as st
import os
import gzip
import requests
import fasttext

# Function to download files from Google Drive
def download_from_drive(url, save_path):
    st.write(f"Downloading {save_path} from Google Drive...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
    st.write(f"Downloaded {save_path} successfully.")

# Decompress .gz file
def decompress_gz(gz_path, vec_path):
    st.write(f"Decompressing {gz_path}...")
    with gzip.open(gz_path, "rb") as gz_file:
        with open(vec_path, "wb") as vec_file:
            vec_file.write(gz_file.read())
    st.write(f"Decompressed to {vec_path}")

# File URLs from Google Drive (ensure public sharing)
en_url = "YOUR_ENGLISH_GZ_FILE_LINK"
ko_url = "YOUR_KOREAN_GZ_FILE_LINK"

# Local paths
en_gz_path = "cc.en.300.vec.gz"
ko_gz_path = "cc.ko.300.vec.gz"
en_vec_path = "cc.en.300.vec"
ko_vec_path = "cc.ko.300.vec"

# Download files if not present
if not os.path.exists(en_gz_path):
    download_from_drive(en_url, en_gz_path)
if not os.path.exists(ko_gz_path):
    download_from_drive(ko_url, ko_gz_path)

# Decompress files if not already decompressed
if not os.path.exists(en_vec_path):
    decompress_gz(en_gz_path, en_vec_path)
if not os.path.exists(ko_vec_path):
    decompress_gz(ko_gz_path, ko_vec_path)

# Load FastText models
st.write("Loading FastText models, please wait...")
en_model = fasttext.load_model(en_vec_path)
ko_model = fasttext.load_model(ko_vec_path)
st.write("FastText models loaded successfully!")

# Translator Function
def translate(text, source_model, target_model):
    words = text.split()
    translated_words = []
    for word in words:
        try:
            # Get source word vector
            word_vec = source_model.get_word_vector(word)
            # Find closest word in target language
            closest_word = target_model.get_nearest_neighbors(word_vec, k=1)[0][1]
            translated_words.append(closest_word)
        except Exception as e:
            translated_words.append(word)  # Append original word if not found
    return " ".join(translated_words)

# Streamlit App
st.title("EnKoreS: English-Korean Translator with Summarization")

# Input text area
input_text = st.text_area("Enter text to translate:", "")

# Switch language logic
translation_direction = st.radio("Select Translation Direction:", ["EN to KR", "KR to EN"])

if st.button("Translate"):
    if input_text.strip():
        st.write("Translating...")
        if translation_direction == "EN to KR":
            translated_text = translate(input_text, en_model, ko_model)
        else:
            translated_text = translate(input_text, ko_model, en_model)
        st.text_area("Translated Text:", translated_text, height=200)
    else:
        st.warning("Please enter text to translate.")

# Summarization (Dummy logic for now, replace with NLP logic)
if st.button("Translate & Summarize"):
    if input_text.strip():
        st.write("Translating and summarizing...")
        if translation_direction == "EN to KR":
            translated_text = translate(input_text, en_model, ko_model)
        else:
            translated_text = translate(input_text, ko_model, en_model)
        # Placeholder for summarization logic
        summarized_text = "Summarized content goes here."
        st.text_area("Translated Text:", translated_text, height=200)
        st.text_area("Summarized Text:", summarized_text, height=150)
    else:
        st.warning("Please enter text to translate.")