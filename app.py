import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import requests
from io import StringIO

model_name = "Helsinki-NLP/opus-mt-en-ko"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def translate_text(text, src_lang="en", tgt_lang="ko"):
    tokenized_input = tokenizer(text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**tokenized_input)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

csv_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/master/sample_dataset.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    st.error("Failed to load CSV file from GitHub.")
    data = None

VALID_LANG_CODES = ["en_XX", "ko_KR"]

st.title("EnKoreS: English-Korean Translator")

def switch_languages():
    if st.session_state.lang_direction == "EN to KR":
        st.session_state.lang_direction = "KR to EN"
    else:
        st.session_state.lang_direction = "EN to KR"
    st.session_state.input_text, st.session_state.output_text = st.session_state.output_text, st.session_state.input_text

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

col1, col_switch, col2 = st.columns([4, 1, 4])

with col1:
    st.header("English" if st.session_state.lang_direction == "EN to KR" else "Korean")
    input_text = st.text_area(
        "Input Text",
        value=st.session_state.input_text,
        height=200,
        key="input_text_box"
    )
    if input_text != st.session_state.input_text:
        st.session_state.input_text = input_text
        lang_dir = st.session_state.lang_direction
        st.session_state.output_text = get_translation(st.session_state.input_text, data)

with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

with col2:
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English")
    st.text_area(
        "Translated Text",
        value=st.session_state.output_text,
        height=200,
        disabled=True,
        key="output_text_box"
    )

def get_translation(input_text, data, lang_column="question2_ko"):
    if data is not None:
        existing_translation = data[data['question2_en'] == input_text]
        if not existing_translation.empty:
            translated_text = existing_translation[lang_column].iloc[0]
            st.write(f"Found existing translation: {translated_text}")
        else:
            translated_text = translate_text(input_text, "en", "ko")
            st.write(f"Generated new translation: {translated_text}")
        return translated_text
    return ""

if input_text != st.session_state.input_text:
    st.session_state.output_text = get_translation(st.session_state.input_text, data)
    st.write(f"Live Translation: {st.session_state.output_text}")
