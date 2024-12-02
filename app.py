import streamlit as st
import pandas as pd
from transformers import pipeline
import requests
from io import StringIO

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

csv_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/master/sample_dataset.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    st.error("Failed to load CSV file from GitHub.")
    data = None

def get_translation(input_text, data, lang_column="question2_ko"):
    if data is not None:
        existing_translation = data[data['question2_en'] == input_text]
        if not existing_translation.empty:
            translated_text = existing_translation[lang_column].iloc[0]
            st.write(f"Found existing translation: {translated_text}")
        else:
            translated_text = translator(input_text)[0]['translation_text']
            st.write(f"Generated new translation: {translated_text}")
        return translated_text
    return ""

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

def switch_languages():
    if st.session_state.lang_direction == "EN to KR":
        st.session_state.lang_direction = "KR to EN"
    else:
        st.session_state.lang_direction = "EN to KR"
    st.session_state.input_text, st.session_state.output_text = st.session_state.output_text, st.session_state.input_text

st.title("EnKoreS")

col1, col_switch, col2 = st.columns([4, 1, 4])

with col1:
    input_text = st.text_area(
        "Type text to translate",
        value=st.session_state.input_text,
        height=200,
        key="input_text_box",
        label_visibility="collapsed",
        help="Type text to be translated."
    )
    if input_text != st.session_state.input_text:
        st.session_state.input_text = input_text
        st.session_state.output_text = get_translation(st.session_state.input_text, data)

with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

with col2:
    st.text_area(
        "Translated Text",
        value=st.session_state.output_text,
        height=200,
        disabled=True,
        key="output_text_box"
    )

if input_text != st.session_state.input_text:
    st.session_state.output_text = get_translation(st.session_state.input_text, data)
    st.experimental_rerun()
