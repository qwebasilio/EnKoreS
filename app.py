import streamlit as st
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import requests
from io import StringIO

model_name = "KETI-AIR/ke-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

csv_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/master/sample_dataset.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    st.error("Failed to load CSV file from GitHub.")
    data = None

def translate_with_t5(input_text, lang_direction):
    if lang_direction == "EN to KR":
        translation_prompt = f"translate English to Korean: {input_text}"
    else:
        translation_prompt = f"translate Korean to English: {input_text}"
    inputs = tokenizer(translation_prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
    st.header("English" if st.session_state.lang_direction == "EN to KR" else "Korean", anchor="center")
    input_text = st.text_area(
        "",
        value=st.session_state.input_text,
        height=200,
        key="input_text_box",
        label_visibility="collapsed",
        help="Type text to be translated."
    )
    if input_text != st.session_state.input_text:
        st.session_state.input_text = input_text
        st.session_state.output_text = translate_with_t5(st.session_state.input_text, st.session_state.lang_direction)

with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

with col2:
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English", anchor="center")
    st.text_area(
        "",
        value=st.session_state.output_text,
        height=200,
        disabled=True,
        key="output_text_box"
    )

if input_text != st.session_state.input_text:
    st.session_state.output_text = translate_with_t5(st.session_state.input_text, st.session_state.lang_direction)
    st.experimental_rerun()
