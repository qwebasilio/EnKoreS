import streamlit as st
import pandas as pd
from asian_bart import AsianBartTokenizer, AsianBartForConditionalGeneration
import torch
from transformers.models.bart.modeling_bart import shift_tokens_right
import requests
from io import StringIO

model_name = "hyunwoongko/asian-bart-ecjk"
tokenizer = AsianBartTokenizer.from_pretrained(model_name)
model = AsianBartForConditionalGeneration.from_pretrained(model_name)

csv_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/master/sample_dataset.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    st.error("Failed to load CSV file from GitHub.")
    data = None

def translate_with_asian_bart(input_text, lang_direction):
    src_lang = "en_XX" if lang_direction == "EN to KR" else "ko_KR"
    tgt_lang = "ko_KR" if lang_direction == "EN to KR" else "en_XX"
    tokens = tokenizer.prepare_seq2seq_batch(src_texts=input_text, src_langs=src_lang, tgt_langs=tgt_lang)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

def switch_languages():
    st.session_state.lang_direction = "KR to EN" if st.session_state.lang_direction == "EN to KR" else "EN to KR"
    st.session_state.input_text, st.session_state.output_text = st.session_state.output_text, st.session_state.input_text

st.title("EnKoreS - AsianBart ECJK")

col1, col_switch, col2 = st.columns([4, 1, 4])

with col1:
    st.header("English" if st.session_state.lang_direction == "EN to KR" else "Korean")
    input_text = st.text_area("", value=st.session_state.input_text, height=200, key="input_text_box", label_visibility="collapsed")
    if input_text != st.session_state.input_text:
        st.session_state.input_text = input_text
        st.session_state.output_text = translate_with_asian_bart(st.session_state.input_text, st.session_state.lang_direction)

with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

with col2:
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English")
    st.text_area("", value=st.session_state.output_text, height=200, disabled=True, key="output_text_box")

if input_text != st.session_state.input_text:
    st.session_state.output_text = translate_with_asian_bart(st.session_state.input_text, st.session_state.lang_direction)
    st.experimental_rerun()
