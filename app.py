import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import streamlit as st

model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

csv_file = "your_file.csv"
data = pd.read_csv(csv_file)

def translate_text(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded_input = tokenizer(text, return_tensors="pt", padding=True)
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return translated_text

def get_translation(input_text, data, lang_column="question2_ko"):
    existing_translation = data[data['question2_en'] == input_text]
    
    if not existing_translation.empty:
        translated_text = existing_translation[lang_column].iloc[0]
        st.write(f"Found existing translation: {translated_text}")
    else:
        translated_text = translate_text(input_text, "en_XX", "ko_XX")
        st.write(f"Generated new translation: {translated_text}")
    return translated_text

st.title("EnKoreS")

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
        if lang_dir == "EN to KR":
            st.session_state.output_text = get_translation(st.session_state.input_text, data, lang_column="question2_ko")
        else:
            st.session_state.output_text = get_translation(st.session_state.input_text, data, lang_column="question2_en")

with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

# Right column (translated text)
with col2:
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English")
    st.text_area(
        "Translated Text",
        value=st.session_state.output_text,
        height=200,
        disabled=True,
        key="output_text_box"
    )
