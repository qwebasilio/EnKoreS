import os
import gdown
import zipfile
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Define the path for the downloaded zip and extraction folder
current_dir = os.getcwd()
zip_file_path = os.path.join(current_dir, 'ke-t5-small-finetuned.zip')  # Updated to your model's zip file name
extraction_path = os.path.join(current_dir, 'model')

# Google Drive direct download URL for the fine-tuned model
google_drive_url = 'https://drive.google.com/uc?id=1W0qpVfmcGzXNPESMVVU0Iq1tpSS8SGr0&export=download'

# Download the fine-tuned model zip file from Google Drive
if not os.path.exists(extraction_path):
    gdown.download(google_drive_url, zip_file_path, quiet=False)
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
    st.write("Model downloaded and extracted successfully.")
else:
    st.write("Model already exists in the content folder.")

# Load the model and tokenizer from the extracted directory
model = AutoModelForSeq2SeqLM.from_pretrained(extraction_path)
tokenizer = AutoTokenizer.from_pretrained(extraction_path)

# Translation function
def translate_with_marian(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize session state variables if not already initialized
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Switch language direction
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
        st.session_state.output_text = translate_with_marian(st.session_state.input_text)

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
    st.session_state.output_text = translate_with_marian(st.session_state.input_text)
    st.experimental_rerun()
