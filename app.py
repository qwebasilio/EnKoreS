import os
import gdown
import zipfile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st

# Google Drive file download link
google_drive_url = 'https://drive.google.com/uc?id=1W0qpVfmcGzXNPESMVVU0Iq1tpSS8SGr0'
zip_file_path = '/content/ke-t5-small-finetuned.zip'
extraction_path = '/content/model'

# Check if the extracted model directory exists, if not, download and extract it
if not os.path.exists(extraction_path):
    # Download the ZIP file from Google Drive
    gdown.download(google_drive_url, zip_file_path, quiet=False)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

# Load the fine-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(extraction_path)
tokenizer = AutoTokenizer.from_pretrained(extraction_path)

# Define the translation function
def translate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Streamlit UI
st.title('Text Translation with Fine-Tuned Model')

input_text = st.text_area("Enter text to translate:")

if input_text:
    translated_text = translate_text(input_text)
    st.write("Translated Text:")
    st.write(translated_text)
