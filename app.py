import pandas as pd
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import requests
from io import StringIO  # Import StringIO from the io module

# Load the pre-trained KETI-AIR/ke-t5-small model and tokenizer from Hugging Face
model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR/ke-t5-small")
tokenizer = AutoTokenizer.from_pretrained("KETI-AIR/ke-t5-small", legacy=False)

# Function to translate text using the model
def translate_text(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# Load the dataset from GitHub
dataset_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/main/sample_dataset.csv"  # Update this with the actual URL to your dataset
response = requests.get(dataset_url)
data = pd.read_csv(StringIO(response.text))  # Use StringIO from the io module to read the CSV data

# Display the dataset
st.title("Text Translation with Fine-Tuned Model")
st.write("Dataset Overview:")
st.dataframe(data.head())  # Show the first few rows of the dataset

# UI for input text
input_text = st.text_area("Enter text to translate:")

# Translate text when the button is clicked
if input_text:
    translated_text = translate_text(input_text)
    st.write("Translated Text:")
    st.write(translated_text)
