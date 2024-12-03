import streamlit as st
import pandas as pd
from transformers import pipeline
import requests
from io import StringIO

# Initialize MarianMT model for translation (Korean to English)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

# Load CSV from GitHub URL
csv_url = "https://raw.githubusercontent.com/qwebasilio/EnKoreS/master/sample_dataset.csv"
response = requests.get(csv_url)

if response.status_code == 200:
    data = pd.read_csv(StringIO(response.text))
else:
    st.error("Failed to load CSV file from GitHub.")
    data = None

# Function to get translation: If input found in CSV, use that, else translate with model
def get_translation(input_text, data, lang_column="question2_ko"):
    if data is not None:
        # Check if the input text already exists in the CSV (based on English column)
        existing_translation = data[data['question2_en'] == input_text]
        if not existing_translation.empty:
            # Return existing translation from CSV
            translated_text = existing_translation[lang_column].iloc[0]
            st.write(f"Found existing translation: {translated_text}")
        else:
            # Translate text using the MarianMT model (new translation)
            translated_text = translator(input_text)[0]['translation_text']
            st.write(f"Generated new translation: {translated_text}")
        return translated_text
    return ""

# Initialize session state for language direction and text inputs
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KR"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

# Function to switch languages (between EN to KR and KR to EN)
def switch_languages():
    if st.session_state.lang_direction == "EN to KR":
        st.session_state.lang_direction = "KR to EN"
    else:
        st.session_state.lang_direction = "EN to KR"
    # Swap the text values when language direction changes
    st.session_state.input_text, st.session_state.output_text = st.session_state.output_text, st.session_state.input_text

# Set up Streamlit layout
st.title("EnKoreS")

# Add columns for input and output
col1, col_switch, col2 = st.columns([4, 1, 4])

# Column for user input
with col1:
    # Set header to reflect language direction
    st.header("English" if st.session_state.lang_direction == "EN to KR" else "Korean", anchor="center")
    
    # Input text area
    input_text = st.text_area(
        "Type text to translate",
        value=st.session_state.input_text,
        height=200,
        key="input_text_box",
        label_visibility="collapsed",
        help="Type text to be translated."
    )
    
    # Update session state when input text changes
    if input_text != st.session_state.input_text:
        st.session_state.input_text = input_text
        st.session_state.output_text = get_translation(st.session_state.input_text, data)

# Button to switch languages (EN <-> KR)
with col_switch:
    st.button("⇋", on_click=switch_languages, use_container_width=True)

# Column for showing translated text
with col2:
    # Set header to reflect language direction
    st.header("Korean" if st.session_state.lang_direction == "EN to KR" else "English", anchor="center")
    
    # Disabled text area for translated text
    st.text_area(
        "Translated Text",
        value=st.session_state.output_text,
        height=200,
        disabled=True,
        key="output_text_box"
    )

# Trigger translation when input text is updated (this ensures the output updates immediately)
if input_text != st.session_state.input_text:
    st.session_state.output_text = get_translation(st.session_state.input_text, data)
    st.experimental_rerun()  # Re-run the app to update outputs dynamically