import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from googletrans import Translator
from nltk.corpus import stopwords
from pyAutoSummarizer.base import summarization

@st.cache_data
def download_nltk_data():
    nltk.download("punkt_tab")
    nltk.download("stopwords")

download_nltk_data()

translator = Translator()

korean_stopwords = [
    "이", "그", "저", "은", "는", "이었", "으로", "에서", "를", "에", "와", "과", "도", "로", "의", "게",
    "을", "한", "들", "임", "다", "고", "하", "되", "있", "등", "을", "입니다", "합니다", "이것", "저것"
]

translated_text = ""
summarized_text = ""

def translate_text_google(input_text, src_lang, tgt_lang):
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def summarize_with_pyAutoSummarizer(text, num_sentences=3, stop_words_lang='en'):
    try:
        stop_words = korean_stopwords if stop_words_lang == 'ko' else []
        
        parameters = {
            'stop_words': stop_words, 
            'n_words': 100,  # Limit to 100 words for summarization
            'n_chars': -1,
            'lowercase': True,
            'rmv_accents': True,
            'rmv_special_chars': True,
            'rmv_numbers': False,
            'rmv_custom_words': [],
            'verbose': False
        }
        
        if not text:
            st.error("The text is empty, unable to summarize.")
            return "Text is too short to summarize."
        
        # Create summarization instance
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        
        if rank is None or len(rank) == 0:
            st.error("Summarization failed. Try providing a more detailed text.")
            return "Summarization failed."
        
        # Show the summarized text
        summary = smr.show_summary(rank, n=num_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error during summarization."

# Streamlit UI setup
st.title("EnKoreS")

# Session state setup for language direction and input/output texts
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""

# Language direction selection
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Reset state when language direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

# Text input area for translation
st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

# Translation button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        src_lang = "en" if st.session_state.lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if st.session_state.lang_direction == "EN to KO" else "en"
        translated_text = translate_text_google(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.translated_text = translated_text
        st.session_state.summarized_text = ""  # Clear any previous summaries

# Display translated text
if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    # Summarize button
    if st.button("Summarize"):
        lang = "english" if st.session_state.lang_direction == "KO to EN" else "korean"
        if st.session_state.translated_text.strip():
            summarized_text = summarize_with_pyAutoSummarizer(st.session_state.translated_text, stop_words_lang=('ko' if lang == 'korean' else 'en'))
            st.session_state.summarized_text = summarized_text

# Display summarized text
if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
