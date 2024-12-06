import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from googletrans import Translator
from pyAutoSummarizer.base import summarization

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
    if stop_words_lang == 'ko':
        stop_words = korean_stopwords
        
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
        
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        
        if rank is None or len(rank) == 0:
            st.error("Summarization failed. Try providing a more detailed text.")
            return "Summarization failed."
        
        summary = smr.show_summary(rank, n=num_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error during summarization."


elif stop_words_lang='en'        
        parameters = {
            'stop_words': ['en'], 
            'n_words': -1,  # Limit to 100 words for summarization
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
        
        smr = summarization(text, **parameters)
        rank = smr.summ_ext_LSA(embeddings=False, model='all-MiniLM-L6-v2')
        
        if rank is None or len(rank) == 0:
            st.error("Summarization failed. Try providing a more detailed text.")
            return "Summarization failed."
        
        summary = smr.show_summary(rank, n=num_sentences)
        return summary
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error during summarization."

st.title("EnKoreS")

if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""

lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.translated_text = ""
    st.session_state.summarized_text = ""

st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

if st.button("Translate"):
    if st.session_state.input_text.strip():
        src_lang = "en" if st.session_state.lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if st.session_state.lang_direction == "EN to KO" else "en"
        translated_text = translate_text_google(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.translated_text = translated_text
        st.session_state.summarized_text = ""

if st.session_state.translated_text:
    st.text_area("Translated Text:", value=st.session_state.translated_text, height=150, disabled=True)

    if st.button("Summarize"):
        lang = "english" if st.session_state.lang_direction == "KO to EN" else "korean"
        if st.session_state.translated_text.strip():
            summarized_text = summarize_with_pyAutoSummarizer(translated_text, stop_words_lang=('ko' if lang == 'korean' else 'en'))
            st.session_state.summarized_text = summarized_text

if st.session_state.summarized_text:
    st.text_area("Summarized Text:", value=st.session_state.summarized_text, height=150, disabled=True)
