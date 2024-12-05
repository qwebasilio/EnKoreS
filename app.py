import streamlit as st
import nltk
import os
from easynmt import EasyNMT
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import heapq

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt_tab")):
    nltk.download('punkt_tab', download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "tokenizers", "punkt")):
    nltk.download('punkt', download_dir=nltk_data_dir)
if not os.path.exists(os.path.join(nltk_data_dir, "corpora", "stopwords")):
    nltk.download('stopwords', download_dir=nltk_data_dir)

model = EasyNMT('m2m_100_418M')

def translate_text(text, src_lang, tgt_lang):
    if src_lang == "en" and tgt_lang == "ko":
        translated_text = model.translate(text, source_lang="en", target_lang="ko")
    elif src_lang == "ko" and tgt_lang == "en":
        translated_text = model.translate(text, source_lang="ko", target_lang="en")
    else:
        raise ValueError(f"Unsupported language pair: {src_lang} to {tgt_lang}")
    return translated_text

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word not in stop_words and word.isalnum():
                word_frequencies[word] = word_frequencies.get(word, 0) + 1

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        words = word_tokenize(sentence.lower())
        sentence_scores[i] = sum(word_frequencies.get(word, 0) for word in words)

    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(sentences[i] for i in sorted(best_sentences))
    return summary

st.title("EnKoreS")

st.sidebar.title("Settings")
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KR", "KR to EN"])
st.session_state.lang_direction = lang_direction

input_text = st.text_area("Enter text to translate:")
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if input_text != st.session_state.input_text:
    st.session_state.input_text = input_text
    st.session_state.output_text = translate_text(input_text, "en", "ko" if lang_direction == "EN to KR" else "en")

translate_button = st.button("Translate")

if translate_button:
    if input_text:
        st.session_state.output_text = translate_text(input_text, "en", "ko" if lang_direction == "EN to KR" else "en")
        st.write("Translated Text:")
        st.write(st.session_state.output_text)
    else:
        st.warning("Please enter text to translate.")

if st.session_state.output_text:
    summarize_button = st.button("Summarize")
    if summarize_button:
        summary = summarize_text(st.session_state.output_text)  # Summarize the translated text
        st.write("Summarized Text:")
        st.write(summary)

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)