import streamlit as st
import nltk
import os
from easynmt import EasyNMT
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

model = EasyNMT('m2m_100_418M')

VALID_LANG_CODES = ['ko', 'en']

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
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_score = 0
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in word_frequencies:
                sentence_score += word_frequencies[word]
        sentence_scores[i] = sentence_score

    best_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get())
    summary = [sentences[i] for i in sorted(best_sentences)]
    return ' '.join(summary)

st.title("EnKoreS")

st.sidebar.title("Settings")
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KR", "KR to EN"])
st.session_state.lang_direction = lang_direction

input_text = st.text_area("Enter text to translate:")
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

translate_button = st.button("Translate")

translate_and_summarize_button = st.button("Translate & Summarize")

if translate_button:
    if input_text:
        src_lang = "en" if lang_direction == "EN to KR" else "ko"
        tgt_lang = "ko" if lang_direction == "EN to KR" else "en"
        st.session_state.output_text = translate_text(input_text, src_lang, tgt_lang)
        st.subheader("Translated Text:")
        st.write(st.session_state.output_text)
    else:
        st.warning("Please enter text to translate.")

if translate_and_summarize_button:
    if input_text:
        src_lang = "en" if lang_direction == "EN to KR" else "ko"
        tgt_lang = "ko" if lang_direction == "EN to KR" else "en"
        translated_text = translate_text(input_text, src_lang, tgt_lang)
        st.subheader("Translated Text:")
        st.write(translated_text)

        summarized_text = summarize_text(translated_text)
        st.subheader("Summarized Text:")
        st.write(summarized_text)
    else:
        st.warning("Please enter text to translate and summarize.")

st.text_area("Translated Text:", value=st.session_state.output_text, height=150)
