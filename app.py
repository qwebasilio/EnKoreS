import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from googletrans import Translator
from nltk.corpus import stopwords
import heapq

nltk.download('punkt')
nltk.download('stopwords')

translator = Translator()

def translate_text_google(input_text, src_lang, tgt_lang):
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text, language='english')
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

st.title("Translation and Summarization")

lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

input_text = st.text_area("Enter text to translate:")
if "output_text" not in st.session_state:
    st.session_state.output_text = ""

if input_text and st.button("Translate"):
    src_lang = "en" if lang_direction == "EN to KO" else "ko"
    tgt_lang = "ko" if lang_direction == "EN to KO" else "en"
    st.session_state.output_text = translate_text_google(input_text, src_lang, tgt_lang)

if st.session_state.output_text:
    st.text_area("Translated Text:", value=st.session_state.output_text, height=150)
    if st.button("Summarize"):
        summary = summarize_text(st.session_state.output_text)
        st.write("Summarized Text:")
        st.write(summary)