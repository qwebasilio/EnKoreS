import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from googletrans import Translator
from nltk.corpus import stopwords
import heapq

# Download NLTK resources explicitly
nltk.download('punkt_tab')
nltk.download('stopwords')

translator = Translator()

# Korean stopwords (customized)
korean_stopwords = [
    "이", "그", "저", "은", "는", "이었", "으로", "에서", "를", "에", "와", "과", "도", "로", "의", "게",
    "을", "한", "들", "임", "다", "고", "하", "되", "있", "등", "을", "입니다", "합니다"
]

def translate_text_google(input_text, src_lang, tgt_lang):
    """Translate text using Google Translator."""
    try:
        translation = translator.translate(input_text, src=src_lang, dest=tgt_lang)
        return translation.text
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return ""

def tokenize_text(text, lang):
    """Tokenize text based on language."""
    try:
        if lang == "english":
            return sent_tokenize(text, language="english")
        elif lang == "korean":
            return sent_tokenize(text, language="korean")
        else:
            return []
    except LookupError:
        # Fallback tokenization for languages without proper support
        return text.split(". ")

def summarize_text(text, num_sentences=3, lang="english"):
    """Summarize text."""
    sentences = tokenize_text(text, lang)
    stop_words = set(stopwords.words(lang if lang in stopwords.fileids() else 'english'))
    if lang == "korean":
        stop_words.update(korean_stopwords)
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

# Streamlit app layout
st.title("EnKoreS")

# Initialize session state variables
if "lang_direction" not in st.session_state:
    st.session_state.lang_direction = "EN to KO"
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "output_text" not in st.session_state:
    st.session_state.output_text = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""

# Translation direction toggle
lang_direction = st.sidebar.radio("Select Translation Direction", ["EN to KO", "KO to EN"])

# Clear input and output when language direction changes
if lang_direction != st.session_state.lang_direction:
    st.session_state.lang_direction = lang_direction
    st.session_state.input_text = ""
    st.session_state.output_text = ""
    st.session_state.summary_text = ""

# Input text area
st.session_state.input_text = st.text_area("Enter text to translate:", value=st.session_state.input_text)

# Translate button
if st.button("Translate"):
    if st.session_state.input_text.strip():
        src_lang = "en" if st.session_state.lang_direction == "EN to KO" else "ko"
        tgt_lang = "ko" if st.session_state.lang_direction == "EN to KO" else "en"
        st.session_state.output_text = translate_text_google(st.session_state.input_text, src_lang, tgt_lang)
        st.session_state.summary_text = ""

# Display translated text
if st.session_state.output_text:
    st.text_area("Translated Text:", value=st.session_state.output_text, height=150, disabled=True)

    # Summarize button
    if st.button("Summarize"):
        lang = "english" if st.session_state.lang_direction == "KO to EN" else "korean"
        if st.session_state.output_text.strip():
            st.session_state.summary_text = summarize_text(st.session_state.output_text, lang=lang)

# Display summarized text
if st.session_state.summary_text:
    st.text_area("Summarized Text:", value=st.session_state.summary_text, height=150, disabled=True)