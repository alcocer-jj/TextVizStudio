import streamlit as st
import pandas as pd
import numpy as np
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, XLMRobertaTokenizer, AutoConfig
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
from collections import defaultdict
import re
from datetime import datetime
from pathlib import Path
from io import BytesIO, StringIO
from scipy.special import softmax
import sentencepiece
import stanza


# Set the Streamlit page configuration
st.set_page_config(
    page_title="Text2Sentiment",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Size caps per method
# ---------------------------------------------------------------------------
SIZE_LIMITS = {
    "Dictionary - VADER": 50000,
    "Dictionary - NRC Lexicon (Default)": 25000,
    "LLM - XLM-RoBERTa-Twitter-Sentiment": 5000,
}

# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@st.cache_resource
def get_gsheets_client():
    """Authenticate and return a gspread client. Cached for the life of the app."""
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    return gspread.authorize(creds)


@st.cache_resource
def load_xlm_model():
    """Load XLM-RoBERTa-Twitter-Sentiment once, share across all sessions."""
    MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    if not hasattr(config, 'id2label') or not config.id2label:
        config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    model.eval()  # Disable dropout etc. for inference
    return tokenizer, config, model


@st.cache_resource
def initialize_stanza_pipeline(language_code):
    """Initialize Stanza pipeline. Cached per language across all users."""
    stanza.download(language_code, processors='tokenize,lemma', verbose=False)
    return stanza.Pipeline(lang=language_code, processors='tokenize,lemma')


@st.cache_data
def load_nrc_lexicon(language_filename):
    """
    Load NRC lexicon CSV and build the word -> {emotion: condition} dict.
    Cached per language CSV across all users.

    Replaces the previous iterrows() loop with pivot_table for speed.
    aggfunc='last' preserves the original last-write-wins semantics in case
    of duplicate (word, emotion) pairs in the source data.
    """
    nrc_data = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / language_filename
    )
    nrc_data = nrc_data.dropna(subset=['word'])
    pivoted = nrc_data.pivot_table(
        index='word',
        columns='emotion',
        values='condition',
        fill_value=0,
        aggfunc='last',
    )
    return pivoted.to_dict(orient='index')


# ---------------------------------------------------------------------------
# Feedback sidebar
# ---------------------------------------------------------------------------
try:
    client = get_gsheets_client()
    sheet = client.open("TextViz Studio Feedback").sheet1

    st.sidebar.markdown("### **Feedback**")
    feedback = st.sidebar.text_area(
        "Experiencing bugs/issues? Have ideas to better the application tool?",
        placeholder="Leave feedback or error code here"
    )

    if st.sidebar.button("Submit"):
        if feedback:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row(["Text2Sentiment:", feedback, timestamp])
                st.sidebar.success("✅ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error("⚠️ Failed to submit feedback.")
                st.sidebar.caption(f"Error: {e}")
        else:
            st.sidebar.error("⚠️ Feedback cannot be empty!")

except Exception as e:
    st.sidebar.error("⚠️ Could not load feedback form.")
    st.sidebar.caption(f"Details: {e}")

st.sidebar.markdown("")

st.sidebar.markdown(
    "For full documentation and updates, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)

st.sidebar.markdown("")


st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment Discovery</h1>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
**Text2Sentiment** is an interactive tool for performing sentiment and emotion analysis on text data.
Upload a CSV file containing your text, and select your preferred sentiment analysis model. Choose from options like 
the **VADER** model for short, informal text, **Zero-shot Classifier** for broader domain adaptability, or the 
**NRC Emotion Lexicon** to conduct detailed emotion analysis across multiple languages, including French, Spanish, 
Italian, Portuguese, Chinese, and Arabic. Visualize the results with sentiment proportions and emotion breakdowns 
through bar charts and data tables. Configure language and model settings to tailor the analysis to your needs, and 
download the results for further analysis.
""")

st.markdown("")
st.markdown("")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()


def extract_text_from_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error("🚫 Failed to read the uploaded file as a CSV.")
        st.caption(f"Details: {e}")
        return None, None

    df.columns = df.columns.str.lower()

    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)
        return df.reset_index(drop=True), df
    else:
        st.error("⚠️ The uploaded CSV file must contain a column named 'text'.")
        return None, None


def preprocess_text_VADER(text, lowercase=False, remove_urls=True):
    if lowercase:
        text = text.lower()
    if remove_urls:
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    return text


def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    label = (
        "positive" if compound >= 0.05
        else "negative" if compound <= -0.05
        else "neutral"
    )
    return compound, label, scores['neg'], scores['neu'], scores['pos']


def preprocess_text_xlm(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def analyze_xlm_batch(texts, tokenizer, model, config, batch_size=32):
    """
    Batched replacement for the per-row .apply() pattern.

    Methodology preserved:
      - softmax over the 3 class logits per example
      - score_diff < 0.05 -> "neutral" rule applied identically per row
      - returns (negative, neutral, positive, sentiment) tuples in input order
    """
    total = len(texts)
    if total == 0:
        return []

    results = []
    progress = st.progress(0.0, text="Running XLM-RoBERTa sentiment analysis...")

    for i in range(0, total, batch_size):
        batch = [preprocess_text_xlm(t) for t in texts[i:i + batch_size]]
        encoded = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            output = model(**encoded)
        scores_batch = softmax(output[0].detach().numpy(), axis=1)

        for scores in scores_batch:
            negative, neutral, positive = scores[0], scores[1], scores[2]

            # --- Preserved methodology: neutral-if-top-two-are-close rule ---
            score_diff = abs(max(scores) - sorted(scores)[-2])
            if score_diff < 0.05:
                sentiment = "neutral"
            else:
                sentiment = config.id2label[int(np.argmax(scores))]
            # ----------------------------------------------------------------

            results.append((negative, neutral, positive, sentiment))

        progress.progress(min((i + batch_size) / total, 1.0))

    progress.empty()
    return results

def preprocess_text_stanza(text, nlp_pipeline):
    doc = nlp_pipeline(text)
    lemmatized_text = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
    return lemmatized_text


def analyze_nrc(text, emotion_dict):
    emotions = ['anger', 'fear', 'trust', 'joy', 'anticipation',
                'disgust', 'surprise', 'sadness', 'negative', 'positive']
    emotion_counts = defaultdict(int)

    words = re.findall(r'\b\w+\b', text.lower())

    for word in words:
        if word in emotion_dict:
            word_emotions = emotion_dict[word]
            for emotion in emotions:
                # .get() handles the case where pivot_table didn't include
                # this emotion column for this word
                emotion_counts[emotion] += word_emotions.get(emotion, 0)

    positive_score = emotion_counts['positive']
    negative_score = emotion_counts['negative']
    sentiment = (
        'positive' if positive_score > negative_score
        else 'negative' if negative_score > positive_score
        else 'neutral'
    )

    return pd.Series([
        emotion_counts['anger'], emotion_counts['fear'], emotion_counts['trust'],
        emotion_counts['joy'], emotion_counts['anticipation'], emotion_counts['disgust'],
        emotion_counts['surprise'], emotion_counts['sadness'],
        negative_score, positive_score, sentiment
    ])


# Plotly configuration
configuration = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 1
    }
}


# ---------------------------------------------------------------------------
# Main page flow
# ---------------------------------------------------------------------------

st.subheader("Import Data", divider=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")

df, original_csv = None, None

if uploaded_file is not None:
    df, original_csv = extract_text_from_csv(uploaded_file)

    if df is not None:
        st.write(f"CSV file successfully processed. ({len(df):,} rows)")

        st.subheader("Set Model Parameters")

        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["Dictionary - NRC Lexicon (Default)", "Dictionary - VADER", "LLM - XLM-RoBERTa-Twitter-Sentiment"]
        )

        # Size-cap check based on selected method. Halts the page if exceeded.
        max_rows = SIZE_LIMITS.get(sentiment_method, 10000)
        if len(df) > max_rows:
            st.error(
                f"⚠️ Your file has {len(df):,} rows, which exceeds the limit of "
                f"{max_rows:,} for the **{sentiment_method}** method. "
                f"Please reduce the file size or choose a lighter method "
                f"(VADER allows up to {SIZE_LIMITS['Dictionary - VADER']:,} rows)."
            )
            st.stop()

        with st.expander("Which model is right for me?"):
            st.markdown("""
            [**NRC Emotion Lexicon**](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm):
            - Default choice. Currently can handle 10 languages. If you want a specific language added, send an inquiry
            - Includes ability to analyze both sentiment and emotion analysis
            - Strengths: Provides granular emotional insights since it uses a pre-defined dictionary of ~10,000 words
            - Limitations: Less suited for nuanced, context-dependent entries

            [**VADER** (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment):
            - Best for short, informal texts (e.g., social media, product reviews)
            - Strengths: Fast processing, effective at handling negation (e.g., "not happy")
            - Limitations: Less accurate for longer, more complex text

            [**XLM-RoBERTa-Twitter-Sentiment**](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment):
            - Large Language Model fine-tuned on 198M Tweets across different eight languages (Arabic, English, French, German, Hindi, Italian, Portuguese, and Spanish)
            - Strengths: Can handle lexical diversity, nuances, and semantic context from tweets and short text across 65 languages (see [paper](https://arxiv.org/pdf/2104.12250) for more info)
            - Limitations: Slower processing time than dictionary methods; not good for long text entries unless broken down into chunks. 
            """)

        # Language selection (only for NRC).
        language = None
        selected_language_code = None
        language_codes = {
            "English": "english.csv",
            "French": "french.csv",
            "Spanish": "spanish.csv",
            "Italian": "italian.csv",
            "Portuguese": "portuguese.csv",
            "Chinese (Traditional)": "chinese_traditional.csv",
            "Chinese (Simplified)": "chinese_simplified.csv",
            "Arabic": "arabic.csv",
            "Turkish": "turkish.csv",
            "Korean": "korean.csv"
        }
        if sentiment_method == "Dictionary - NRC Lexicon (Default)":
            language = st.selectbox(
                "Select Language for NRC Lexicon Analysis",
                list(language_codes.keys())
            )
            selected_language_code = language_codes[language]

        st.subheader("Analyze", divider=True)

        # Analyze sentiment on button click
        if st.button("Analyze Sentiment"):
            try:
                with st.spinner("Running sentiment analysis..."):

                    if sentiment_method == "Dictionary - VADER":
                        vader = SentimentIntensityAnalyzer()
                        # Write cleaned text to a SEPARATE column so the
                        # original 'text' is preserved for re-runs
                        df['text_clean'] = df['text'].apply(preprocess_text_VADER)
                        df[['compound', 'sentiment', 'neg', 'neu', 'pos']] = df['text_clean'].apply(
                            lambda x: pd.Series(analyze_vader(x))
                        )

                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(df['sentiment'].value_counts().reset_index())

                        with col2:
                            sentiment_counts = df['sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig_sentiment = px.bar(
                                sentiment_counts, x='Sentiment', y='Count',
                                title='Sentiment Count Distribution', text='Count', color='Sentiment'
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)

                    elif sentiment_method == "Dictionary - NRC Lexicon (Default)":
                        stanza_language_codes = {
                            "English": "en",
                            "French": "fr",
                            "Spanish": "es",
                            "Italian": "it",
                            "Portuguese": "pt",
                            "Chinese (Traditional)": "zh",
                            "Chinese (Simplified)": "zh",
                            "Arabic": "ar",
                            "Turkish": "tr",
                            "Korean": "ko"
                        }

                        # CACHED: loads once per language across all users
                        emotion_dict = load_nrc_lexicon(selected_language_code)

                        # CACHED: loads once per language across all users
                        stanza_lang_code = stanza_language_codes.get(language)
                        nlp_pipeline = initialize_stanza_pipeline(stanza_lang_code)

                        # Lemmatize into a SEPARATE column to preserve original text
                        df['text_clean'] = df['text'].apply(
                            lambda x: preprocess_text_stanza(x, nlp_pipeline)
                        )
                        df[['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise',
                            'sadness', 'negative', 'positive', 'sentiment']] = df['text_clean'].apply(
                            lambda x: analyze_nrc(x, emotion_dict)
                        )

                        emotion_cols = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
                        emotion_counts = df[emotion_cols].sum().reset_index()
                        emotion_counts.columns = ['Emotion', 'Count']

                        st.subheader("Emotion Counts (NRC Lexicon)")
                        col1, col2 = st.columns([0.3, 0.7])
                        with col1:
                            st.write("Emotion Counts Dataframe:")
                            st.dataframe(emotion_counts)
                        with col2:
                            fig_emotions = px.bar(
                                emotion_counts, x='Emotion', y='Count',
                                title='Emotion Counts Distribution', text='Count', color='Emotion'
                            )
                            st.plotly_chart(fig_emotions, use_container_width=True, config=configuration)

                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']

                        st.subheader("Sentiment Counts (NRC Lexicon)")
                        col1, col2 = st.columns([0.3, 0.7])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(sentiment_counts)
                        with col2:
                            fig_sentiment = px.bar(
                                sentiment_counts, x='Sentiment', y='Count',
                                title='Sentiment Count Distribution', text='Count', color='Sentiment'
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)

                    elif sentiment_method == "LLM - XLM-RoBERTa-Twitter-Sentiment":
                        # CACHED: model loads ONCE per app process
                        tokenizer, config, model = load_xlm_model()

                        # Batched inference replaces the per-row .apply()
                        # score_diff rule preserved inside analyze_xlm_batch
                        results = analyze_xlm_batch(
                            df['text'].tolist(), tokenizer, model, config
                        )
                        df[['negative', 'neutral', 'positive', 'sentiment']] = pd.DataFrame(
                            results,
                            columns=['negative', 'neutral', 'positive', 'sentiment'],
                            index=df.index,
                        )

                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(df['sentiment'].value_counts().reset_index())
                        with col2:
                            sentiment_counts = df['sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig_sentiment = px.bar(
                                sentiment_counts, x='Sentiment', y='Count',
                                title='Sentiment Count Distribution', text='Count', color='Sentiment'
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)

                    st.write("Dataframe Results:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")