import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import nltk
import time

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

st.set_page_config(
    page_title="Text2Sentiment",
    layout="wide"
)

# Authenticate with Google Sheets API using Streamlit Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet for feedback
sheet = client.open("TextViz Studio Feedback").sheet1

st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area(
    "Experiencing bugs/issues? Have ideas to improve the tool?",
    placeholder="Leave feedback or error code here"
)

if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown(
    "For full documentation and updates, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)

st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment and Emotion Analysis</h1>", unsafe_allow_html=True)

def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

st.subheader("Upload CSV Data", divider=True)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** Ensure the text data is in a column named 'text'.")

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)
        return df[['doc_id', 'text']].reset_index(drop=True)
    else:
        st.error("The CSV file must contain a 'text' column.")
        return None

data = None
if uploaded_file:
    data = extract_text_from_csv(uploaded_file)

vader = SentimentIntensityAnalyzer()

try:
    zero_shot_classifier = pipeline(
        "zero-shot-classification", 
        model="cross-encoder/nli-distilroberta-base"
    )
    st.success("Zero-shot model loaded successfully!")
except Exception as e:
    st.error(f"Error loading zero-shot classifier: {e}")

st.subheader("Set Model Parameters", divider=True)
sentiment_method = st.selectbox(
    "Choose Sentiment Analysis Method",
    ["VADER", "Zero-shot Classifier", "NRCLex"]
)
enable_emotion = st.checkbox("Enable Emotion Analysis")

def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    label = (
        "positive" if compound >= 0.05
        else "negative" if compound <= -0.05
        else "neutral"
    )
    return compound, label, scores['neg'], scores['neu'], scores['pos']

def analyze_zero_shot(text):
    labels = ["positive", "negative", "neutral"]
    result = zero_shot_classifier(text, labels)
    sentiment = result["labels"][0]
    confidence = result["scores"][0]
    return sentiment, confidence

def analyze_nrc_sentiment(text):
    emotions = NRCLex(text)
    pos_score = emotions.affect_frequencies.get("positive", 0.0)
    neg_score = emotions.affect_frequencies.get("negative", 0.0)
    label = (
        "positive" if pos_score > neg_score
        else "negative" if neg_score > pos_score
        else "neutral"
    )
    return label, pos_score, neg_score

def analyze_emotion(text):
    emotions = NRCLex(text)
    return emotions.raw_emotion_scores

if st.button("Analyze Sentiment"):
    if data is not None:
        try:
            with st.spinner("Running sentiment analysis..."):
                if sentiment_method == "VADER":
                    data[['compound', 'sentiment', 'neg', 'neu', 'pos']] = data['text'].apply(
                        lambda x: pd.Series(analyze_vader(x))
                    )
                elif sentiment_method == "Zero-shot Classifier":
                    data[['sentiment', 'confidence']] = data['text'].apply(
                        lambda x: pd.Series(analyze_zero_shot(x))
                    )
                elif sentiment_method == "NRCLex":
                    data[['sentiment', 'positive', 'negative']] = data['text'].apply(
                        lambda x: pd.Series(analyze_nrc_sentiment(x))
                    )

                st.write("Sentiment Analysis Results:")
                st.dataframe(data)

                sentiment_counts = data['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Proportion',
                             text='Count', color='Sentiment', barmode='group')
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error during analysis: {e}")

    else:
        st.warning("Please upload a CSV file for analysis.")

if enable_emotion:
    if data is not None:
        with st.spinner("Running emotion analysis..."):
            emotion_data = data['text'].apply(analyze_emotion).apply(pd.Series)
            data = pd.concat([data, emotion_data], axis=1)

            st.write("Emotion Analysis Results:")
            st.dataframe(data)

            emotion_counts = emotion_data.sum().sort_values(ascending=False).reset_index()
            emotion_counts.columns = ['Emotion', 'Count']
            fig = px.bar(emotion_counts, x='Emotion', y='Count', title='Emotion Proportion',
                         text='Count', color='Emotion')
            st.plotly_chart(fig)
    else:
        st.warning("Please upload a CSV file for emotion analysis.")
