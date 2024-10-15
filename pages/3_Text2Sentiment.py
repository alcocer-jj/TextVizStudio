import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from transformers import pipeline
import hashlib 
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(
    page_title="Text2Sentiment",
    layout="wide"
)

# Authenticate with Google Sheets API using Streamlit Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("TextViz Studio Feedback").sheet1


# Feedback form in the sidebar
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area("Experiencing bugs/issues? Have ideas to better the application tool?", placeholder="Leave feedback or error code here")

# Submit feedback
if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("")

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.markdown("<h1 style='text-align: center'>Text2Sentiment: Classifying Sentiment and Emotion of Text</h1>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

# Function to create unique identifiers for each document
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

st.subheader("Import Data", divider=True)
# Right-hand column: The app functionality
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")

# Function to extract text from CSV file and add unique identifiers (doc_id)
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)  # Create unique doc_id for each text
        return df[['doc_id', 'text']].reset_index(drop=True), df
    else:
        st.error("The CSV file must contain a 'text' column.")
        return None, None

st.subheader("Set Model Parameters", divider=True)

# Initialize Sentiment Analyzers
vader = SentimentIntensityAnalyzer()
zero_shot_classifier = pipeline("zero-shot-classification", model="tasksource/deberta-small-long-nli")

# User Input Section
st.write("Enter text below to analyze its sentiment and emotions:")
user_input = st.text_area("Your Text", "")

st.title("Options")
sentiment_method = st.selectbox(
    "Choose Sentiment Analysis Method", ["VADER", "Zero-shot Classifier", "NRCLex"]
)
enable_emotion = st.checkbox("Enable Emotion Analysis")

# Sentiment Analysis Functions
def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    label = (
        "positive" if compound >= 0.05
        else "negative" if compound <= -0.05
        else "neutral"
    )
    return label, compound

def analyze_zero_shot(text):
    labels = ["positive", "negative", "neutral"]
    result = zero_shot_classifier(text, labels)
    return result["labels"][0], result["scores"][0]

def analyze_nrc_sentiment(text):
    emotions = NRCLex(text)
    pos_score = emotions.affect_frequencies.get("positive", 0.0)
    neg_score = emotions.affect_frequencies.get("negative", 0.0)
    return pos_score, neg_score

# Emotion Analysis Function
def analyze_emotion(text):
    emotions = NRCLex(text)
    emotion_scores = emotions.raw_emotion_scores
    return pd.DataFrame(emotion_scores.items(), columns=["Emotion", "Score"])

# Sentiment Analysis
if st.button("Analyze Sentiment"):
    if user_input:
        if sentiment_method == "VADER":
            label, score = analyze_vader(user_input)
            st.write(f"Sentiment: **{label}** (Score: {score})")
        elif sentiment_method == "Zero-shot Classifier":
            label, confidence = analyze_zero_shot(user_input)
            st.write(f"Sentiment: **{label}** (Confidence: {confidence:.2f})")
        elif sentiment_method == "NRCLex":
            pos, neg = analyze_nrc_sentiment(user_input)
            st.write(f"Positive Score: {pos:.2f}, Negative Score: {neg:.2f}")
    else:
        st.warning("Please enter text for analysis.")

# Optional Emotion Analysis
if enable_emotion:
    if user_input:
        emotion_df = analyze_emotion(user_input)
        st.write("Emotion Scores:")
        st.dataframe(emotion_df)
        st.bar_chart(emotion_df.set_index("Emotion"))
    else:
        st.warning("Please enter text for emotion analysis.")

