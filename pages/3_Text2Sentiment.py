import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from PyPDF2 import PdfReader  # PDF reading library

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
feedback = st.sidebar.text_area(
    "Experiencing bugs/issues? Have ideas to improve the tool?",
    placeholder="Leave feedback or error code here"
)

# Submit feedback
if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("")
st.sidebar.markdown(
    "For full documentation and updates, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)

st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment and Emotion Analysis</h1>", unsafe_allow_html=True)

# Function to create unique IDs for each document
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

st.subheader("Import Data", divider=True)

# Upload file: Accept PDF or CSV files
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=["csv", "pdf"])

st.warning("**Instructions:** For CSV, ensure the text data is in a column named 'text'. PDFs will be converted to text.")

# Function to extract text from CSV
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)  # Unique ID for each text
        return df[['doc_id', 'text']].reset_index(drop=True), df
    else:
        st.error("The CSV file must contain a 'text' column.")
        return None, None

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text_data = []
    for page in reader.pages:
        text_data.append(page.extract_text())
    full_text = "\n".join(text_data)
    doc_id = create_unique_id(full_text)  # Generate a unique ID for the PDF content
    return pd.DataFrame([{"doc_id": doc_id, "text": full_text}])

# Load data from the uploaded file
data = None
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        data, _ = extract_text_from_csv(uploaded_file)
    elif uploaded_file.name.endswith(".pdf"):
        data = extract_text_from_pdf(uploaded_file)

if data is not None:
    st.dataframe(data)

st.subheader("Set Model Parameters", divider=True)

# Initialize sentiment analysis tools
vader = SentimentIntensityAnalyzer()
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Choose sentiment analysis method
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

# Perform sentiment analysis
if st.button("Analyze Sentiment"):
    text_to_analyze = user_input or (data["text"].iloc[0] if data is not None else "")
    if text_to_analyze:
        if sentiment_method == "VADER":
            label, score = analyze_vader(text_to_analyze)
            st.write(f"Sentiment: **{label}** (Score: {score})")
        elif sentiment_method == "Zero-shot Classifier":
            label, confidence = analyze_zero_shot(text_to_analyze)
            st.write(f"Sentiment: **{label}** (Confidence: {confidence:.2f})")
        elif sentiment_method == "NRCLex":
            pos, neg = analyze_nrc_sentiment(text_to_analyze)
            st.write(f"Positive Score: {pos:.2f}, Negative Score: {neg:.2f}")
    else:
        st.warning("Please enter text or upload a file for analysis.")

# Optional emotion analysis
if enable_emotion:
    if user_input or data is not None:
        text_to_analyze = user_input or data["text"].iloc[0]
        emotion_df = analyze_emotion(text_to_analyze)
        st.write("Emotion Scores:")
        st.dataframe(emotion_df)
        st.bar_chart(emotion_df.set_index("Emotion"))
    else:
        st.warning("Please enter text or upload a file for emotion analysis.")
