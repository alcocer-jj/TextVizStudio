import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px  # New: Import Plotly Express

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

# Sidebar Feedback Form
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area(
    "Experiencing bugs/issues? Have ideas to improve the tool?",
    placeholder="Leave feedback or error code here"
)

# Submit Feedback Button
if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

# Sidebar Documentation Link
st.sidebar.markdown(
    "For full documentation and updates, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)

# Page Title
st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment and Emotion Analysis</h1>", unsafe_allow_html=True)

# Function to Create Unique IDs for Documents
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# File Upload Section
st.subheader("Upload CSV Data", divider=True)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** Ensure the text data is in a column named 'text'.")

# Function to Extract Text from CSV
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)  # Unique ID for each text
        return df[['doc_id', 'text']].reset_index(drop=True)
    else:
        st.error("The CSV file must contain a 'text' column.")
        return None

# Load Data from Uploaded File
data = None
if uploaded_file:
    data = extract_text_from_csv(uploaded_file)

# Initialize Sentiment Analysis Tools
vader = SentimentIntensityAnalyzer()
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sentiment Method Selection
st.subheader("Set Model Parameters", divider=True)
sentiment_method = st.selectbox(
    "Choose Sentiment Analysis Method",
    ["VADER", "Zero-shot Classifier", "NRCLex"]
)
enable_emotion = st.checkbox("Enable Emotion Analysis")

# Sentiment Analysis Functions
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
    return result["labels"][0], result["scores"]

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

# Perform Sentiment and Emotion Analysis on All Rows
if st.button("Analyze Sentiment"):
    if data is not None:
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

        # Plot Sentiment Proportions using Plotly
        sentiment_counts = data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Proportion',
                     text='Count', color='Sentiment', barmode='group')
        st.plotly_chart(fig)

    else:
        st.warning("Please upload a CSV file for analysis.")

# Perform Emotion Analysis (if enabled)
if enable_emotion:
    if data is not None:
        emotion_data = data['text'].apply(analyze_emotion).apply(pd.Series)
        data = pd.concat([data, emotion_data], axis=1)

        st.write("Emotion Analysis Results:")
        st.dataframe(data)

        # Plot Emotion Proportions using Plotly
        emotion_counts = emotion_data.sum().sort_values(ascending=False).reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        fig = px.bar(emotion_counts, x='Emotion', y='Count', title='Emotion Proportion',
                     text='Count', color='Emotion')
        st.plotly_chart(fig)
    else:
        st.warning("Please upload a CSV file for emotion analysis.")
