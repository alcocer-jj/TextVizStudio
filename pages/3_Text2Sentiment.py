import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from transformers import pipeline

# Initialize Sentiment Analyzers
vader = SentimentIntensityAnalyzer()
zero_shot_classifier = pipeline("zero-shot-classification", model="tasksource/deberta-small-long-nli")

# Page Layout
st.title("Sentiment & Emotion Analysis")

# User Input Section
st.write("Enter text below to analyze its sentiment and emotions:")
user_input = st.text_area("Your Text", "")

# Sidebar Options
st.sidebar.title("Options")
sentiment_method = st.sidebar.selectbox(
    "Choose Sentiment Analysis Method", ["VADER", "Zero-shot Classifier", "NRCLex"]
)
enable_emotion = st.sidebar.checkbox("Enable Emotion Analysis")

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

