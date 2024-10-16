import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
from collections import defaultdict
import re
from pathlib import Path


# Set the Streamlit page configuration
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

# Sidebar: Feedback form
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

st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment Analysis</h1>", unsafe_allow_html=True)

# Create unique IDs for each text entry
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Extract text from uploaded CSV files
def extract_text_from_csv(file):
    try:
        df = pd.read_csv(file, low_memory=False)
        if 'text' not in df.columns:
            st.error("The CSV file must contain a 'text' column.")
            return None, None
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)
        return df[['doc_id', 'text']].reset_index(drop=True), df
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        return None, None

# VADER Sentiment Analysis Function
def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    label = (
        "positive" if compound >= 0.05
        else "negative" if compound <= -0.05
        else "neutral"
    )
    return compound, label, scores['neg'], scores['neu'], scores['pos']

# Zero-shot Sentiment Analysis Function
def analyze_zero_shot(text):
    labels = ["positive", "negative", "neutral"]
    result = zero_shot_classifier(text, labels)
    sentiment = result["labels"][0]
    confidence = result["scores"][0]
    return sentiment, confidence

# NRC Sentiment Analysis Function
def load_nrc_emotion_lexicon():
    # Load the NRC emotion lexicon data
    nrc_data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "NRC-emo-sent-EN.csv")

    emotion_dict = defaultdict(lambda: defaultdict(int))
    for _, row in nrc_data.iterrows():
        word = row['word']
        if pd.notna(word):  # Skip NaN values
            word = word.lower()
            emotion = row['emotion']
            emotion_dict[word][emotion] = row['condition']
    return emotion_dict

def analyze_nrc(text, emotion_dict):
    emotions = ['anger', 'fear', 'trust', 'joy', 'anticipation', 
                'disgust', 'surprise', 'sadness']
    emotion_counts = defaultdict(int)

    # Preprocess the input text
    words = re.findall(r'\b\w+\b', text.lower())

    # Match words with NRC data and accumulate counts
    for word in words:
        if word in emotion_dict:
            for emotion in emotions:
                emotion_counts[emotion] += emotion_dict[word][emotion]

    # Calculate positive and negative scores
    positive_score = emotion_counts['joy'] + emotion_counts['trust'] + emotion_counts['anticipation']
    negative_score = emotion_counts['anger'] + emotion_counts['fear'] + emotion_counts['disgust'] + emotion_counts['sadness']

    # Determine sentiment label
    sentiment = 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'

    return pd.Series([emotion_counts['anger'], emotion_counts['fear'], emotion_counts['trust'], 
                      emotion_counts['joy'], emotion_counts['anticipation'], emotion_counts['disgust'], 
                      emotion_counts['surprise'], emotion_counts['sadness'], negative_score, positive_score, sentiment])

# Sidebar file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
df, original_csv = None, None

# Process the uploaded CSV file
if uploaded_file is not None:
    df, original_csv = extract_text_from_csv(uploaded_file)

    if df is not None:
        st.write("CSV file successfully processed.")

        # Initialize VADER sentiment analyzer
        vader = SentimentIntensityAnalyzer()

        # Load the Zero-shot classification model
        try:
            zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-distilroberta-base"
            )
        except Exception as e:
            st.error(f"Error loading Zero-shot classifier: {e}")

        # Load the NRC Emotion Lexicon
        emotion_dict = load_nrc_emotion_lexicon()

        # Model selection for sentiment analysis
        st.subheader("Set Model Parameters")

            # Information about model selection
        with st.expander("Which model is right for me?"):
            st.markdown("""
            [**NRC Emotion Lexicon**](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm):
            - Default choice
            - Includes ability to analyze both sentiment and emotion analysis
            - Strengths: Provides granular emotional insights since it uses pre-defined dictionary of ~10,000 words
            - Limitations: Not best for understanding nuanced context-dependent text
            
            [**VADER** (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment):
            - Best for short, informal texts (e.g., social media, product reviews).
            - Strengths: Speed and ability to handle negation (e.g., "not happy").
            - Limitations: Less accurate for longer, more complex text.

            [**Zero-shot Classifier**](https://huggingface.co/cross-encoder/nli-distilroberta-base):
            - Best for general sentiment analysis across any domain.
            - Strengths: No need for pre-defined categories; adapts to various tasks, and has better understanding of semantic context 
            - Limitations: Slower due to it being a transformer-based model
            """)
            st.warning('Currently, only NRC Lexicon model handles multiple languages. The latter two only handle English text.', icon="⚠️")

        
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["NRC Lexicon (Default)", "VADER", "Zero-shot Classifier"]
        )

        # Analyze sentiment on button click
        if st.button("Analyze Sentiment"):
            try:
                with st.spinner("Running sentiment analysis..."):
                    if sentiment_method == "VADER":
                        df[['compound', 'sentiment', 'neg', 'neu', 'pos']] = df['text'].apply(
                            lambda x: pd.Series(analyze_vader(x))
                        )
                    elif sentiment_method == "Zero-shot Classifier":
                        df[['sentiment', 'confidence']] = df['text'].apply(
                            lambda x: pd.Series(analyze_zero_shot(x))
                        )
                    elif sentiment_method == "NRC Lexicon (Default)":
                        df[['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 
                            'sadness', 'negative', 'positive', 'sentiment']] = df['text'].apply(
                                lambda x: analyze_nrc(x, emotion_dict)
                            )

                        # Generate a bar chart for emotion counts
                        emotion_cols = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
                        emotion_counts = df[emotion_cols].sum().reset_index()
                        emotion_counts.columns = ['Emotion', 'Count']

                        st.subheader("Emotion Counts (NRC Lexicon)")
                        fig_emotions = px.bar(
                            emotion_counts, x='Emotion', y='Count',
                            title='Emotion Counts Distribution', text='Count', color='Emotion'
                        )

                        col1, col2 = st.columns([0.2,0.8])
                        with col1:
                            st.write("Emotion Counts Dataframe:")
                            st.dataframe(emotion_counts)

                        with col2:
                            st.plotly_chart(fig_emotions, use_container_width=True)
                            

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
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    st.write("Sentiment Analysis Dataframe Results:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")
else:
    st.warning("Please upload a CSV file for analysis.")
