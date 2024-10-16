import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from pathlib import Path


# Streamlit page configuration
st.set_page_config(page_title="Text2Sentiment", layout="wide")

nrc_path = Path(__file__).resolve().parent.parent / "data" / "NRC-emo-sent-EN.csv"

# Ensure NLTK resources are available
nltk.download('stopwords')

# Load NRC CSV into a DataFrame at the top for reuse
@st.cache_data
def load_nrc_csv():
    df = pd.read_csv(nrc_path).dropna(subset=["word"])
    df = df[df["condition"] == 1]  # Use only words associated with emotion
    return df

nrc_df = load_nrc_csv()

# Initialize stopwords list
STOPWORDS = set(stopwords.words('english'))

# Authenticate with Google Sheets API using Streamlit secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet for feedback
sheet = client.open("TextViz Studio Feedback").sheet1

# Sidebar Feedback Form
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area("Experiencing bugs/issues? Have ideas to improve the tool?", placeholder="Leave feedback or error code here")

if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("For full documentation, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

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

# VADER Sentiment Analysis
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

# Zero-shot classifier caching
@st.cache_resource
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")

# NRC-based Sentiment Analysis
def analyze_nrc_sentiment(text):
    # Preprocess the text: lowercase and remove stopwords
    words = [word for word in text.lower().split() if word not in STOPWORDS]
    positive_count = nrc_df[(nrc_df["word"].isin(words)) & (nrc_df["emotion"] == "positive")].shape[0]
    negative_count = nrc_df[(nrc_df["word"].isin(words)) & (nrc_df["emotion"] == "negative")].shape[0]

    # Determine sentiment
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# Sidebar file uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
df, original_csv = None, None

# Process the uploaded CSV file
if uploaded_file is not None:
    df, original_csv = extract_text_from_csv(uploaded_file)

    if df is not None:
        st.write("CSV file successfully processed.")

        # Load sentiment models
        vader = load_vader()
        zero_shot_classifier = load_zero_shot_classifier()

        # Model selection for sentiment analysis
        st.subheader("Set Model Parameters")
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["NRC Lexicon (Default): Best for predefined word-based sentiment in structured text",
                "VADER: Optimized for informal social media content",
                "Zero-shot Classifier: Flexible for dynamic topics without specific pre-training"],
            index=0  # Default to NRC Lexicon
        )

        with st.expander("Learn more about each model"):
            st.markdown("""
            ### NRC Lexicon (Default)
            - **Description:** A predefined word-association-based model that assigns sentiment (positive or negative) to words.
            - **Best Use Case:** Effective for structured text, such as survey responses, interviews, or reports, where keywords are indicative of sentiment.
            - **Limitations:** May miss context and nuances, as it only matches predefined words.

            ### VADER (Valence Aware Dictionary and sEntiment Reasoner)
            - **Description:** A model optimized for analyzing sentiment in informal text, such as social media posts, with support for emojis, slang, and negation handling.
            - **Best Use Case:** Ideal for tweets, reviews, and other short-form content where informal language is prevalent.
            - **Limitations:** Less effective for longer texts and complex narratives.

            ### Zero-shot Classifier
            - **Description:** A transformer-based model from Hugging Face that can classify text into any category without specific pre-training on those categories.
            - **Best Use Case:** Useful for emerging topics, dynamic content, or when the sentiment categories are not predefined.
            - **Limitations:** Requires more computational resources and an active internet connection to load the model.
            """)
            
            st.warning("⚠️ Note: VADER and Zero-shot Classifier perform best with **English** text.")

        # Analyze sentiment on button click
        if st.button("Analyze Sentiment"):
            try:
                with st.spinner("Running sentiment analysis..."):
                    if sentiment_method == "VADER":
                        df[['compound', 'sentiment', 'neg', 'neu', 'pos']] = df['text'].apply(
                            lambda x: pd.Series(vader.polarity_scores(x))
                        )
                    elif sentiment_method == "Zero-shot Classifier":
                        df[['sentiment', 'confidence']] = df['text'].apply(
                            lambda x: pd.Series(analyze_zero_shot(x))
                        )
                    elif sentiment_method == "NRC Lexicon (Default)":
                        df['sentiment'] = df['text'].apply(analyze_nrc_sentiment)

                    st.write("Sentiment Analysis Results:")
                    st.dataframe(df)

                    # Plot sentiment distribution
                    sentiment_counts = df['sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig = px.bar(
                        sentiment_counts, x='Sentiment', y='Count',
                        title='Sentiment Proportion', text='Count', color='Sentiment'
                    )
                    st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")
else:
    st.warning("Please upload a CSV file for analysis.")
