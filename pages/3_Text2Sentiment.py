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

# Load NRC CSV into a DataFrame
@st.cache_data
def load_nrc_csv():
    df = pd.read_csv(nrc_path).dropna(subset=["word"])
    df = df[df["condition"] == 1]  # Filter for relevant words
    return df

nrc_df = load_nrc_csv()

# Initialize stopwords list
STOPWORDS = set(stopwords.words('english'))

# Authenticate with Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)
sheet = client.open("TextViz Studio Feedback").sheet1

# Sidebar: Feedback Form
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area("Experiencing bugs/issues? Have ideas to improve the tool?", placeholder="Leave feedback here")

if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Sentiment: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("For documentation, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

# Main Page Header
st.markdown("<h1 style='text-align: center'>Text2Sentiment: Sentiment Analysis</h1>", unsafe_allow_html=True)

# Unique ID generator
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Extract text from uploaded CSV
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

# VADER and Zero-shot model loaders
@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_resource
def load_zero_shot_classifier():
    return pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")

# NRC Lexicon Sentiment Analysis
def analyze_nrc_sentiment(text):
    words = [word for word in text.lower().split() if word not in STOPWORDS]
    positive_count = nrc_df[(nrc_df["word"].isin(words)) & (nrc_df["emotion"] == "positive")].shape[0]
    negative_count = nrc_df[(nrc_df["word"].isin(words)) & (nrc_df["emotion"] == "negative")].shape[0]

    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"

# File uploader in sidebar
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
df, original_csv = None, None

if uploaded_file is not None:
    df, original_csv = extract_text_from_csv(uploaded_file)

    if df is not None:
        st.write("CSV file successfully processed.")

        vader = load_vader()
        zero_shot_classifier = load_zero_shot_classifier()

        st.subheader("Set Model Parameters")
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            [
                "NRC Lexicon (Default): Best for structured text with predefined words",
                "VADER: Optimized for informal social media content",
                "Zero-shot Classifier: Flexible for dynamic topics"
            ],
            index=0  # Default to NRC Lexicon
        )

        with st.expander("Which model is best for me?"):
            st.markdown("""
            ### NRC Lexicon (Default)
            - **Description:** Word-based sentiment model for structured text.
            - **Use Case:** Surveys, interviews, or structured data.
            - **Limitation:** May miss nuance.

            ### VADER
            - **Description:** Best for social media with emojis and informal language.
            - **Use Case:** Tweets, reviews, short-form text.
            - **Limitation:** Less effective for long texts.

            ### Zero-shot Classifier
            - **Description:** Classifies text without pre-training for specific categories.
            - **Use Case:** Emerging topics, dynamic content.
            - **Limitation:** Requires more resources.
            """)

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

                    # Check if 'sentiment' column exists
                    if 'sentiment' not in df.columns:
                        st.error("Error: Sentiment analysis failed to generate the 'sentiment' column.")
                        st.stop()

                    # Calculate sentiment proportions
                    sentiment_counts = df['sentiment'].value_counts(normalize=True).reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Proportion']

                    # Create two columns for layout
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.write("Sentiment Proportions:")
                        st.dataframe(sentiment_counts)

                    with col2:
                        fig = px.bar(
                            sentiment_counts, x='Sentiment', y='Proportion',
                            title='Sentiment Proportion', text='Proportion', color='Sentiment'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Display the complete DataFrame at the bottom
                    st.write("Complete Sentiment Analysis Data:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")
else:
    st.warning("Please upload a CSV file for analysis.")
