import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px

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
            st.success("Zero-shot model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading Zero-shot classifier: {e}")

        # Model selection for sentiment analysis
        st.subheader("Set Model Parameters")
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["VADER", "Zero-shot Classifier"]
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


                    col1, col2 = st.columns([1,1)
                    with col1:
                        st.write("Sentiment Analysis Results")
                        st.dataframe(df)

                    with col2:
                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        fig = px.bar(sentiment_counts, x='Sentiment', 'Count',
                                     title='Sentiment Proportions', text='Count', color='Sentiment'
                                    )
                        st.plotly_chart(fig)
                    
                    st.write("Sentiment Analysis Results:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")
else:
    st.warning("Please upload a CSV file for analysis.")
