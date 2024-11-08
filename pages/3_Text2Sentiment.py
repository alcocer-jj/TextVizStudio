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
from io import BytesIO, StringIO

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
        
st.sidebar.markdown("")
        
st.sidebar.markdown(
    "For full documentation and updates, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)

st.sidebar.markdown("")

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.0) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


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

# Create unique IDs for each text entry
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Function to extract text from CSV file and add unique identifiers (doc_id)
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    
    df.columns = df.columns.str.lower()

    if 'text' in df.columns:
        # Drop rows where the 'text' column is NaN
        df = df.dropna(subset=['text'])
        
        # Create unique doc_id for each text
        df['doc_id'] = df['text'].apply(create_unique_id)
        
        # Return the doc_id and text columns
        return df[['doc_id', 'text']].reset_index(drop=True), df
    else:
        st.error("The CSV file must contain a 'text' column.")
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

def analyze_nrc(text, emotion_dict):
    emotions = ['anger', 'fear', 'trust', 'joy', 'anticipation', 
                'disgust', 'surprise', 'sadness', 'negative', 'positive']
    emotion_counts = defaultdict(int)

    # Preprocess the input text
    words = re.findall(r'\b\w+\b', text.lower())

    # Match words with NRC data and accumulate counts
    for word in words:
        if word in emotion_dict:
            for emotion in emotions:
                emotion_counts[emotion] += emotion_dict[word][emotion]

    # Calculate positive and negative scores
    positive_score = emotion_counts['positive']
    negative_score = emotion_counts['negative']

    # Determine sentiment label
    sentiment = 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'

    return pd.Series([emotion_counts['anger'], emotion_counts['fear'], emotion_counts['trust'], 
                      emotion_counts['joy'], emotion_counts['anticipation'], emotion_counts['disgust'], 
                      emotion_counts['surprise'], emotion_counts['sadness'], negative_score, positive_score, sentiment])

# Set Plotly configuration for high-resolution and theme
config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 1
    }
}

st.subheader("Import Data", divider=True)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")

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

        # Model selection for sentiment analysis
        st.subheader("Set Model Parameters")
        
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["NRC Lexicon (Default)", "VADER", "Zero-shot Classifier"]
        )
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
            
        # Only allow language selection if NRC Lexicon is chosen
        if sentiment_method == "NRC Lexicon (Default)":
            language = st.selectbox(
                "Select Language for NRC Lexicon Analysis",
                ["English", "French", "Spanish", "Italian", "Portuguese", "Chinese (Traditional)",
                 "Chinese (Simplified)", "Arabic", "Turkish", "Korean"]
            )
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
            selected_language_code = language_codes[language]
            nrc_data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / selected_language_code)
            emotion_dict = defaultdict(lambda: defaultdict(int))
            for _, row in nrc_data.iterrows():
                word = row['word']
                if pd.notna(word):  # Skip NaN values
                    emotion = row['emotion']
                    emotion_dict[word][emotion] = row['condition']

        st.subheader("Analyze", divider=True)
        
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
                            st.plotly_chart(fig_emotions, use_container_width=True, config=config)
                            

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
                        st.plotly_chart(fig_sentiment, use_container_width=True, config=config)
                    
                    st.write("Sentiment Analysis Dataframe Results:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")

    
