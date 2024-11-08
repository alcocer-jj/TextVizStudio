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
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)
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
    emotions = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
    emotion_counts = defaultdict(int)
    
    # Convert text to lowercase before analysis to match the lowercase dictionary
    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if word in emotion_dict:
            for emotion in emotions:
                emotion_counts[emotion] += emotion_dict[word][emotion]
    
    positive_score = emotion_counts['joy'] + emotion_counts['trust'] + emotion_counts['anticipation']
    negative_score = emotion_counts['anger'] + emotion_counts['fear'] + emotion_counts['disgust'] + emotion_counts['sadness']
    sentiment = 'positive' if positive_score > negative_score else 'negative' if negative_score > positive_score else 'neutral'
    
    return pd.Series([emotion_counts[e] for e in emotions] + [negative_score, positive_score, sentiment])

# Set Plotly configuration for high-resolution and theme
config = {
    'toImageButtonOptions': {
        'format': 'svg',
        'filename': 'custom_image',
        'height': 500,
        'width': 700,
        'scale': 3
    }
}

# Streamlit UI
st.subheader("Import Data", divider=True)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")
df, original_csv = None, None

if uploaded_file is not None:
    df, original_csv = extract_text_from_csv(uploaded_file)
    if df is not None:
        st.write("CSV file successfully processed.")
        vader = SentimentIntensityAnalyzer()
        
        try:
            zero_shot_classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-distilroberta-base")
        except Exception as e:
            st.error(f"Error loading Zero-shot classifier: {e}")

        st.subheader("Set Model Parameters")
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["NRC Lexicon (Default)", "VADER", "Zero-shot Classifier"]
        )
        
        with st.expander("Which model is right for me?"):
            st.markdown("""
            [**NRC Emotion Lexicon**](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm): Default choice for sentiment and emotion analysis.
            
            [**VADER**](https://github.com/cjhutto/vaderSentiment): Best for short, informal texts. Handles negation (e.g., "not happy").
            
            [**Zero-shot Classifier**](https://huggingface.co/cross-encoder/nli-distilroberta-base): Adapts to various tasks with better semantic context understanding.
            """)
            st.warning('NRC Lexicon supports multiple languages; VADER and Zero-shot only support English.', icon="⚠️")
        
        if sentiment_method == "NRC Lexicon (Default)":
            language = st.selectbox(
                "Select Language for NRC Lexicon Analysis",
                ["English", "French", "Spanish", "Italian", "Portuguese", "Chinese (Traditional)", "Chinese (Simplified)", "Arabic"]
            )
            language_codes = {
                "English": "english.csv", "French": "french.csv", "Spanish": "spanish.csv",
                "Italian": "italian.csv", "Portuguese": "portuguese.csv", "Chinese (Traditional)": "chinese_traditional.csv",
                "Chinese (Simplified)": "chinese_simplified.csv", "Arabic": "arabic.csv"
            }
            selected_language_code = language_codes[language]
            nrc_data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / selected_language_code)
            emotion_dict = defaultdict(lambda: defaultdict(int))
            for _, row in nrc_data.iterrows():
                word = row['word']
                if pd.notna(word):
                    emotion = row['emotion']
                    emotion_dict[word][emotion] = row['condition']

        st.subheader("Analyze", divider=True)
        
        # Checkbox and Analyze button
        toggle_proportion = st.checkbox("Display Proportions as Percentages")
        
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
                        df[['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness', 'negative', 'positive', 'sentiment']] = df['text'].apply(
                            lambda x: analyze_nrc(x, emotion_dict)
                        )

                        # Sentiment Data
                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']
                        
                        # Emotion Data
                        emotion_cols = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
                        emotion_counts = df[emotion_cols].sum().reset_index()
                        emotion_counts.columns = ['Emotion', 'Count']

                        # Apply proportion settings
                        if toggle_proportion:
                            sentiment_counts['Count'] = (sentiment_counts['Count'] / sentiment_counts['Count'].sum() * 100).round(2)
                            y_axis_label_sentiment = "Proportion (%)"
                            
                            emotion_counts['Count'] = (emotion_counts['Count'] / emotion_counts['Count'].sum() * 100).round(2)
                            y_axis_label_emotion = "Proportion (%)"
                        else:
                            y_axis_label_sentiment = "Count"
                            y_axis_label_emotion = "Count"

                        # Layout Columns (1:3)
                        col1, col2 = st.columns([1, 3])
                        
                        # Display Sentiment Analysis DataFrame and Plot
                        with col1:
                            st.markdown("### Sentiment Distribution")
                            st.dataframe(sentiment_counts, use_container_width=True)
                        
                        with col2:
                            fig_sentiment = px.bar(
                                sentiment_counts, 
                                x='Sentiment', y='Count', 
                                title='Sentiment Distribution', 
                                text='Count', color='Sentiment', 
                                labels={'Count': y_axis_label_sentiment}
                            )
                            fig_sentiment.update_layout(template="ggplot2")
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=config)
                        
                        # Display Emotion Analysis DataFrame and Plot
                        with col1:
                            st.markdown("### Emotion Distribution")
                            st.dataframe(emotion_counts, use_container_width=True)

                        with col2:
                            fig_emotions = px.bar(
                                emotion_counts, 
                                x='Emotion', y='Count', 
                                title='Emotion Distribution', 
                                text='Count', color='Emotion', 
                                labels={'Count': y_axis_label_emotion}
                            )
                            fig_emotions.update_layout(template="ggplot2")
                            st.plotly_chart(fig_emotions, use_container_width=True, config=config)

                st.write("Sentiment Analysis Dataframe Results:")
                st.dataframe(df[['doc_id', 'text', 'sentiment']], use_container_width=True)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")
