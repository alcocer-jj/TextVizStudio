import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, XLMRobertaTokenizer, AutoConfig
import hashlib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.express as px
from collections import defaultdict
import re
from pathlib import Path
from io import BytesIO, StringIO
from scipy.special import softmax
import sentencepiece

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

# Function to preprocess text for VADER
def preprocess_text_VADER(text, lowercase=False, remove_urls=True):
    # Lowercase the text if needed
    if lowercase:
        text = text.lower()
    
    # Remove URLs and special characters
    if remove_urls:
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Remove special characters, keeping only alphanumerics and spaces
    
    return text

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

# Preprocess function to clean text for XLM
def preprocess_text_xlm(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Function to analyze sentiment with XLM-RoBERTa
def analyze_xlm(text):
    # Preprocess and encode text
    text = preprocess_text_xlm(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    
    # Get scores and apply softmax
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Assign scores to specific columns
    negative, neutral, positive = scores[0], scores[1], scores[2]
    
    # Determine label based on highest probability
    score_diff = abs(max(scores) - sorted(scores)[-2])
    if score_diff < 0.05:
        sentiment = "neutral"
    else:
        sentiment = config.id2label[np.argmax(scores)]
    
    return negative, neutral, positive, sentiment

# Function to analyze class labels with mDeBERTa
def analyze_mdeberta(text):
    output = classifier(text, candidate_labels, multi_label=multi_label)
    # Keep scores as decimals to three decimal places
    scores = {label: round(score, 3) for label, score in zip(output["labels"], output["scores"])}
    # Determine the top label(s) based on the highest probability
    top_label = output["labels"][0] if not multi_label else ", ".join([label for label in scores if scores[label] >= 0.5])
    return scores, top_label

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
configuration = {
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

        # Model selection for sentiment analysis
        st.subheader("Set Model Parameters")
        
        sentiment_method = st.selectbox(
            "Choose Sentiment Analysis Method",
            ["Dictionary - NRC Lexicon (Default)", "Dictionary - VADER", "LLM - XLM-RoBERTa-Twitter-Sentiment", "LLM - mDeBERTa-v3-xnli-multilingual"]
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
        if sentiment_method == "Dictionary - NRC Lexicon (Default)":
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

        # Only allow label input and multi-label checkbox if mDeBERTa-v3 Zero-Shot Classification is chosen
        if sentiment_method == "LLM - mDeBERTa-v3-xnli-multilingual":
            st.write("Enter candidate labels for zero-shot classification:")
            label_input = st.text_input("Labels (comma-separated)", "negative, neutral, positive")
            candidate_labels = [label.strip() for label in label_input.split(",")]

            # Checkbox for multi-label option
            multi_label = st.checkbox("Allow multiple labels?", value=False)
        
        st.subheader("Analyze", divider=True)
        
        # Analyze sentiment on button click
        if st.button("Analyze Sentiment"):
            try:
                with st.spinner("Running sentiment analysis..."):
                    if sentiment_method == "Dictionary - VADER":
                        # Initialize VADER sentiment analyzer
                        vader = SentimentIntensityAnalyzer()
                        df['text'] = df['text'].apply(preprocess_text_VADER)
                        df[['compound', 'sentiment', 'neg', 'neu', 'pos']] = df['text'].apply(
                            lambda x: pd.Series(analyze_vader(x))
                        )
                        
                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(df['sentiment'].value_counts().reset_index())

                        with col2:
                            sentiment_counts = df['sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='Count',
                                                   title='Sentiment Count Distribution', text='Count', color='Sentiment')
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)
                    
                    elif sentiment_method == "LLM - mDeBERTa-v3-xnli-multilingual":
                        # Initialize the zero-shot classification pipeline
                        classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")

                        # Apply zero-shot classification to each row in 'text' column
                        results = df['text'].apply(lambda x: analyze_mdeberta(x))
                        df[["scores", "label"]] = pd.DataFrame(results.tolist(), index=df.index)

                        # Expand each label’s score into its own column
                        for label in candidate_labels:
                            df[label] = df["scores"].apply(lambda x: x.get(label, 0))

                        # Remove the 'scores' column as it is redundant
                        df = df.drop(columns=["scores"])
                
                        # Calculate label proportions
                        if multi_label:
                            # Split multi-label entries and count each occurrence
                            label_counts = pd.Series(sum(df['label'].str.split(", "), [])).value_counts()
                        else:
                            # Count single-label entries
                            label_counts = df['label'].value_counts()

                        # Create a bar plot for label proportions
                        st.subheader("Label Proportions")
                        fig_labels = px.bar(label_counts.reset_index(), x='index', y='label',
                                            title='Proportion of Each Label in the Dataset',
                                            labels={'index': 'Label', 'label': 'Count'},
                                            text='label', color='index')

                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Label Counts Dataframe:")
                            st.dataframe(label_counts.reset_index().rename(columns={'index': 'Label', 'label': 'Count'}))

                        with col2:
                            st.plotly_chart(fig_labels, use_container_width=True)









                    elif sentiment_method == "NRC Lexicon (Default)":
                        # Apply NRC Lexicon analysis to each row in 'text' column
                        df[['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 
                            'sadness', 'negative', 'positive', 'sentiment']] = df['text'].apply(lambda x: analyze_nrc(x, emotion_dict))

                        # Calculate emotion counts
                        emotion_cols = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
                        emotion_counts = df[emotion_cols].sum().reset_index()
                        emotion_counts.columns = ['Emotion', 'Count']

                        # Display emotion counts DataFrame and plot
                        st.subheader("Emotion Counts (NRC Lexicon)")
    
                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Emotion Counts Dataframe:")
                            st.dataframe(emotion_counts)

                        with col2:
                            fig_emotions = px.bar(emotion_counts, x='Emotion', y='Count',title='Emotion Counts Distribution', text='Count', color='Emotion')
                            st.plotly_chart(fig_emotions, use_container_width=True, config=configuration)

                        # Calculate sentiment counts
                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']

                        # Display sentiment counts DataFrame and plot
                        st.subheader("Sentiment Counts (NRC Lexicon)")
    
                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(sentiment_counts)

                        with col2:
                            fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='Count',title='Sentiment Count Distribution', text='Count', color='Sentiment')
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)
                
                    elif sentiment_method == "LLM - XLM-RoBERTa-Twitter-Sentiment":
                        # Initialize the model
                        MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                        tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL)
                        config = AutoConfig.from_pretrained(MODEL)
                        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

                        # Set up id2label if not defined in the config
                        if not hasattr(config, 'id2label'):
                            config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

                        # Apply the XLM-R sentiment analysis function to each row in 'text' column
                        df[['negative', 'neutral', 'positive', 'sentiment']] = df['text'].apply(
                                lambda x: pd.Series(analyze_xlm(x))
                            )

                        col1, col2 = st.columns([0.2, 0.8])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(df['sentiment'].value_counts().reset_index())

                        with col2:
                            sentiment_counts = df['sentiment'].value_counts().reset_index()
                            sentiment_counts.columns = ['Sentiment', 'Count']
                            fig_sentiment = px.bar(sentiment_counts, x='Sentiment', y='Count',
                                                   title='Sentiment Count Distribution', text='Count', color='Sentiment')
                            st.plotly_chart(fig_sentiment, use_container_width=True, config=configuration)
                    
                    st.write("Dataframe Results:")
                    st.dataframe(df)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Failed to process the uploaded CSV file.")

    
