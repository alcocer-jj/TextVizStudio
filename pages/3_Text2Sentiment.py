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
import stanza


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

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.1) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


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
        return df.reset_index(drop=True), df
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

def initialize_stanza_pipeline(language_code):
    stanza.download(language_code, processors='tokenize,lemma', verbose=False)
    return stanza.Pipeline(lang=language_code, processors='tokenize,lemma')

# Preprocess text with Stanza for tokenization and lemmatization
def preprocess_text_stanza(text, nlp_pipeline):
    doc = nlp_pipeline(text)
    # Lemmatize each word token in the text
    lemmatized_text = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
    return lemmatized_text

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
            ["Dictionary - NRC Lexicon (Default)", "Dictionary - VADER", "LLM - XLM-RoBERTa-Twitter-Sentiment"]
        )
            # Information about model selection
        with st.expander("Which model is right for me?"):
            st.markdown("""
            [**NRC Emotion Lexicon**](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm):
            - Default choice. Currently can handle 10 languages. If you want a specific language added, send an inquiry
            - Includes ability to analyze both sentiment and emotion analysis
            - Strengths: Provides granular emotional insights since it uses a pre-defined dictionary of ~10,000 words
            - Limitations: Less suited for nuanced, context-dependent entries

            [**VADER** (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment):
            - Best for short, informal texts (e.g., social media, product reviews)
            - Strengths: Fast processing, effective at handling negation (e.g., "not happy")
            - Limitations: Less accurate for longer, more complex text

            [**XLM-RoBERTa-Twitter-Sentiment**](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment):
            - Large Language Model fine-tuned on 198M Tweets across different eight languages (Arabic, English, French, German, Hindi, Italian, Portuguese, and Spanish)
            - Strengths: Can handle lexical diversity, nuances, and semantic context from tweets and short text across 65 languages (see [paper](https://arxiv.org/pdf/2104.12250) for more info)
            - Limitations: Slower processing time than dictionary methods; not good for long text entries unless broken down into chunks. 
            """)
        
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

                    elif sentiment_method == "Dictionary - NRC Lexicon (Default)":
                        stanza_language_codes = {
                                "English": "en",
                                "French": "fr",
                                "Spanish": "es",
                                "Italian": "it",
                                "Portuguese": "pt",
                                "Chinese (Traditional)": "zh",
                                "Chinese (Simplified)": "zh",
                                "Arabic": "ar",
                                "Turkish": "tr",
                                "Korean": "ko"
                            }
                        
                        # Load NRC data
                        selected_language_code = language_codes[language]
                        nrc_data = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / selected_language_code)
                        emotion_dict = defaultdict(lambda: defaultdict(int))
                        for _, row in nrc_data.iterrows():
                            word = row['word']
                            if pd.notna(word):
                                emotion = row['emotion']
                                emotion_dict[word][emotion] = row['condition']

                        # Initialize Stanza pipeline for the selected language
                        stanza_lang_code = stanza_language_codes.get(language)
                        nlp_pipeline = initialize_stanza_pipeline(stanza_lang_code)

                        df['text'] = df['text'].apply(lambda x: preprocess_text_stanza(x, nlp_pipeline))
                        df[['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 
                            'sadness', 'negative', 'positive', 'sentiment']] = df['text'].apply(lambda x: analyze_nrc(x, emotion_dict))

                        # Calculate emotion counts
                        emotion_cols = ['anger', 'fear', 'trust', 'joy', 'anticipation', 'disgust', 'surprise', 'sadness']
                        emotion_counts = df[emotion_cols].sum().reset_index()
                        emotion_counts.columns = ['Emotion', 'Count']

                        # Display emotion counts DataFrame and plot
                        st.subheader("Emotion Counts (NRC Lexicon)")
    
                        col1, col2 = st.columns([0.3, 0.7])
                        with col1:
                            st.write("Emotion Counts Dataframe:")
                            st.dataframe(emotion_counts)

                        with col2:
                            # Create and display the emotion distribution bar plot
                            fig_emotions = px.bar(
                                emotion_counts,
                                x='Emotion',
                                y='Count',
                                title='Emotion Counts Distribution',
                                text='Count',
                                color='Emotion'
                            )
                            st.plotly_chart(fig_emotions, use_container_width=True, config=configuration)

                        # Calculate sentiment counts
                        sentiment_counts = df['sentiment'].value_counts().reset_index()
                        sentiment_counts.columns = ['Sentiment', 'Count']

                        # Display sentiment counts DataFrame and plot
                        st.subheader("Sentiment Counts (NRC Lexicon)")
                        col1, col2 = st.columns([0.3, 0.7])
                        with col1:
                            st.write("Sentiment Counts Dataframe:")
                            st.dataframe(sentiment_counts)

                        with col2:
                            # Create and display the sentiment distribution bar plot
                            fig_sentiment = px.bar(
                            sentiment_counts,
                            x='Sentiment',
                            y='Count',
                            title='Sentiment Count Distribution',
                            text='Count',
                            color='Sentiment')
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

    
