import streamlit as st
import pandas as pd
import numpy as np
import random
from bertopic.representation import KeyBERTInspired, OpenAI, TextGeneration
import openai
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import ast  # To safely evaluate string input to list format
import hashlib  # To create unique identifiers
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.io as pio

st.set_page_config(
    page_title="Text2Topics: Unsupervised",
    layout="wide"
)

# Authenticate with Google Sheets API using Streamlit Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("TextViz Studio Feedback").sheet1

# Feedback form in the sidebar
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area("Experiencing bugs/issues? Have ideas to better the application tool?", placeholder="Leave feedback or error code here")

# Submit feedback
if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["Text2Topics: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("")

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.sidebar.markdown("")

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.2) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


# Sidebar: Title and description (sticky)
st.markdown("<h1 style='text-align: center'>Text2Topics: Large Language Unsupervised Topic Modeling</h1>", unsafe_allow_html=True)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
**Text2Topic Analysis** is an interactive tool for extracting and visualizing topics from text data.
Upload a CSV file containing your text data for analysis. Select topic generation
preferences define a specific number of topics or let the model determine the optimal
number. Choose advanced options like outlier reduction and the use of OpenAI's 
GPT-4o for improved topic labels. Visualize the results through a topic summary, 
intertopic distance map, and document-topic probabilities. Results can be downloaded 
for further analyses of topics. Configure the parameters to customize your topic 
modeling experience.
""")

st.markdown("")
st.markdown("")

# Initialize session state to keep the model and topics across reruns
if "BERTmodel" not in st.session_state:
    st.session_state.BERTmodel = None
    st.session_state.topics = None
    st.session_state.text_data = None
    st.session_state.doc_ids = None

configuration = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 1
    }
}

# Create header for the app
st.subheader("Import Data", divider=True)

# Upload CSV file containing text data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Topic Modeling Configuration", divider=True)

    # Reset previous session state
    if "BERTmodel" in st.session_state:
        del st.session_state["BERTmodel"]
    if "topics" in st.session_state:
        del st.session_state["topics"]
    if "topic_info" in st.session_state:
        del st.session_state["topic_info"]
        
    # Dropdown to select the 'text' column
    text_column = st.selectbox("Select the text column", options=data.columns, key="text_column")

    # Ensure selected column has valid data
    data = data.dropna(subset=[text_column])
    text_data = data[text_column]

    if len(text_data) == 0:
        st.error("No valid text data found. Please check your file.")
        st.stop()
    else:
        # Dropdown to choose topic modeling approach
        method = st.selectbox("Choose Topic Modeling Method", ["Unsupervised", "Zero-Shot"], key="method")
        
        if method == "Unsupervised":
            st.markdown("### Unsupervised Topic Modeling")
            # Example implementation for unsupervised topic modeling
            vectorizer = CountVectorizer(stop_words="english")
            embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(text_data.tolist())
            umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine").fit_transform(embeddings)
            topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", vectorizer_model=vectorizer)
            topics, probs = topic_model.fit_transform(text_data.tolist())
            st.session_state.BERTmodel = topic_model
            st.session_state.topics = topics
            st.write("Topics generated successfully!")
            
        elif method == "Zero-Shot":
            st.markdown("### Zero-Shot Topic Modeling")
            # Example implementation for zero-shot topic modeling
            openai.api_key = st.secrets["openai_api_key"]
            topic_labels = []
            for text in text_data:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"Generate a topic label for the following text: {text}",
                    max_tokens=10
                )
                topic_labels.append(response.choices[0].text.strip())
            st.session_state.topics = topic_labels
            st.write("Zero-shot topics generated successfully!")
        