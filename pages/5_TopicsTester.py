import streamlit as st
import pandas as pd
import numpy as np
import random
import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, OpenAI, TextGeneration
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import ast  # To safely evaluate string input to list format
import hashlib  # To create unique identifiers
from transformers import pipeline
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.io as pio


st.set_page_config(
    page_title="Text2Topics",
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
st.markdown("<h1 style='text-align: center'>Text2Topics: Large Language Topic Modeling</h1>", unsafe_allow_html=True)

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
    st.session_state.doc_ids = None  # To track document IDs
    st.session_state.original_csv_with_ids = None  # Store original CSV with doc_ids

# Function to create unique identifiers for each document
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Function to extract text from CSV file and add unique identifiers (doc_id)
def extract_text_from_csv(file):
    df = pd.read_csv(file)

    # Convert all column names to lowercase
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
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Model selection for topic modeling: Unsupervised or Zero-shot
    model_selection = st.selectbox("Select Topic Modeling Method", ["Unsupervised", "Zero-Shot"])

    # Begin Logic for Unsupervised Topic Modeling
    if model_selection == "Unsupervised":
        
        # Dropdown to select the 'text' column
        text_column = st.selectbox(
            "Select the column to be designated as 'text' for the model",
            options=data.columns,
            key="text_column"
            )
        
        # Use `text_column` as the designated text column
        text = data[text_column]

        st.subheader("Topic Modeling Configuration", divider=True)
        
        # Input field for UMAP random_state (user seed)
        umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
        st.info("**Tip:** Using a seed number ensures that the results can be reproduced. Not providing a seed number results in a random one being generated.")

        # Language selection dropdown
        language_option = st.selectbox(
            "Select the language model to use for topic modeling:",
            ("English", "Multilanguage")
            )

        # Set the language for BERTopic
        language = "english" if language_option == "English" else "multilingual"

        # Select topic generation mode
        topic_option = st.selectbox(
            "Select how you want the number of topics to be handled:",
            ("Auto", "Specific Number")
            )
        
        # Default nr_topics value
        nr_topics = None if topic_option == "Auto" else st.number_input("Enter the number of topics you want to generate", min_value=1, step=1)

        # Option to apply outlier reduction
        reduce_outliers_option = st.checkbox("Apply Outlier Reduction?", value=True)
        st.success("**Note:** This process assigns documents that were initially classified as outliers (i.e., assigned to the topic -1), to more suitable existing topics. Reducing outliers can help improve the overall quality of the topics generated. However, it may also lead to the merging of topics that are semantically distinct, thus creating noise. Experiment with and without this option to see what works best for your case.")

        if reduce_outliers_option:
            c_tf_idf_threshold = st.slider("Set c-TF-IDF Threshold for Outlier Reduction", 0.0, 1.0, 0.1)
            st.info("**Tip:** You can set a threshold (between 0.0 and 1.0), which determines how strict or lenient the reassignment of outlier documents will be. A lower threshold (closer to 0.0) will reassign more outliers to topics, while a higher threshold (closer to 1.0) will reassign fewer documents.")

        # Option for OpenAI API use
        use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
        st.success("**Note:** OpenAI's GPT-4o can be used to generate topic labels based on the documents and keywords provided. You must provide an OpenAI API key to use this feature.")





            
    # Begin Logic for Zero-Shot Topic Modeling
    elif model_selection == "Zero-Shot":