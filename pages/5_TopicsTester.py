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
    page_title="Text2Topics: Zero-Shot",
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
        sheet.append_row(["Text2Topics (ZS): ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("")

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.sidebar.markdown("")

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.2) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


# Sidebar: Title and description (sticky)
st.markdown("<h1 style='text-align: center'>Text2Topics: Large Language Zero Shot Topic Modeling</h1>", unsafe_allow_html=True)

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
        
    # Dropdown to select the 'text' column
    text_column = st.selectbox(
            "Select the column to be designated as 'text' for the model",
            options=data.columns,
            key="text_column"
            )
        
    # Filter out rows with NaN values in the selected text column
    data = data.dropna(subset=[text_column])
        
    # Use `text_column` as the designated text column 
    text_data = data[text_column]
        
    # Input field for UMAP random_state (user seed)
    umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
    st.info("**Tip:** Using a seed number ensures that the results can be reproduced. Not providing a seed number results in a random one being generated.")
    
    # Option for OpenAI API use
    use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
    st.success("**Note:** OpenAI's GPT-4o can be used to generate topic labels based on the documents and keywords provided. You must provide an OpenAI API key to use this feature.")

    # Ask for OpenAI API key if user chooses to use OpenAI
    api_key = None
    if use_openai_option:
            api_key = st.text_input("Enter your OpenAI API Key", type="password")

    st.subheader("Analyze", divider=True)

    # User input for predefined topics
    predefined_topics_input = st.text_area("Enter predefined topics (comma-separated):", "")
    st.warning("**Instructions:** Enter the topics you want the model to detect in the text data. Separate them with commas, e.g., 'Politics, Technology, Environment'.")

    # Convert input string to list
    predefined_topics = [topic.strip() for topic in predefined_topics_input.split(',') if topic.strip()]

    # Ensure predefined topics are formatted correctly
    zeroshot_topic_list = predefined_topics if predefined_topics else []

    # Parameter for minimum similarity threshold in zero-shot topic modeling
    zeroshot_min_similarity = st.slider("Set Minimum Similarity for Zero-Shot Topic Matching", 0.0, 1.0, 0.85)
    st.info("**Tip:** Adjust this parameter to control how closely a document must match a predefined topic.")

    # Parameter for minimum number of topics
    min_topic_size = st.number_input("Set Minimum Number of Topics", min_value=1, max_value=100, value=5, step=1)
    st.info("**Tip:** Adjust this to control the minimum number of topics the model will generate.")

    if not predefined_topics:
        st.error("Please enter at least one predefined topic.")
    else:
        run_zero_shot_btn = st.button("Run Zero-Shot Topic Model")

        if run_zero_shot_btn:
            with st.spinner("Running Zero-Shot Topic Model..."):
                try:
                    # Generate a random seed if not provided
                    if umap_random_state is None:
                        umap_random_state = random.randint(1, 10000)
                        st.write(f"No seed provided, using random seed: {umap_random_state}")
                    else:
                        st.write(f"Using user-provided seed: {umap_random_state}")

                    # Initialize submodels
                    model = SentenceTransformer("thenlper/gte-small")
                    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)

                    # Representation model
                    representation_model = {"Unique Keywords": KeyBERTInspired()}

                    # OpenAI topic labeling
                    if use_openai_option and api_key:
                        try:
                            client = openai.OpenAI(api_key=api_key)
                            label_prompt = """
                                Given the topic described by the following keywords: [KEYWORDS],
                                and the following representative documents: [DOCUMENTS],
                                provide a short label and a concise description in the format:
                                <label>; <description>
                                """
                            openai_model = OpenAI(client=client, model="gpt-4o", prompt=label_prompt, chat=True, nr_docs=10, delay_in_seconds=3)
                            representation_model["GPT Topic Label"] = openai_model
                        except Exception as e:
                            st.error(f"Failed to initialize OpenAI API: {e}")
                            representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback
                        
                        # Fallback to Flan-T5 if OpenAI is not used
                    else:
                        representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback

                    # Initialize BERTopic model with zero-shot topic list
                    BERTmodel = BERTopic(
                            representation_model=representation_model,
                            umap_model=umap_model,
                            embedding_model=model,
                            min_topic_size=min_topic_size,
                            zeroshot_topic_list=zeroshot_topic_list,
                            zeroshot_min_similarity=zeroshot_min_similarity)

                    # Fit BERTopic with predefined topics
                    topics, _ = BERTmodel.fit_transform(text_data)
                        
                    # Extract topic info from zero shot
                    topic_info = BERTmodel.get_topic_info()
                        
                    unique_topics = set(topics) - {-1}

                    if len(unique_topics) < 3:
                        st.warning("The model detected fewer than 3 topics. Try refining the predefined topics or using more diverse data.")
                    else:
                        # Get topic info and document-topic probabilities
                        topic_docs = BERTmodel.get_document_info(text_data)
                        topic_docs = topic_docs[['Document']]
                        probabilities = BERTmodel.transform(text_data)
                        probabilities = pd.DataFrame({'Topic': probabilities[0], 'Probability': probabilities[1]})
                        topic_docs = pd.concat([topic_docs, probabilities], axis=1)
                            
                        # Display topic info and document-topic probabilities in another two-column layout below
                        topic_info_col, doc_prob_col = st.columns([1, 1])
    
                        with topic_info_col:
                            st.write("Identified Topics:")
                            st.dataframe(topic_info)

                        with doc_prob_col:
                            st.write("Document-Topic Probabilities:")
                            st.dataframe(topic_docs)
                                
                except Exception as e:
                    st.error(f"An error occurred during Zero-Shot Topic Modeling: {e}")