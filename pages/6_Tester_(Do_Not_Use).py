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
    data = pd.read_csv(uploaded_file)
    st.subheader("Topic Modeling Configuration", divider=True)

    # Reset session state
    for key in ["BERTmodel", "topics", "topic_info"]:
        st.session_state.pop(key, None)

    text_column = st.selectbox("Select the text column", options=data.columns, key="text_column")
    data = data.dropna(subset=[text_column])
    text_data = data[text_column]

    if len(text_data) == 0:
        st.error("Error: No valid text data found. Please check your file.")
        st.stop()
    else:
        method = st.selectbox("Choose Topic Modeling Method", ["Unsupervised", "Zero-Shot"], key="method")

        if method == "Unsupervised":
            st.subheader("Unsupervised Topic Modeling", divider=True)

        elif method == "Zero-Shot":
            st.subheader("Zero-Shot Topic Modeling", divider=True)
            
            # User input for predefined topics
            predefined_topics_input = st.text_area("Enter predefined topics (comma-separated):", "")
            predefined_topics = [topic.strip() for topic in predefined_topics_input.split(',') if topic.strip()]
            zeroshot_topic_list = predefined_topics if predefined_topics else []
            
            if not zeroshot_topic_list:
                st.error("Error: No predefined topics entered. Please enter at least one topic.")
                st.stop()
            else:    
                # Parameters for Zero-Shot modeling
                zeroshot_min_similarity = st.slider("Set Minimum Similarity for Zero-Shot Topic Matching", 0.0, 1.0, 0.85)
                min_topic_size = st.number_input("Set Minimum Number of Topics", min_value=1, max_value=100, value=5, step=1)
                
                # Input field for UMAP random_state (user seed)
                umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
                if umap_random_state is None:
                    umap_random_state = random.randint(1, 10000)
                    st.write(f"No seed provided, using random seed: {umap_random_state}")
                else:
                    st.write(f"Using user-provided seed: {umap_random_state}")
                    
                # Option for OpenAI API use
                use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
                api_key = None
                if use_openai_option:
                    api_key = st.text_input("Enter your OpenAI API Key", type="password")
                
                run_zero_shot_btn = st.button("Run Zero-Shot Topic Model")
                
                if run_zero_shot_btn:
                    from transformers import pipeline
                    with st.spinner("Running Zero-Shot Topic Model..."):
                        try:
                            st.write("Initializing Sentence Transformer model...")
                            model = SentenceTransformer("thenlper/gte-small")
                            
                            st.write("Initializing UMAP model...")
                            umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)

                            # Initialize representation model
                            st.write("Initializing representation model...")
                            representation_model = {"Unique Keywords": KeyBERTInspired()}

                            # OpenAI topic labeling integration
                            if use_openai_option and api_key:
                                try:
                                    # Set up OpenAI client
                                    client = openai.OpenAI(api_key=api_key)                    
                                    label_prompt = """
                                    Given the topic described by the following keywords: [KEYWORDS],
                                    and the following representative documents: [DOCUMENTS],
                                    provide a short label and a concise description in the format:
                                    <label>; <description>
                                    """
                                    # OpenAI initialization
                                    openai_model = OpenAI(
                                        client=client,
                                        model="gpt-4o",
                                        prompt=label_prompt,
                                        chat=True,
                                        nr_docs=10,
                                        delay_in_seconds=3)
                                                
                                    representation_model["GPT Topic Label"] = openai_model
                                except Exception as e:
                                    st.error(f"Error: Failed to initialize OpenAI API: {e}")
                                    representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback

                            # Initialize BERTopic model with zero-shot topic list
                            st.write("Initializing BERTopic model...")
                            BERTmodel = BERTopic(
                                representation_model=representation_model,
                                umap_model=umap_model,
                                embedding_model=model,
                                min_topic_size=min_topic_size,
                                zeroshot_topic_list=zeroshot_topic_list,
                                zeroshot_min_similarity=zeroshot_min_similarity
                            )

                            st.write("Running BERTopic.fit_transform()...")
                            topics, _ = BERTmodel.fit_transform(text_data)

                            # Extract topic info
                            st.write("Extracting topic info...")
                            topic_info = BERTmodel.get_topic_info()

                            # Check if topics exist before running transform()
                            unique_topics = set(topics) - {-1}
                            if len(unique_topics) > 0:
                                st.write("Extracting document-topic probabilities...")
                                topic_docs = BERTmodel.get_document_info(text_data)
                                probabilities = BERTmodel.transform(text_data)
                                probabilities = pd.DataFrame({'Topic': probabilities[0], 'Probability': probabilities[1]})
                                topic_docs = pd.concat([topic_docs[['Document']], probabilities], axis=1)
                            else:
                                st.warning("Warning: No valid topics were found. Skipping probability calculation.")
                                topic_docs = pd.DataFrame()

                            # Display topic info and document-topic probabilities
                            topic_info_col, doc_prob_col = st.columns([1, 1])

                            with topic_info_col:
                                st.write("**Identified Topics:**")
                                st.dataframe(topic_info)

                            with doc_prob_col:
                                st.write("**Document-Topic Probabilities:**")
                                st.dataframe(topic_docs)

                        except Exception as e:
                            st.error(f"Error: An error occurred: {e}")        