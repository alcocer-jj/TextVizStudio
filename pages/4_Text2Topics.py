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

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.1) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


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

st.subheader("Import Data", divider=True)
# Right-hand column: The app functionality
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
st.warning("**Instructions:** For CSV files, ensure that the text data is in a column named 'text'.")

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

st.subheader("Set Model Parameters", divider=True)

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

# Ask for OpenAI API key if user chooses to use OpenAI
api_key = None
if use_openai_option:
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

st.subheader("Analyze", divider=True)

# Get the topic pairs to merge
topics_to_merge_input = st.text_input("Enter topic pairs to merge (optional):", "[]")

st.warning("**Instructions:** Provide a list of lists with the topic pairs you want to merge. For example, `[[1, 2], [3, 4]]` will merge topics 1 and 2, and 3 and 4. This must be done after running the topic model.")

# Run the topic model button and merge button side by side
run_col, merge_col = st.columns([2, 1])
with run_col:
    run_model_btn = st.button("Run Topic Model")
with merge_col:
    merge_topics_btn = st.button("Merge Topics")

# Define function to display outputs (reused after both model fitting and topic merging)
def display_outputs(BERTmodel, text_data, doc_ids):
    # Fetch topic info and remove unnecessary columns if they exist
    topic_info_df = BERTmodel.get_topic_info()
    columns_to_remove = ['Name', 'Representation']
    topic_info_df = topic_info_df.drop(columns=[col for col in columns_to_remove if col in topic_info_df.columns], errors='ignore')

    # Generate hierarchical topics from the model
    hierarchical_topics = BERTmodel.hierarchical_topics(text_data)
    
    # Visualize hierarchy and intertopic distance in a two-column layout
    hierarchy_col, map_col = st.columns([1, 1])
    
    with hierarchy_col:
        st.write("Topic Hierarchy:")
        hierarchy_fig = BERTmodel.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
        st.plotly_chart(hierarchy_fig, config = configuration)
        
    with map_col:
        st.write("Intertopic Distance Map:")
        intertopic_map = BERTmodel.visualize_topics()
        st.plotly_chart(intertopic_map, config = configuration)

    # Display topic info and document-topic probabilities in another two-column layout below
    topic_info_col, doc_prob_col = st.columns([1, 1])
    
    with topic_info_col:
        st.write("Identified Topics:")
        st.dataframe(topic_info_df)

    with doc_prob_col:
        st.write("Document-Topic Probabilities:")
        # Get document info and add doc_id to facilitate merging later
        doc_info_df = BERTmodel.get_document_info(text_data)
        doc_info_df['doc_id'] = doc_ids['doc_id'].tolist()
        
        # Drop unnecessary columns
        columns_to_remove = ['Name', 'Top_n_words', 'Representative Docs', 'Representative_document']
        doc_info_df = doc_info_df.drop(columns=[col for col in columns_to_remove if col in doc_info_df.columns], errors='ignore')

        st.dataframe(doc_info_df)

# Function to create download link for DataFrame as CSV
def create_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    st.download_button(label=link_text, data=csv, file_name=filename)

# Run the topic model functionality
if uploaded_file is not None:
    # Ensure the uploaded file is CSV only
    st.write("CSV file uploaded.")
    df, original_csv = extract_text_from_csv(uploaded_file)
    text_data = df['text'].tolist()
    doc_ids = df[['doc_id']]  # Store doc_id for reference
    st.session_state.doc_ids = doc_ids  # Store doc_ids in session_state
    st.session_state.original_csv_with_ids = original_csv  # Store the original CSV with doc_ids

    # Proceed if text data was successfully extracted
    if text_data:
        st.session_state.text_data = text_data  # Store the text data for later use

        if run_model_btn:
            with st.spinner("Running topic model..."):

                # Generate a random seed if the user didn't provide one
                if umap_random_state is None:
                    umap_random_state = random.randint(1, 10000)  # Random seed between 1 and 10000
                    st.write(f"No seed provided, using random seed: {umap_random_state}")
                else:
                    st.write(f"Using user-provided seed: {umap_random_state}")

                # Initialize SentenceTransformer, UMAP, and CountVectorizer models
                model = SentenceTransformer("all-MiniLM-L6-v2")
                umap_model = UMAP(n_neighbors=10,
                                  n_components=5,
                                  min_dist=0.0,
                                  metric='cosine',
                                  random_state=umap_random_state)  # Use either the user-defined or random seed
                vectorizer_model = CountVectorizer(stop_words='english',
                                                   min_df=1,
                                                   max_df=0.9,
                                                   ngram_range=(1, 3))

                # Use KeyBERTInspired for keywords representation
                representation_model = {"Unique Keywords": KeyBERTInspired()}

                # Check if user wants to use OpenAI for topic labels    
                if use_openai_option and api_key:
                    try:
                        # Set up OpenAI API client
                        client = openai.OpenAI(api_key=api_key)

                        label_prompt = """
                        I have a topic that is described by the following keywords: [KEYWORDS]
                        In this topic, the following documents are a small but representative subset of all documents in the topic:
                        [DOCUMENTS]

                        Based on the information above, please give a short label and an informative description of this topic in the following format:
                        <label>; <description>
                        """
                        
                        # Create OpenAI representation model
                        openai_model = OpenAI(client=client, 
                                              model="gpt-4o",
                                              prompt=label_prompt,
                                              chat=True,
                                              nr_docs=10,
                                              delay_in_seconds=3
                                              )
                        
                        # Add OpenAI to the representation model
                        representation_model["GPT Topic Label"] = openai_model
                    except Exception as e:
                        st.error(f"Failed to initialize OpenAI API: {e}")
                        representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback to KeyBERT only
                else:
                    # Fallback to Hugging Face text2text generation (TextGeneration model)
                    try:
                        prompt = "I have a topic described by the following keywords: [KEYWORDS]. Based on the previous keywords, tell me in few words what is this topic about?"
                        generator = pipeline('text2text-generation', model='google/flan-t5-base')
                        text2text_model = TextGeneration(generator)
                        representation_model["T2T Topic Label"] = text2text_model
                    except Exception as e:
                        st.error(f"Failed to initialize Text2Text generation model: {e}")
                        representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback to KeyBERT only

                # Initialize BERTopic model with the selected representation models
                BERTmodel = BERTopic(
                    representation_model=representation_model,
                    umap_model=umap_model,
                    embedding_model=model,
                    vectorizer_model=vectorizer_model,
                    top_n_words=10,  # Set top_n_words to avoid issues
                    nr_topics=nr_topics,  # Use the chosen number of topics
                    language=language,  # Use selected language option (English or Multilanguage)
                    calculate_probabilities=True,
                    verbose=True
                )

                # Fit and transform the topic model
                topics, probs = BERTmodel.fit_transform(text_data)
                st.session_state.BERTmodel = BERTmodel  # Store the model in session state
                st.session_state.topics = topics  # Store topics in session state

                unique_topics = set(topics) - {-1}  # Remove outliers from unique topics

                if len(unique_topics) < 3:
                    st.warning("The model generated fewer than 3 topics. This can happen if the data lacks diversity or is too homogeneous. "
                                "Please try using a dataset with more variability in the text content.")
                else:
                    # Apply outlier reduction if the option was selected
                    if reduce_outliers_option:
                        # First, reduce outliers using the "c-tf-idf" strategy with the chosen threshold
                        new_topics = BERTmodel.reduce_outliers(text_data, topics, strategy="c-tf-idf", threshold=c_tf_idf_threshold)
                        # Then, reduce remaining outliers with the "distributions" strategy
                        new_topics = BERTmodel.reduce_outliers(text_data, new_topics, strategy="distributions")
                        st.write(f"Outliers reduced using c-TF-IDF threshold {c_tf_idf_threshold} and distributions strategy.")

                        # Update topic representations based on the new topics
                        BERTmodel.update_topics(text_data, topics=new_topics)
                        st.session_state.topics = new_topics
                        st.write("Topics and their representations have been updated based on the new outlier-free documents.")

                    # Display the outputs (topics table, intertopic map, probabilities)
                    display_outputs(BERTmodel, text_data, st.session_state.doc_ids)

                    # Provide download link for original CSV with unique IDs
                    st.write("Download your original CSV with unique document IDs:")
                    create_download_link(st.session_state.original_csv_with_ids, "original_csv_with_ids.csv", "Download CSV with IDs")
                    st.info("**Tip:** Download the CSV file to keep a record of the unique document IDs assigned to each text document. This will help you merge topics with the original documents later for further analysis.")

# Manual topic merge functionality
if merge_topics_btn and st.session_state.BERTmodel is not None and st.session_state.topics is not None:
    try:
        topics_to_merge = ast.literal_eval(topics_to_merge_input)  # Convert input to list

        # Ensure it's a list of lists
        if isinstance(topics_to_merge, list) and all(isinstance(pair, list) for pair in topics_to_merge):
            merged_topics = st.session_state.BERTmodel.merge_topics(st.session_state.text_data, topics_to_merge)
            st.success("Topics have been successfully merged!")

            # Update topic representations after merging
            st.session_state.BERTmodel.update_topics(st.session_state.text_data, topics=merged_topics)
            st.session_state.topics = merged_topics

            # Re-display the outputs (topics table, intertopic map, probabilities)
            display_outputs(st.session_state.BERTmodel, st.session_state.text_data, st.session_state.doc_ids)
        else:
            st.error("Invalid input. Please provide a list of lists in the format `[[1, 2], [3, 4]]`.")
    except Exception as e:
        st.error(f"An error occurred while merging topics: {e}")
