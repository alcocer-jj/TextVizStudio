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
import ast 
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
**Text2Topic Analysis** is an interactive Streamlit application that supports both
Unsupervised Topic Modeling and Zero-Shot Classification for analyzing text data.
Users can upload a CSV file containing textual content, select the appropriate column,
and choose between two analysis modes: generating topics automatically using BERTopic
or classifying text into predefined labels using zero-shot methods. The app offers
multilingual embedding support, customizable stop word filtering, and advanced options
like outlier reduction and reproducibility through random seed input. For enhanced
topic labeling, users can integrate OpenAI’s GPT-4o to generate concise and descriptive
labels. Visual outputs include a hierarchical topic tree, an intertopic distance map, and
document-topic probability tables. Topics can be manually merged post hoc to refine results,
and final outputs are available for download. The interface is designed to provide a
customizable and user-friendly experience for exploring themes within large text datasets.
""")

st.markdown("")
st.markdown("")

# Initialize session state to keep the model and topics across reruns
if "BERTmodel" not in st.session_state:
    st.session_state.BERTmodel = None
    st.session_state.topics = None
    st.session_state.text_data = None

# Configuration for Plotly chart download
configuration = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'plotly_image',
        'height': 800,
        'width': 1200,
        'scale': 2  # Higher resolution image
    }
}

# Create header for the app
st.subheader("Import Data")

# Track file uploads and session state
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None

# File uploader for CSV files
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Clear session state if file is removed
if uploaded_file is None and st.session_state.last_file_hash is not None:
    st.session_state.last_file_hash = None
    for key in ["BERTmodel", "topics", "topic_info", "text_data"]:
        st.session_state.pop(key, None)

# Begin app logic
if uploaded_file:
    file_hash = hash(uploaded_file.getvalue())
    if st.session_state.last_file_hash != file_hash:
        st.session_state.last_file_hash = file_hash
        for key in ["BERTmodel", "topics", "topic_info", "text_data"]:
            st.session_state.pop(key, None)

    # Load the CSV file
    data = pd.read_csv(uploaded_file)
    st.subheader("Topic Modeling Configuration")

    # Load proper text column
    column_options = [""] + list(data.columns)
    text_column = st.selectbox("Select the text column", options=column_options, key="text_column")
    st.info("📝 Rows with missing or empty text values will be automatically excluded.")
    
    if text_column == "":
        st.warning("⚠️ Please select a valid text column to continue.")
        st.stop()
    
    data = data.dropna(subset=[text_column])
    text_data = data[text_column]

    # Save to session state for persistence
    st.session_state.text_data = text_data

    # Guard clause: stop early if no valid data
    if st.session_state.text_data is None or st.session_state.text_data.empty:
        st.error("Error: No valid text data found. Please check your file.")
        st.stop()

    else:
        # Choose topic modeling method
        method_selection = st.selectbox("Choose Topic Modeling Method", ["", "Unsupervised",
                                                                         "Zero-Shot"], key="method")
        st.info("📝 \n \n **Unsupervised**: Discover topics automatically from your data.\n- **Zero-Shot**: Use your own topic labels without training.\n\nChoose based on whether you want to explore unknown patterns or classify into known themes.")
        
        if method_selection == "":
            st.warning("⚠️ Please select a topic modeling method to continue.")
            st.stop()
        elif method_selection == "Unsupervised":
            method = "Unsupervised"
        elif method_selection == "Zero-Shot":
            method = "Zero-Shot"
        
        # Begin logic for unsupervised topic modeling
        if method == "Unsupervised":
            st.subheader("Unsupervised Topic Modeling")
            
            # Language selection dropdown
            language_option = st.selectbox(
                "Select the language model to use for topic modeling:",
                ("English", "Multilanguage"))

            language = "english" if language_option == "English" else "multilingual"

            if language == "multilingual":
                stop_word_language = st.selectbox(
                    "Select the stop word language for CountVectorizer via NLTK:",
                    ("none", "czech", "danish", "dutch", "estonian", "finnish", "french", "german", "greek",
                     "italian", "porwegian", "polish", "portuguese", "russian", "slovene", "spanish",
                     "swedish", "turkish"))
                st.success("**Note:** The stop words for CountVectorizer are set to the selected language. The embedding model handles more languages than these but the stop words are limited to the languages supported by NLTK.")
                
            # Select topic generation mode
            topic_option = st.selectbox(
                "Select how you want the number of topics to be handled:",
                ("Auto", "Specific Number"))

            # Default nr_topics value
            nr_topics = None if topic_option == "Auto" else st.number_input("Enter the number of topics you want to generate", min_value=1, step=1)

            # Option to apply outlier reduction
            reduce_outliers_option = st.checkbox("Apply Outlier Reduction?", value=True)
            st.success("**Note:** This process assigns documents that were initially classified as outliers (i.e., assigned to the topic -1), to more suitable existing topics. Reducing outliers can help improve the overall quality of the topics generated. However, it may also lead to the merging of topics that are semantically distinct, thus creating noise. Experiment with and without this option to see what works best for your case.")

            if reduce_outliers_option:
                c_tf_idf_threshold = st.slider("Set c-TF-IDF Threshold for Outlier Reduction", 0.0, 1.0, 0.3)
                st.info("**Tip:** You can set a threshold (between 0.0 and 1.0), which determines how strict or lenient the reassignment of outlier documents will be. A lower threshold (closer to 0.0) will reassign more outliers to topics, while a higher threshold (closer to 1.0) will reassign fewer documents.")

            # Input field for UMAP random_state (user seed)
            umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
            st.info("💡 Using a seed number ensures that the results can be reproduced. Not providing a seed number results in a random one being generated.")

            # Option for OpenAI API use
            use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
            st.success("**Note:** OpenAI's GPT-4o can be used to generate topic labels based on the documents and keywords provided. You must provide an OpenAI API key to use this feature.")

            # Ask for OpenAI API key if user chooses to use OpenAI
            api_key = None
            if use_openai_option:
                api_key = st.text_input("Enter your OpenAI API Key", type="password")
            run_model_btn = st.button("Run Unsupervised Topic Model")
            
            # Define function to display outputs (reused after both model fitting and topic merging)
            def display_unsupervised_outputs(BERTmodel, text_data):
                topic_info_df = BERTmodel.get_topic_info()
                columns_to_remove = ['Name', 'Representation']
                topic_info_df = topic_info_df.drop(columns=[col for col in columns_to_remove if col in topic_info_df.columns], errors='ignore')

                # Generate hierarchical topics from the model
                hierarchical_topics = BERTmodel.hierarchical_topics(text_data)
    
                # Visualize hierarchy and intertopic distance in a two-column layout
                hierarchy_col, map_col = st.columns([1, 1])
    
                with hierarchy_col:
                    st.write("**Topic Hierarchy:**")
                    hierarchy_fig = BERTmodel.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
                    hierarchy_fig.update_layout(width=1200, height=800, margin=dict(l=80, r=80, t=100, b=80))
                    st.plotly_chart(hierarchy_fig, config = configuration)
        
                with map_col:
                    st.write("**Intertopic Distance Map:**")
                    intertopic_map = BERTmodel.visualize_topics()
                    intertopic_map.update_layout(width=1200, height=800, margin=dict(l=80, r=80, t=100, b=80))
                    st.plotly_chart(intertopic_map, config = configuration)

                # Display topic info and document-topic probabilities in another two-column layout below
                topic_info_col, doc_prob_col = st.columns([1, 1])
    
                with topic_info_col:
                    st.write("**Identified Topics:**")
                    # Create two new columns from 'GPT Topic Label'
                    if 'GPT Topic Label' in topic_info_df.columns:
                        topic_info_df['GPT Topic Label'] = topic_info_df['GPT Topic Label'].astype(str)
                        topic_info_df['GPT Label'] = topic_info_df['GPT Topic Label'].str.split(';').str[0].str.strip()
                        topic_info_df['GPT Description'] = topic_info_df['GPT Topic Label'].str.split(';').str[1].str.strip()
                        # Remove unwanted characters from 'GPT Label' and 'GPT Description'
                        topic_info_df['GPT Label'] = topic_info_df['GPT Label'].str.replace(r"[\"'\[\]]", "", regex=True)
                        topic_info_df['GPT Description'] = topic_info_df['GPT Description'].str.replace(r"'\]$", "", regex=True)
                        topic_info_df = topic_info_df.drop(columns=['GPT Topic Label'])        
                    st.dataframe(topic_info_df)

                with doc_prob_col:
                    st.write("**Document-Topic Probabilities:**")
                    doc_info_df = BERTmodel.get_document_info(text_data)
        
                    # Drop unnecessary columns
                    columns_to_remove = ['Name', 'Top_n_words', 'Representative Docs', 'Representative_document',
                                         'Representation', 'Unique Keywords', 'GPT Topic Label', 'Representative_Docs']
                    doc_info_df = doc_info_df.drop(columns=[col for col in columns_to_remove if col in doc_info_df.columns], errors='ignore')
                    st.dataframe(doc_info_df)

            # Begin logic for running the model
            if run_model_btn:
                from transformers import pipeline
                with st.spinner("Running Unsupervised Topic Model..."):
                    try:
                        # Initialize Sentence Transformer model
                        st.write("Initializing Sentence Transformer model...")
                        if language == "english":
                            model = SentenceTransformer("all-MiniLM-L6-v2")
                        else:
                            model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                        
                        # Initialize UMAP model
                        st.write("Initializing UMAP model...")
                        if umap_random_state is None:
                            umap_random_state = random.randint(1, 10000)  # Random seed between 1 and 10000
                            st.write(f"No seed provided, using random seed: {umap_random_state}")
                        else:
                            st.write(f"Using user-provided seed: {umap_random_state}")
                        
                        umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)

                        # Initialize CountVectorizer
                        st.write("Initializing CountVectorizer...")
                        if language == "multilingual":
                            from nltk.corpus import stopwords
                            
                            if stop_word_language != "none":
                                stop_word_language = stopwords.words(stop_word_language)
                            else:
                                stop_word_language = None
                            
                            vectorizer_model = CountVectorizer(stop_words=stop_word_language,
                                                               min_df=1,
                                                               max_df=0.9,
                                                               ngram_range=(1, 3))
                        else:
                            vectorizer_model = CountVectorizer(stop_words='english',
                                                   min_df=1,
                                                   max_df=0.9,
                                                   ngram_range=(1, 3))

                        # Initialize representation model
                        st.write("Initializing representation model(s)...")
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
                                                      delay_in_seconds=3)
                        
                                # Add OpenAI to the representation model
                                representation_model["GPT Topic Label"] = openai_model
                            except Exception as e:
                                st.error(f"Failed to initialize OpenAI API: {e}")
                                representation_model = {"Unique Keywords": KeyBERTInspired()}  # Fallback to KeyBERT only
                        
                        # Initialize BERTopic model with the selected representation models
                        BERTmodel = BERTopic(
                            representation_model=representation_model,
                            umap_model=umap_model,
                            embedding_model=model,
                            vectorizer_model=vectorizer_model,
                            top_n_words=10,
                            nr_topics=nr_topics,
                            #language=language,
                            calculate_probabilities=True,
                            verbose=True
                            )

                        # Fit and transform the topic model
                        topics, probs = BERTmodel.fit_transform(text_data)
                        st.session_state.BERTmodel = BERTmodel
                        st.session_state.topics = topics

                        unique_topics = set(topics) - {-1}  # Remove outliers from unique topics

                        if len(unique_topics) < 1:
                            st.warning("The model generated fewer than 1 topic. This can happen if the data lacks diversity or is too homogeneous. "
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
                            st.subheader("Output", divider=True)
                            display_unsupervised_outputs(BERTmodel, text_data)
                                
                    except Exception as e:
                        st.error(f"Error: An error occurred: {e}")   
                       
            # Post hoc topic merging                            
            if ("BERTmodel" in st.session_state and
                "topics" in st.session_state and
                "text_data" in st.session_state):
                st.subheader("Post Hoc Topic Merging", divider=True)

                # Use a form to capture both the input and the submit button
                topics_to_merge_input = st.text_input("Enter topic pairs to merge (e.g. [[1, 2], [3, 4]]):",
                                                      "[]", key="merge_input")
                st.warning("**Instructions:** Provide a list of lists like `[[1, 2], [3, 4]]` to merge topics.")
                merge_topics_btn = st.button("Merge Topics")

                # Begin logic for merging topics
                if merge_topics_btn:
                    try:
                        topics_to_merge = ast.literal_eval(st.session_state["merge_input"])
                        if isinstance(topics_to_merge, list) and all(isinstance(pair, list) for pair in topics_to_merge):
                            merged_topics = st.session_state.BERTmodel.merge_topics(st.session_state.text_data, topics_to_merge)
                            st.success("Topics successfully merged!")
                        
                            st.session_state.BERTmodel.update_topics(st.session_state.text_data, topics=merged_topics)
                            st.session_state.topics = merged_topics

                            display_unsupervised_outputs(st.session_state.BERTmodel, st.session_state.text_data)
                        else:
                            st.error("Input must be a list of topic pairs, e.g., [[1, 2], [3, 4]]")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")
                
        # Begin logic for Zero-Shot topic modeling                    
        elif method == "Zero-Shot":
            st.subheader("Zero-Shot Topic Modeling", divider=True)
            
            # Language selection dropdown
            language_option = st.selectbox(
                "Select the language model to use for topic modeling:",
                ("English", "Multilanguage"))
            language = "english" if language_option == "English" else "multilingual"
            
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
                
                # Begin logic for running the Zero-Shot model
                if run_zero_shot_btn:
                    from transformers import pipeline
                    with st.spinner("Running Zero-Shot Topic Model..."):
                        try:
                            st.write("Initializing Sentence Transformer model...")
                            if language == "english":
                                model = SentenceTransformer("thenlper/gte-small")
                            else:
                                model = SentenceTransformer("intfloat/multilingual-e5-small")
                            
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
                            topic_info = pd.DataFrame(topic_info)

                            # Create two new columns from 'GPT Topic Label'
                            if 'GPT Topic Label' in topic_info.columns:
                                topic_info['GPT Topic Label'] = topic_info['GPT Topic Label'].astype(str)
                                topic_info['GPT Label'] = topic_info['GPT Topic Label'].str.split(';').str[0].str.strip()
                                topic_info['GPT Description'] = topic_info['GPT Topic Label'].str.split(';').str[1].str.strip()
                                # Remove unwanted characters from 'GPT Label' and 'GPT Description'
                                topic_info['GPT Label'] = topic_info['GPT Label'].str.replace(r"[\"'\[\]]", "", regex=True)
                                topic_info['GPT Description'] = topic_info['GPT Description'].str.replace(r"'\]$", "", regex=True)
                                topic_info = topic_info.drop(columns=['GPT Topic Label'])
                            
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
                            st.subheader("Output", divider=True)
                            topic_info_col, doc_prob_col = st.columns([1, 1])

                            with topic_info_col:
                                st.write("**Identified Topics:**")
                                st.dataframe(topic_info)

                            with doc_prob_col:
                                st.write("**Document-Topic Probabilities:**")
                                st.dataframe(topic_docs)

                        except Exception as e:
                            st.error(f"Error: An error occurred: {e}")        