import io, hashlib
import streamlit as st
import pandas as pd
import numpy as np
import random
from bertopic.representation import KeyBERTInspired, OpenAI, TextGeneration, LiteLLM
import openai
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import ast 
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.io as pio
import time
from datetime import datetime
import chardet

st.set_page_config(
    page_title="Text2Topics",
    layout="wide"
)

# Authenticate with Google Sheets API using Streamlit Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], scope
)
client = gspread.authorize(creds)

# Try to open the Google Sheet safely
try:
    sheet = client.open("TextViz Studio Feedback").sheet1

    # Feedback form in the sidebar
    st.sidebar.markdown("### **Feedback**")
    feedback = st.sidebar.text_area(
        "Experiencing bugs/issues? Have ideas to better the application tool?",
        placeholder="Leave feedback or error code here"
    )

    if st.sidebar.button("Submit"):
        if feedback:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row(["Text2Topics:", feedback, timestamp])
                st.sidebar.success("✅ Thank you for your feedback!")
            except Exception as e:
                st.sidebar.error("⚠️ Failed to submit feedback.")
                st.sidebar.caption(f"Error: {e}")
        else:
            st.sidebar.error("⚠️ Feedback cannot be empty!")

except Exception as e:
    st.sidebar.error("⚠️ Could not load feedback form.")
    st.sidebar.caption(f"Details: {e}")


st.sidebar.markdown("")

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.sidebar.markdown("")

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
if "hierarchy_fig" not in st.session_state:
    st.session_state.hierarchy_fig = None
if "intertopic_fig" not in st.session_state:
    st.session_state.intertopic_fig = None
if "umap_random_state" not in st.session_state:
    st.session_state.umap_random_state = None

# Configuration for Plotly chart download
configuration = {
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'plotly_image',
        'height': 600,
        'width': 1000,
        'scale': 2
    }
}

# Track file uploads and session state
if "last_file_hash" not in st.session_state:
    st.session_state.last_file_hash = None

# File uploader for CSV files

st.subheader("Import Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if not uploaded_file:
    st.stop()

# Read raw bytes once
raw_bytes = uploaded_file.read()
uploaded_file.seek(0)

# Hash for caching
file_hash = hashlib.md5(raw_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def load_csv_from_bytes(raw_bytes: bytes, file_hash: str):
    # 1) Detect (best-effort)
    det = chardet.detect(raw_bytes) or {}
    detected = (det.get("encoding") or "").lower()

    # 2) Try candidates in order
    candidates = [
        "utf-8",          # common default
        "utf-8-sig",      # handles BOM
        detected if detected else None,
        "cp1252",         # Windows Western
        "latin1",         # ISO-8859-1
        "utf-16", "utf-16le", "utf-16be"
    ]
    tried = set()
    last_err = None

    for enc in [c for c in candidates if c and c not in tried]:
        tried.add(enc)
        try:
            # Use StringIO when passing text; but let pandas handle bytes + encoding directly
            with io.BytesIO(raw_bytes) as bio:
                df = pd.read_csv(
                    bio,
                    encoding=enc,
                    sep=None,           # sniff delimiter
                    engine="python",    # needed for sep=None
                )
            return df, enc, False  # False => no replacement
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # Non-encoding issues still bubble up later
            last_err = e
            continue

    # 3) Last-resort: decode with replacement to avoid crashing
    try:
        text = raw_bytes.decode("latin1", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
        return df, "latin1 (errors=replace)", True
    except Exception as e:
        # If we get here, surface the most informative error
        raise last_err or e

with st.spinner("Reading CSV..."):
    try:
        df, used_encoding, used_replacement = load_csv_from_bytes(raw_bytes, file_hash)
        info = st.empty()
        msg = f"**Detected/used encoding:** `{used_encoding}`"
        if used_replacement:
            msg += "  — encountered undecodable bytes; replaced invalid characters."
        info.info(msg)
        time.sleep(1.2)
        info.empty()

        st.subheader("Data Preview")
        st.dataframe(df.head(5))

    except Exception as e:
        st.error(f"**ERROR:** Failed to read the CSV file: {e}")

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
    data = df
    st.subheader("Topic Modeling Configuration")

    # Load proper text column
    column_options = [""] + list(data.columns)
    text_column = st.selectbox("Select the text column", options=column_options, key="text_column")
    st.info("**NOTE:** Rows with missing or empty text values will be automatically excluded.")
    
    if text_column == "":
        st.warning("**WARNING:** Please select a valid text column to continue.")
        st.stop()
    
    data = data.dropna(subset=[text_column])
    text_data = data[text_column]

    # Save to session state for persistence
    st.session_state.text_data = text_data

    # Guard clause: stop early if no valid data
    if st.session_state.text_data is None or st.session_state.text_data.empty:
        st.error("**ERROR:** No valid text data found. Please check your file.")
        st.stop()

    else:
        # Choose topic modeling method
        method_selection = st.selectbox("Choose Topic Modeling Method", ["", "Unsupervised",
                                                                         "Zero-Shot"], key="method")
        with st.expander("Which model is right for me?"):
            st.markdown("""
            [**Unsupervised**](https://maartengr.github.io/BERTopic/algorithm/algorithm.html):
            - **Best for:** Exploring unknown or unstructured data where no labels exist.
            - **Strengths:** Automatically discovers latent topic structures; great for insight discovery and exploratory analysis.  
            - **Limitations:** Topics may be harder to interpret or control; results depend on data quality and clustering.

            [**Zero-Shot**](https://maartengr.github.io/BERTopic/getting_started/zeroshot/zeroshot.html):
            - **Best for:** Classifying documents into predefined topics, especially when domain expertise or prior knowledge is available.
            - **Strengths:** No training needed; matches documents to known topics and can generate new topics for unmatched content — flexible across domains.
            - **Limitations:** If all documents match predefined topics, no new ones are created. If none match, it defaults to unsupervised topic modeling. Performance depends on how well the predefined topics cover the data.
            """)        
        
        if method_selection == "":
            st.warning("**WARNING:** Please select a topic modeling method to continue.")
            st.stop()
        elif method_selection == "Unsupervised":
            method = "Unsupervised"
        elif method_selection == "Zero-Shot":
            method = "Zero-Shot"
        
        # Begin logic for unsupervised topic modeling
        if method == "Unsupervised":
            st.subheader("Unsupervised Topic Modeling")
            
            # Input field for UMAP random_state (user seed)
            umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
            st.success("**TIP:** Using a seed number ensures that the results can be reproduced. Not providing a seed number results in a random one being generated.")
            
            # generate two columns for the layout
            param1, param2 = st.columns([1, 1])
            with param1:
                # Language selection dropdown
                language_option = st.selectbox("Select the language model to use for topic modeling:", ("English", "Multilanguage"))
                language = "english" if language_option == "English" else "multilingual"
                with st.expander("A Note on Language Selection"):
                    st.markdown("""
                                Text2Topics supports two main language options, each powered by a specialized sentence transformer:

                                [**English (`bge-small-en-v1.5`)**](https://huggingface.co/BAAI/bge-small-en-v1.5)  
                                - **Best for**: High-performance topic modeling on English-only datasets.  
                                - **Strengths**: State-of-the-art performance for its size class on the MTEB benchmark; trained with contrastive learning and hard negative mining for sharper semantic clustering. Significantly outperforms older MiniLM-based models on clustering and retrieval tasks while remaining lightweight (~127MB).  
                                - **Limitations**: Only supports English. Input texts longer than 512 tokens are truncated.

                                [**Multilingual (`multilingual-e5-small`)**](https://huggingface.co/intfloat/multilingual-e5-small)  
                                - **Best for**: Working with non-English or mixed-language datasets (supports 100+ languages).  
                                - **Strengths**: Continuously trained on a diverse mixture of multilingual datasets using contrastive learning. Delivers strong and consistent performance across retrieval, clustering, and semantic similarity tasks in both high- and low-resource languages. Also used as the embedding model in the Zero-Shot pipeline, ensuring consistency across both modeling approaches.  
                                - **Limitations**: Larger than the English model (~470MB) due to its broad multilingual vocabulary; performance may degrade on very low-resource languages.
                                """)
                if language == "multilingual":
                    stop_word_language = st.selectbox("Select the stop word language for CountVectorizer via NLTK:",
                        ("none", "czech", "danish", "dutch", "estonian", "finnish", "french", "german", "greek",
                        "italian", "porwegian", "polish", "portuguese", "russian", "slovene", "spanish",
                        "swedish", "turkish"))
                    st.info("**NOTE:** The stop words for CountVectorizer are set to the selected language. The embedding model handles more languages than these but the stop words are limited to the languages supported by NLTK.")
                
                # Option to apply outlier reduction
                reduce_outliers_option = st.checkbox("Apply Outlier Reduction?", value=False)
                with st.expander("A Note on Outlier Reduction"):
                    st.markdown("""
                                This parameter controls whether the model assigns documents that were initially classified as outliers (i.e., assigned to the topic -1), to more suitable existing topics.
                                
                                Reducing outliers can help improve the overall quality of the topics generated. However, it may also lead to the merging of topics that are semantically distinct, thus creating noise.
                                
                                Experiment with and without this option to see what works best for your case.
                                """)
                if reduce_outliers_option:
                    c_tf_idf_threshold = st.slider("Set c-TF-IDF Threshold for Outlier Reduction", 0.0, 1.0, 0.3)
                    st.success("**TIP:** A lower threshold (closer to 0.0) will reassign more outliers to topics, while a higher threshold (closer to 1.0) will reassign fewer documents.")
                
            with param2:
                # Select topic generation mode
                topic_option = st.selectbox("Select how you want the number of topics to be handled:", ("Auto", "Specific Number"))
                # Default nr_topics value
                nr_topics = None if topic_option == "Auto" else st.number_input("Enter the number of topics you want to generate", min_value=1, step=1)
                with st.expander("A Note on Topic Number Selection"):
                    st.markdown("""
                                This parameter controls how many topics you want after the model is trained.

                                **Use `"auto"`**:  
                                - Automatically reduces topics based on topic similarity. Internally, BERTopic uses clustering techniques like **HDBSCAN** to find a natural number of coherent topics.  
                                - Best for exploratory analysis when you're unsure how many topics to expect.

                                **Set a number (e.g., `20`)**:
                                - Reduces the number of discovered topics to that exact value. This is useful when you want a fixed number of topics for interpretability or downstream tasks.  
                                - Can be computationally expensive; each reduction step requires a new c-TF-IDF calculation.
                                """)
                    
                          # Option for LLM provider
                llm_provider = st.selectbox(
                    "Use an LLM for Enhanced Topic Labels?",
                    ["None", "OpenAI (GPT)", "Claude (Anthropic)"],
                    help=(
                        "**OpenAI (GPT)** — Uses OpenAI’s Chat Completions API. "
                        "Requires an OpenAI API key.\n\n"
                        "**Claude (Anthropic)** — Uses Anthropic’s Claude models via LiteLLM. "
                        "Requires an Anthropic API key from console.anthropic.com.\n\n"
                        "**None** — Skip LLM labeling. Topics labeled by c-TF-IDF keywords only."
                    )
                )

                api_key = None
                openai_model_choice = "gpt-5.4-mini"
                claude_model_choice = "claude-sonnet-4-6"

                if llm_provider == "OpenAI (GPT)":
                    api_key = st.text_input("Enter your OpenAI API Key", type="password")
                    openai_model_choice = st.selectbox(
                        "Select OpenAI model for topic labeling:",
                        options=["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.4-nano"],
                        index=0,
                        help=(
                            "**gpt-5.5** — Flagship. Highest capability, highest cost ($5/1M input · $30/1M output).\n\n"
                            "**gpt-5.4** — Previous flagship. Strong capability at lower cost.\n\n"
                            "**gpt-5.4-mini** — Fast and affordable. Best balance for most labeling tasks.\n\n"
                            "**gpt-5.4-nano** — Fastest and cheapest. Good for high-volume, simpler labels."
                        )
                    )
                elif llm_provider == "Claude (Anthropic)":
                    api_key = st.text_input("Enter your Anthropic API Key", type="password")
                    claude_model_choice = st.selectbox(
                        "Select Claude model for topic labeling:",
                        options=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-20250514"],
                        index=1,
                        help=(
                            "**claude-opus-4** — Most capable Claude model. Best for nuanced labels. Highest cost.\n\n"
                            "**claude-sonnet-4** — Balanced capability and cost. Recommended for most use cases.\n\n"
                            "**claude-haiku-4** — Fastest and most economical. Good for straightforward topic labels."
                        )
                    )

                with st.expander("A Note on using an LLM for Topic Labels"):
                    st.markdown("""
                                This option passes each topic’s representative keywords and documents to a large language model,
                                which generates a concise label and description for that topic.
                                
                                **OpenAI (GPT)** requires an API key from [platform.openai.com](https://platform.openai.com).
                                
                                **Claude (Anthropic)** requires an API key from [console.anthropic.com](https://console.anthropic.com).
                                Claude is integrated via BERTopic’s built-in LiteLLM backend — no extra setup needed.
                                
                                Higher-tier models produce more interpretable labels but consume more API credits.
                                Delays may occur due to rate limits or API latency.
                                """)
                    
            run_model_btn = st.button("Run Unsupervised Topic Model")
            
            # Define function to display outputs (reused after both model fitting and topic merging)
            def display_unsupervised_outputs(BERTmodel, text_data, umap_random_state=None):
                topic_info_df = BERTmodel.get_topic_info()
                columns_to_remove = ['Name', 'Representation']
                topic_info_df = topic_info_df.drop(
                    columns=[col for col in columns_to_remove if col in topic_info_df.columns],
                    errors='ignore'
                    )

                # --- Parse GPT labels if present ---
                if 'LLM Topic Label' in topic_info_df.columns:
                        topic_info_df['LLM Topic Label'] = topic_info_df['LLM Topic Label'].astype(str)
                        topic_info_df['LLM Label'] = topic_info_df['LLM Topic Label'].str.split(';').str[0].str.strip()
                        topic_info_df['LLM Description'] = topic_info_df['LLM Topic Label'].str.split(';').str[1].str.strip()
                        topic_info_df['LLM Label'] = topic_info_df['LLM Label'].str.replace(r"[\"'\[\]]", "", regex=True)
                        topic_info_df['LLM Description'] = topic_info_df['LLM Description'].str.replace(r"'\]$", "", regex=True)
                        topic_info_df = topic_info_df.drop(columns=['LLM Topic Label'])

                # --- Count valid (non-outlier) topics with documents ---
                non_outlier_topics = topic_info_df[
                        (topic_info_df['Topic'] >= 0) & (topic_info_df.get('Count', 0) > 0)
                        ]
                n_valid_topics = len(non_outlier_topics)

                # Layout for plots
                hierarchy_col, map_col = st.columns([1, 1])

                if n_valid_topics < 1:
                        st.warning(
                            "No non-outlier topics with assigned documents were found. "
                            "Skipping hierarchy and intertopic distance map."
                            )
                elif n_valid_topics == 1:
                        st.info(
                            "Only one non-outlier topic was found. "
                            "A topic hierarchy and intertopic distance map require at least two topics, "
                            "so those visualizations are skipped."
                            )
                else:
                        # --- Only safe to call these when we have ≥ 2 topics ---
                        with hierarchy_col:
                            st.write("**Topic Hierarchy:**")
                            try:
                                if st.session_state.hierarchy_fig is None:
                                    hierarchical_topics = BERTmodel.hierarchical_topics(text_data)
                                    hierarchy_fig = BERTmodel.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
                                    hierarchy_fig.update_layout(
                                        width=1200, height=800,
                                        margin=dict(l=80, r=80, t=100, b=80)
                                    )
                                    st.session_state.hierarchy_fig = hierarchy_fig
                                else:
                                    hierarchy_fig = st.session_state.hierarchy_fig
                                st.plotly_chart(hierarchy_fig, config=configuration)
                                hierarchy_html = hierarchy_fig.to_html(
                                    full_html=True, include_plotlyjs="cdn"
                                ).encode("utf-8")
                                st.download_button(
                                    label="Download Hierarchy Plot (HTML)",
                                    data=hierarchy_html,
                                    file_name="topic_hierarchy.html",
                                    mime="text/html",
                                    key="download_hierarchy"
                                )
                            except Exception as e:
                                st.warning(f"Could not generate topic hierarchy: {e}")

                        with map_col:
                            st.write("**Intertopic Distance Map:**")
                            try:
                                if st.session_state.intertopic_fig is None:
                                    if umap_random_state is not None:
                                        np.random.seed(umap_random_state)
                                    intertopic_map = BERTmodel.visualize_topics()
                                    intertopic_map.update_layout(
                                        width=1200, height=800,
                                        margin=dict(l=80, r=80, t=100, b=80)
                                    )
                                    st.session_state.intertopic_fig = intertopic_map
                                else:
                                    intertopic_map = st.session_state.intertopic_fig
                                st.plotly_chart(intertopic_map, config=configuration)
                                intertopic_html = intertopic_map.to_html(
                                    full_html=True, include_plotlyjs="cdn"
                                ).encode("utf-8")
                                st.download_button(
                                    label="Download Intertopic Distance Map (HTML)",
                                    data=intertopic_html,
                                    file_name="intertopic_distance_map.html",
                                    mime="text/html",
                                    key="download_intertopic"
                                )
                            except Exception as e:
                                st.warning(f"Could not generate intertopic distance map: {e}")

                # --- Tables (these are safe even with 0–1 topics) ---
                topic_info_col, doc_prob_col = st.columns([1, 1])

                with topic_info_col:
                        st.write("**Identified Topics:**")
                        st.dataframe(topic_info_df)

                with doc_prob_col:
                        st.write("**Document-Topic Probabilities:**")
                        doc_info_df = BERTmodel.get_document_info(text_data)
                        cols_to_remove = [
                            'Name', 'Top_n_words', 'Representative Docs', 'Representative_document',
                            'Representation', 'Unique Keywords', 'LLM Topic Label', 'Representative_Docs'
                        ]
                        doc_info_df = doc_info_df.drop(
                            columns=[c for c in cols_to_remove if c in doc_info_df.columns],
                            errors='ignore'
                        )
                        st.dataframe(doc_info_df)

                # --- Model download (pickle) ---
                st.write("**Download Topic Model:**")
                try:
                    import pickle as _pickle
                    _model_buffer = io.BytesIO()
                    _pickle.dump(BERTmodel, _model_buffer)
                    _model_buffer.seek(0)
                    st.download_button(
                        label="Download Topic Model (.pkl)",
                        data=_model_buffer,
                        file_name="bertopic_model.pkl",
                        mime="application/octet-stream",
                        key="download_model_pkl"
                    )
                    st.info(
                        "The .pkl file contains the full BERTopic model object including topic assignments, "
                        "embeddings, and cluster structure. "
                        "Load it in Python with: `import pickle; model = pickle.load(open('bertopic_model.pkl', 'rb'))`. "
                        "You can then call `model.transform(new_docs)` to apply it to new data."
                    )
                except Exception as _pkl_err:
                    try:
                        import pickle as _pickle, copy as _copy
                        _model_copy = _copy.copy(BERTmodel)
                        _model_copy.representation_model = None
                        _model_buffer = io.BytesIO()
                        _pickle.dump(_model_copy, _model_buffer)
                        _model_buffer.seek(0)
                        st.download_button(
                            label="Download Topic Model - Core Only (.pkl)",
                            data=_model_buffer,
                            file_name="bertopic_model_core.pkl",
                            mime="application/octet-stream",
                            key="download_model_pkl_core"
                        )
                        st.warning(
                            "The LLM representation model could not be pickled and was excluded. "
                            "The core topic structure, cluster assignments, and embeddings are fully intact."
                        )
                    except Exception as _pkl_err2:
                        st.warning(f"Could not generate model download: {_pkl_err2}")

            # Begin logic for running the model
            if run_model_btn:
                from transformers import pipeline
                progress = st.progress(0, text="Starting topic modeling...")
                try:
                    # Initialize Sentence Transformer model
                    progress.progress(10, text="Initializing and Loading Sentence Transformer model...")
                    if language == "english":
                        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                    else:
                        model = SentenceTransformer("intfloat/multilingual-e5-small")
                        
                    # Initialize UMAP model
                    progress.progress(25, text="Setting up dimensionality reduction...")
                    if umap_random_state is None:
                        umap_random_state = random.randint(1, 10000)  # Random seed between 1 and 10000
                        st.write(f"No seed provided, using random seed: {umap_random_state}")
                    else:
                        st.write(f"Using user-provided seed: {umap_random_state}")
                        
                    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)
                            
                    # Initialize CountVectorizer
                    progress.progress(40, text="Vectorizing documents...")
                    if language == "multilingual":
                        import nltk
                        from nltk.corpus import stopwords
                        # Ensure the stopwords corpus is available
                        try:
                            _ = stopwords.words("english")  # Trigger lookup
                        except LookupError:
                            nltk.download("stopwords")

                        if stop_word_language != "none":
                            if stop_word_language in stopwords.fileids():
                                st.write(f"Using stopwords for: {stop_word_language}")
                                stop_word_list = stopwords.words(stop_word_language)
                            else:
                                st.warning(f"**WARNING:** NLTK does not support stopwords for '{stop_word_language}'. Proceeding without stopwords.")
                                stop_word_list = None
                        else:
                            stop_word_list = None

                        vectorizer_model = CountVectorizer(
                            stop_words=stop_word_list,
                            min_df=1,
                            max_df=0.9,
                            ngram_range=(1, 3)
                            )
                    else:
                        vectorizer_model = CountVectorizer(
                            stop_words="english",
                            min_df=1,
                            max_df=0.9,
                            ngram_range=(1, 3)
                            )   
    
                    # Initialize representation model
                    progress.progress(55, text="Preparing topic representations...")
                    representation_model = {"Unique Keywords": KeyBERTInspired()}
                        
                    # LLM topic labeling: OpenAI or Claude
                    label_prompt = """
                        I have a topic that is described by the following keywords: [KEYWORDS]
                        In this topic, the following documents are a small but representative subset of all documents in the topic:
                        [DOCUMENTS]

                        Based on the information above, please give a short label and an informative description of this topic in the following format:
                        <label>; <description>
                        """

                    if llm_provider == "OpenAI (GPT)" and api_key:
                        try:
                            import os as _os
                            _os.environ["OPENAI_API_KEY"] = api_key
                            openai_llm = LiteLLM(
                                model=f"openai/{openai_model_choice}",
                                prompt=label_prompt,
                                delay_in_seconds=3
                            )
                            representation_model["LLM Topic Label"] = openai_llm
                        except Exception as e:
                            st.error(f"Failed to initialize OpenAI API: {e}")
                            representation_model = {"Unique Keywords": KeyBERTInspired()}

                    elif llm_provider == "Claude (Anthropic)" and api_key:
                        try:
                            import os as _os
                            _os.environ["ANTHROPIC_API_KEY"] = api_key
                            claude_model = LiteLLM(
                                model=f"anthropic/{claude_model_choice}",
                                prompt=label_prompt,
                                delay_in_seconds=2
                            )
                            representation_model["LLM Topic Label"] = claude_model
                        except Exception as e:
                            st.error(f"Failed to initialize Claude (Anthropic) API: {e}")
                            representation_model = {"Unique Keywords": KeyBERTInspired()}
                        
                    # Initialize BERTopic model with the selected representation models
                    progress.progress(70, text="Building BERTopic model...")
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
                    progress.progress(85, text="Fitting topic model...")
                    topics, probs = BERTmodel.fit_transform(text_data)
                    st.session_state.BERTmodel = BERTmodel
                    st.session_state.topics = topics
                    st.session_state.umap_random_state = umap_random_state
                    st.session_state.hierarchy_fig = None
                    st.session_state.intertopic_fig = None

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
                        progress.progress(100, text="Topic modeling complete!")
                        time.sleep(3)
                        progress.empty()
                        st.subheader("Output")
                        display_unsupervised_outputs(BERTmodel, text_data,
                            umap_random_state=st.session_state.get("umap_random_state"))
                                
                except Exception as e:
                        st.error(f"Error: An error occurred: {e}")   
                       
            # Post hoc topic merging                            
            if ("BERTmodel" in st.session_state and
                "topics" in st.session_state and
                "text_data" in st.session_state):
                st.subheader("Post Hoc Topic Merging")

                # Use a form to capture both the input and the submit button
                topics_to_merge_input = st.text_input("Enter topic pairs to merge (e.g. [[1, 2], [3, 4]]):",
                                                      "[]", key="merge_input")
                st.success("**TIP:** If you want to further combine topics that may be similar based on the model's output, you can specify the topic pairs to merge in the format [[1, 2], [3, 4]]. The first number in each pair is the topic to be merged into, and the second number is the topic to be merged.")
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
                            st.session_state.hierarchy_fig = None
                            st.session_state.intertopic_fig = None
    
                            st.subheader("Output")
                            display_unsupervised_outputs(st.session_state.BERTmodel, st.session_state.text_data,
                                umap_random_state=st.session_state.get("umap_random_state"))
                        else:
                            st.error("Input must be a list of topic pairs, e.g., [[1, 2], [3, 4]]")
                    except Exception as e:
                        st.error(f"Merge failed: {e}")

            # --- Persistent output: re-render on download button reruns ---
            if (st.session_state.get("BERTmodel") is not None
                    and st.session_state.get("text_data") is not None
                    and not run_model_btn
                    and not merge_topics_btn):
                st.subheader("Output")
                display_unsupervised_outputs(
                    st.session_state.BERTmodel,
                    st.session_state.text_data,
                    umap_random_state=st.session_state.get("umap_random_state")
                )
                
        # Begin logic for Zero-Shot topic modeling                    
        elif method == "Zero-Shot":
            st.subheader("Zero-Shot Topic Modeling")
            
            # Input field for UMAP random_state (user seed)
            umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
            st.success("**TIP:** Using a seed number ensures that the results can be reproduced. Not providing a seed number results in a random one being generated.")
            
            # Language selection dropdown
            language_option = st.selectbox(
                "Select the language model to use for topic modeling:",
                ("English", "Multilanguage"))
            language = "english" if language_option == "English" else "multilingual"
            with st.expander("A Note on Language Selection"):
                st.markdown("""
                                Text2Topics supports two main language options, each powered by a specialized sentence transformer:

                                [**English (`gte-small`)**](https://huggingface.co/thenlper/gte-small)  
                                - **Best for**: High-performance zero-shot and unsupervised topic modeling on English datasets.  
                                - **Strengths**: Trained on 800M+ web-sourced text pairs and fine-tuned on multi-task annotated datasets using multi-stage contrastive learning. Despite being lightweight (30M parameters), it outperforms larger models like E5 and OpenAI embeddings on several benchmarks (e.g., BEIR, MTEB).
                                - **Limitations**: Only supports English. Texts longer than 512 tokens are truncated due to its BERT-based architecture.
                                        	
                                [**Multilingual (`multilingual-e5-small`)**](https://huggingface.co/intfloat/multilingual-e5-small)  
                                - **Best for**: High-quality semantic search, topic modeling, and clustering across diverse languages.  
                                - **Strengths**: Trained on ~1 billion multilingual text pairs with contrastive learning and fine-tuned on supervised multilingual tasks. Supports over 100 languages. Competitive performance on MTEB and MIRACL benchmarks.
                                - **Limitations**: Slightly lower performance on English-only tasks compared to large English-specialized models, but provides robust multilingual generalization. Larger and slower than the English model.
                                """)
                                    
            predefined_topics_input = st.text_area("Enter predefined topics (comma-separated):", "")
            predefined_topics = [topic.strip() for topic in predefined_topics_input.split(',') if topic.strip()]
            zeroshot_topic_list = predefined_topics if predefined_topics else []
            
            if not zeroshot_topic_list:
                st.error("Error: No predefined topics entered. Please enter at least one topic.")
                st.stop()
            else:    
                # Parameters for Zero-Shot modeling
                zeroshot_min_similarity = st.slider("Set Minimum Similarity for Zero-Shot Topic Matching", 0.0, 1.0, 0.85)
                st.info("**NOTE:** This parameter controls how many documents are matched to zero-shot topics.")
                min_topic_size = st.number_input("Set Minimum Number of Topics", min_value=1, max_value=100, value=5, step=1)
                st.info("**NOTE:** This parameter sets the minimum number of documents required to form a topic Lower values create more (and smaller) topics, while higher values reduce topic count. If set too high, no topics may be formed at all.")
                st.success("**TIP:** For larger datasets (e.g., hundreds of thousands to millions of documents), increase min_topic_size well beyond the default of 10 — try values like 100 or 500 to avoid excessive micro-clustering. Experimentation is key.")
                
                # Option for LLM provider
                llm_provider = st.selectbox(
                    "Use an LLM for Enhanced Topic Labels?",
                    ["None", "OpenAI (GPT)", "Claude (Anthropic)"],
                    key="zs_llm_provider",
                    help=(
                        "**OpenAI (GPT)** — Uses OpenAI’s Chat Completions API. "
                        "Requires an OpenAI API key.\n\n"
                        "**Claude (Anthropic)** — Uses Anthropic’s Claude models via LiteLLM. "
                        "Requires an Anthropic API key from console.anthropic.com.\n\n"
                        "**None** — Skip LLM labeling. Topics labeled by c-TF-IDF keywords only."
                    )
                )

                api_key = None
                openai_model_choice = "gpt-5.4-mini"
                claude_model_choice = "claude-sonnet-4-6"

                if llm_provider == "OpenAI (GPT)":
                    api_key = st.text_input("Enter your OpenAI API Key", type="password", key="zs_oai_key")
                    openai_model_choice = st.selectbox(
                        "Select OpenAI model for topic labeling:",
                        options=["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-5.4-nano"],
                        index=0,
                        key="zs_oai_model",
                        help=(
                            "**gpt-5.5** — Flagship. Highest capability, highest cost ($5/1M input · $30/1M output).\n\n"
                            "**gpt-5.4** — Previous flagship. Strong capability at lower cost.\n\n"
                            "**gpt-5.4-mini** — Fast and affordable. Best balance for most labeling tasks.\n\n"
                            "**gpt-5.4-nano** — Fastest and cheapest. Good for high-volume, simpler labels."
                        )
                    )
                elif llm_provider == "Claude (Anthropic)":
                    api_key = st.text_input("Enter your Anthropic API Key", type="password", key="zs_ant_key")
                    claude_model_choice = st.selectbox(
                        "Select Claude model for topic labeling:",
                        options=["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-20250514"],
                        index=1,
                        key="zs_claude_model",
                        help=(
                            "**claude-opus-4** — Most capable Claude model. Best for nuanced labels. Highest cost.\n\n"
                            "**claude-sonnet-4** — Balanced capability and cost. Recommended for most use cases.\n\n"
                            "**claude-haiku-4** — Fastest and most economical. Good for straightforward topic labels."
                        )
                    )
                    
                run_zero_shot_btn = st.button("Run Zero-Shot Topic Model")
                
                # Begin logic for running the Zero-Shot model
                if run_zero_shot_btn:
                    from transformers import pipeline
                    progress = st.progress(0, text="Starting topic modeling...")
                    try:
                        progress.progress(10, text="Initializing and Loading Sentence Transformer model...")
                        if language == "english":
                            model = SentenceTransformer("thenlper/gte-small")
                        else:
                            model = SentenceTransformer("intfloat/multilingual-e5-small")
                            
                        progress.progress(25, text="Setting up dimensionality reduction...")
                        if umap_random_state is None:
                            umap_random_state = random.randint(1, 10000)  # Random seed between 1 and 10000
                            st.write(f"No seed provided, using random seed: {umap_random_state}")
                        else:
                            st.write(f"Using user-provided seed: {umap_random_state}")
                        umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)

                        # Initialize representation model
                        progress.progress(55, text="Preparing topic representations...")
                        representation_model = {"Unique Keywords": KeyBERTInspired()}

                        # LLM topic labeling: OpenAI or Claude
                        label_prompt = """
                            Given the topic described by the following keywords: [KEYWORDS],
                            and the following representative documents: [DOCUMENTS],
                            provide a short label and a concise description in the format:
                            <label>; <description>
                            """

                        if llm_provider == "OpenAI (GPT)" and api_key:
                            try:
                                import os as _os
                                _os.environ["OPENAI_API_KEY"] = api_key
                                openai_llm = LiteLLM(
                                    model=f"openai/{openai_model_choice}",
                                    prompt=label_prompt,
                                    delay_in_seconds=3
                                )
                                representation_model["LLM Topic Label"] = openai_llm
                            except Exception as e:
                                st.error(f"Error: Failed to initialize OpenAI API: {e}")
                                representation_model = {"Unique Keywords": KeyBERTInspired()}

                        elif llm_provider == "Claude (Anthropic)" and api_key:
                            try:
                                import os as _os
                                _os.environ["ANTHROPIC_API_KEY"] = api_key
                                claude_model = LiteLLM(
                                    model=f"anthropic/{claude_model_choice}",
                                    prompt=label_prompt,
                                    delay_in_seconds=2
                                )
                                representation_model["LLM Topic Label"] = claude_model
                            except Exception as e:
                                st.error(f"Error: Failed to initialize Claude (Anthropic) API: {e}")
                                representation_model = {"Unique Keywords": KeyBERTInspired()}

                        # Initialize BERTopic model with zero-shot topic list
                        progress.progress(70, text="Building BERTopic model...")
                        BERTmodel = BERTopic(
                                representation_model=representation_model,
                                umap_model=umap_model,
                                embedding_model=model,
                                min_topic_size=min_topic_size,
                                zeroshot_topic_list=zeroshot_topic_list,
                                zeroshot_min_similarity=zeroshot_min_similarity)

                        progress.progress(85, text="Fitting topic model...")
                        topics, _ = BERTmodel.fit_transform(text_data)

                        # Extract topic info
                        topic_info = BERTmodel.get_topic_info()
                        topic_info = pd.DataFrame(topic_info)

                        # Create two new columns from 'LLM Topic Label'
                        if 'LLM Topic Label' in topic_info.columns:
                            topic_info['LLM Topic Label'] = topic_info['LLM Topic Label'].astype(str)
                            topic_info['LLM Label'] = topic_info['LLM Topic Label'].str.split(';').str[0].str.strip()
                            topic_info['LLM Description'] = topic_info['LLM Topic Label'].str.split(';').str[1].str.strip()
                            # Remove unwanted characters from 'GPT Label' and 'GPT Description'
                            topic_info['LLM Label'] = topic_info['LLM Label'].str.replace(r"[\"'\[\]]", "", regex=True)
                            topic_info['LLM Description'] = topic_info['LLM Description'].str.replace(r"'\]$", "", regex=True)
                            topic_info = topic_info.drop(columns=['LLM Topic Label'])
                            
                        # Check if topics exist before running transform()
                        unique_topics = set(topics) - {-1}
                        if len(unique_topics) > 0:
                            topic_docs = BERTmodel.get_document_info(text_data)
                            probabilities = BERTmodel.transform(text_data)
                            probabilities = pd.DataFrame({'Topic': probabilities[0], 'Probability': probabilities[1]})
                            topic_docs = pd.concat([topic_docs[['Document']], probabilities], axis=1)
                        else:
                            st.warning("**Warning:** No valid topics were found. Skipping probability calculation.")
                            topic_docs = pd.DataFrame()

                        # Display topic info and document-topic probabilities
                        progress.progress(100, text="Topic modeling complete!")
                        time.sleep(3)
                        progress.empty()
                        st.subheader("Output")
                        topic_info_col, doc_prob_col = st.columns([1, 1])

                        with topic_info_col:
                                st.write("**Identified Topics:**")
                                st.dataframe(topic_info)

                        with doc_prob_col:
                                st.write("**Document-Topic Probabilities:**")
                                st.dataframe(topic_docs)

                    except Exception as e:
                        st.error(f"**ERROR:** An error occurred: {e}")        