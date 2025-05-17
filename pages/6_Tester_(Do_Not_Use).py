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
import ast
import hashlib
from transformers import pipeline
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import plotly.io as pio

# Unified Text2Topics App: Unsupervised & Zero-Shot
st.set_page_config(
    page_title="Text2Topics: Unified Topic Modeling",
    layout="wide"
)

# Authenticate with Google Sheets API
gscope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]
creds = ServiceAccountCredentials.from_json_keyfile_dict(
    st.secrets["gcp_service_account"], gscope
)
client = gspread.authorize(creds)
sheet = client.open("TextViz Studio Feedback").sheet1

# Sidebar: Method selection and feedback
method = st.sidebar.radio(
    "Select Topic Modeling Method", ["Unsupervised", "Zero-Shot"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area(
    "Experiencing bugs/issues? Have ideas to better the tool?",
    placeholder="Leave feedback or error code here"
)
if st.sidebar.button("Submit Feedback"):
    if feedback:
        prefix = "Text2Topics: Unsupervised" if method == "Unsupervised" else "Text2Topics: Zero-Shot"
        sheet.append_row([prefix, feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")
st.sidebar.markdown("---")
st.sidebar.markdown("For docs and updates: [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")
st.sidebar.markdown("Citation: Alcocer, J. J. (2024). TextViz Studio (Version 1.2) [Software]. Retrieved from https://textvizstudio.streamlit.app/")

# Main Title\ nst.markdown("<h1 style='text-align: center'>Text2Topics: Unified Topic Modeling</h1>", unsafe_allow_html=True)

# Shared utility functions
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

configuration = { 'toImageButtonOptions': {'format':'png','filename':'custom_image','height':1000,'width':1400,'scale':1} }

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    if 'text' in df.columns:
        df = df.dropna(subset=['text'])
        df['doc_id'] = df['text'].apply(create_unique_id)
        return df[['doc_id','text']].reset_index(drop=True), df
    st.error("CSV must contain a 'text' column.")
    return None, None

def display_outputs(BERTmodel, text_data, doc_ids):
    topic_info_df = BERTmodel.get_topic_info()
    for c in ['Name','Representation']:
        if c in topic_info_df.columns:
            topic_info_df.drop(columns=[c], inplace=True)
    hierarchical = BERTmodel.hierarchical_topics(text_data)
    col1,col2 = st.columns([1,1])
    with col1:
        st.write("Topic Hierarchy:")
        st.plotly_chart(BERTmodel.visualize_hierarchy(hierarchical_topics=hierarchical), config=configuration)
    with col2:
        st.write("Intertopic Distance Map:")
        st.plotly_chart(BERTmodel.visualize_topics(), config=configuration)
    tcol,dcol = st.columns([1,1])
    with tcol:
        st.write("Identified Topics:")
        st.dataframe(topic_info_df)
    with dcol:
        st.write("Document-Topic Probabilities:")
        doc_info_df = BERTmodel.get_document_info(text_data)
        doc_info_df['doc_id'] = doc_ids['doc_id'].tolist()
        for c in ['Name','Top_n_words','Representative Docs','Representative_document']:
            if c in doc_info_df.columns: doc_info_df.drop(columns=[c], inplace=True)
        st.dataframe(doc_info_df)

def create_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    st.download_button(label=link_text, data=csv, file_name=filename)

# Unsupervised Branch
if method == "Unsupervised":
    # Header and Description
    st.markdown("<h1 style='text-align: center'>Text2Topics: Large Language Unsupervised Topic Modeling</h1>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown(
        """
**Text2Topic Analysis** is an interactive tool for extracting and visualizing topics from text data.
Upload a CSV file containing your text data for analysis... [rest of original description here]
        """
    )
    st.markdown("")
    st.markdown("")

    # Session state
    if "BERTmodel" not in st.session_state:
        st.session_state.BERTmodel=None; st.session_state.topics=None; st.session_state.text_data=None; st.session_state.doc_ids=None; st.session_state.original_csv_with_ids=None

    # Parameter section
    st.subheader("Set Model Parameters", divider=True)
    umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
    st.info("**Tip:** Using a seed ensures reproducibility. Omit for random.")
    language_option = st.selectbox("Select the language model:", ("English","Multilanguage"))
    language = "english" if language_option=="English" else "multilingual"
    topic_option = st.selectbox("Select number of topics:", ("Auto","Specific Number"))
    nr_topics = None if topic_option=="Auto" else st.number_input("Enter number of topics",min_value=1,step=1)
    reduce_outliers_option = st.checkbox("Apply Outlier Reduction?", value=True)
    st.success("**Note:** Outlier reduction reassigns topic -1 docs but may merge distinct topics.")
    if reduce_outliers_option:
        c_tf_idf_threshold = st.slider("Set c-TF-IDF Threshold for Outlier Reduction",0.0,1.0,0.1)
        st.info("**Tip:** Lower threshold reassigns more outliers.")
    use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
    st.success("**Note:** Provide API key to use GPT-4o for labels.")
    api_key = None
    if use_openai_option:
        api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # Analyze and merge UI
    st.subheader("Analyze", divider=True)
    topics_to_merge_input = st.text_input("Enter topic pairs to merge (optional):", "[]")
    st.warning("Provide list of lists, e.g. [[1,2],[3,4]].")
    run_col, merge_col = st.columns([2,1])
    with run_col:
        run_model_btn = st.button("Run Topic Model")
    with merge_col:
        merge_topics_btn = st.button("Merge Topics")

    # Run model
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"] )
    if uploaded_file is not None:
        df, original_csv = extract_text_from_csv(uploaded_file)
        if df is not None:
            text_data = df['text'].tolist(); doc_ids = df[['doc_id']]
            st.session_state.text_data=text_data; st.session_state.doc_ids=doc_ids; st.session_state.original_csv_with_ids=original_csv
            if run_model_btn:
                with st.spinner("Running topic model..."):
                    if umap_random_state is None:
                        umap_random_state=random.randint(1,10000); st.write(f"No seed provided, using random: {umap_random_state}")
                    else: st.write(f"Using seed: {umap_random_state}")
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    umap_model = UMAP(n_neighbors=10,n_components=5,min_dist=0.0,metric='cosine',random_state=umap_random_state)
                    vectorizer_model = CountVectorizer(stop_words='english',min_df=1,max_df=0.9,ngram_range=(1,3))
                    representation_model={"Unique Keywords": KeyBERTInspired()}
                    if use_openai_option and api_key:
                        try:
                            client_rep=openai.OpenAI(api_key=api_key)
                            prompt="I have a topic that is described by the following keywords: [KEYWORDS]..."
                            openai_model=OpenAIRep(client=client_rep,model="gpt-4o",prompt=prompt,chat=True,nr_docs=10,delay_in_seconds=3)
                            representation_model["GPT Topic Label"]=openai_model
                        except Exception as e:
                            st.error(f"Failed to init OpenAI: {e}")
                    else:
                        try:
                            generator=pipeline('text2text-generation',model='google/flan-t5-base')
                            representation_model["T2T Topic Label"]=TextGeneration(generator)
                        except Exception as e:
                            st.error(f"Failed to init T2T: {e}")
                    BERTmodel=BERTopic(representation_model=representation_model,umap_model=umap_model,embedding_model=model,vectorizer_model=vectorizer_model,top_n_words=10,nr_topics=nr_topics,language=language,calculate_probabilities=True,verbose=True)
                    topics, probs = BERTmodel.fit_transform(text_data)
                    st.session_state.BERTmodel=BERTmodel; st.session_state.topics=topics
                    unique_topics=set(topics)-{-1}
                    if len(unique_topics)<3: st.warning("Fewer than 3 topics generated.")
                    else:
                        if reduce_outliers_option:
                            new_t = BERTmodel.reduce_outliers(text_data, topics, strategy="c-tf-idf", threshold=c_tf_idf_threshold)
                            new_t = BERTmodel.reduce_outliers(text_data, new_t, strategy="distributions")
                            BERTmodel.update_topics(text_data, topics=new_t)
                            st.session_state.topics=new_t
                        display_outputs(BERTmodel, text_data, doc_ids)
                        create_download_link(original_csv, "original_csv_with_ids.csv", "Download CSV with IDs")
            # Merge topics
            if merge_topics_btn and st.session_state.BERTmodel is not None:
                try:
                    pairs = ast.literal_eval(topics_to_merge_input)
                    merged = st.session_state.BERTmodel.merge_topics(text_data, pairs)
                    st.session_state.BERTmodel.update_topics(text_data, topics=merged)
                    display_outputs(st.session_state.BERTmodel, text_data, doc_ids)
                except Exception as e:
                    st.error(f"Error merging topics: {e}")

# Zero-Shot Branch
elif method == "Zero-Shot":
    # Zero-Shot App Configuration
    st.set_page_config(page_title="Text2Topics: Zero Shot", layout="wide", initial_sidebar_state="auto")
    st.markdown("<h1 style='text-align: center'>Text2Topics: Large Language Zero Shot Topic Modeling</h1>", unsafe_allow_html=True)
    
    # Import Data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        
        st.subheader("Topic Modeling Configuration", divider=True)
        # Reset previous session state
        if "BERTmodel" in st.session_state:
            del st.session_state["BERTmodel"]
            del st.session_state["topics"]
        # Select text column
        text_column = st.selectbox("Select the text column", options=data.columns.tolist(), key="zs_text_column")
        data = data.dropna(subset=[text_column])
        text_data = data[text_column].astype(str).tolist()

        if len(text_data) == 0:
            st.error("❌ No valid text data found. Please check your file.")
            st.stop()

        # Predefined topics
        predefined_input = st.text_area("Enter predefined topics (comma-separated):", value="")
        predefined_topics = [t.strip() for t in predefined_input.split(',') if t.strip()]
        if not predefined_topics:
            st.error("❌ No predefined topics entered. Please enter at least one topic.")
            st.stop()

        # Parameters
        zeroshot_min_similarity = st.slider("Set Minimum Similarity for Zero-Shot Topic Matching", 0.0, 1.0, 0.85)
        min_topic_size = st.number_input("Set Minimum Number of Topics", min_value=1, max_value=100, value=5, step=1)
        umap_random_state = st.number_input("Enter a seed number for pseudorandomization (optional)", min_value=0, value=None, step=1)
        if umap_random_state is None:
            umap_random_state = random.randint(1, 10000)
            st.write(f"No seed provided, using random seed: {umap_random_state}")
        else:
            st.write(f"Using user-provided seed: {umap_random_state}")

        use_openai_option = st.checkbox("Use OpenAI's GPT-4o API for Topic Labels?")
        api_key = None
        if use_openai_option:
            api_key = st.text_input("Enter your OpenAI API Key", type="password")

        run_zero_shot_btn = st.button("Run Zero-Shot Topic Model")

        if run_zero_shot_btn:
            with st.spinner("Running Zero-Shot Topic Model..."):
                try:
                    st.write("✅ Initializing Sentence Transformer model")
                    model = SentenceTransformer("thenlper/gte-small")

                    st.write("✅ Initializing UMAP model")
                    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=umap_random_state)

                    st.write("✅ Initializing representation model")
                    representation_model = {"Unique Keywords": KeyBERTInspired()}

                    if use_openai_option and api_key:
                        try:
                            client_zs = openai.OpenAI(api_key=api_key)
                            label_prompt = """
                            Given the topic described by the following keywords: [KEYWORDS],
                            and the following representative documents: [DOCUMENTS],
                            provide a short label and a concise description in the format:
                            <label>; <description>
                            """
                            openai_model = OpenAIRep(
                                client=client_zs,
                                model="gpt-4o",
                                prompt=label_prompt,
                                chat=True,
                                nr_docs=10,
                                delay_in_seconds=3
                            )
                            representation_model["GPT Topic Label"] = openai_model
                        except Exception as e:
                            st.error(f"❌ Failed to initialize OpenAI API: {e}")
                            representation_model = {"Unique Keywords": KeyBERTInspired()}

                    st.write("✅ Initializing BERTopic model")
                    BERTmodel = BERTopic(
                        representation_model=representation_model,
                        umap_model=umap_model,
                        embedding_model=model,
                        min_topic_size=min_topic_size,
                        zeroshot_topic_list=predefined_topics,
                        zeroshot_min_similarity=zeroshot_min_similarity
                    )

                    st.write("✅ Running BERTopic.fit_transform()")
                    topics, _ = BERTmodel.fit_transform(text_data)

                    st.write("✅ Extracting topic info")
                    topic_info = BERTmodel.get_topic_info()

                    unique_topics = set(topics) - {-1}
                    if unique_topics:
                        st.write("✅ Extracting document-topic probabilities")
                        topic_docs = BERTmodel.get_document_info(text_data)
                        probs = BERTmodel.transform(text_data)
                        prob_df = pd.DataFrame({'Topic': probs[0], 'Probability': probs[1]})
                        topic_docs = pd.concat([topic_docs[['Document']], prob_df], axis=1)
                    else:
                        st.warning("⚠ No valid topics were found. Skipping probability calculation.")
                        topic_docs = pd.DataFrame()

                    topic_info_col, doc_prob_col = st.columns([1, 1])
                    with topic_info_col:
                        st.write("Identified Topics:")
                        st.dataframe(topic_info)
                    with doc_prob_col:
                        st.write("Document-Topic Probabilities:")
                        st.dataframe(topic_docs)

                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
