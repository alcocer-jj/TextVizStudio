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

# Unified Text2Topics App: Upload -> Column Select -> Method
st.set_page_config(page_title="Text2Topics: Unified Topic Modeling", layout="wide")

# Authenticate Google Sheets for feedback
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)
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

# Main Title & Description
st.markdown("<h1 style='text-align: center'>Text2Topics: Unified Topic Modeling</h1>", unsafe_allow_html=True)
st.markdown("Upload your dataset, select the text column, then choose a modeling method and configure options.")



# Helper Functions

def create_unique_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

config = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'custom_image',
        'height': 1000,
        'width': 1400,
        'scale': 1
    }
}

def extract_dataframe(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    return df


def create_download_link(df: pd.DataFrame, filename: str, link_text: str):
    csv = df.to_csv(index=False)
    st.download_button(label=link_text, data=csv, file_name=filename)


def display_unsupervised(model: BERTopic, text_data: list, doc_ids: pd.DataFrame, orig_df: pd.DataFrame):
    hier = model.hierarchical_topics(text_data)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Topic Hierarchy")
        st.plotly_chart(model.visualize_hierarchy(hierarchical_topics=hier), config=config)
    with col2:
        st.subheader("Intertopic Distance Map")
        st.plotly_chart(model.visualize_topics(), config=config)
    info_col, prob_col = st.columns(2)
    with info_col:
        st.subheader("Identified Topics")
        topic_info = model.get_topic_info()
        to_drop = [c for c in ['Name', 'Representation'] if c in topic_info.columns]
        st.dataframe(topic_info.drop(columns=to_drop, errors='ignore'))
    with prob_col:
        st.subheader("Document-Topic Probabilities")
        doc_info = model.get_document_info(text_data)
        doc_info['doc_id'] = doc_ids['doc_id'].tolist()
        drop_cols = [c for c in ['Name','Top_n_words','Representative Docs','Representative_document'] if c in doc_info.columns]
        st.dataframe(doc_info.drop(columns=drop_cols, errors='ignore'))
    create_download_link(orig_df, "original_with_ids.csv", "Download CSV with IDs")

# Application Flow
st.title("Text2Topics: Unified Topic Modeling")
st.write("1. Upload CSV → 2. Select text column → 3. Choose method & configure → 4. Run")

# Step 1: Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if not uploaded_file:
    st.stop()

# Step 2: Load & select text column
df = extract_dataframe(uploaded_file)
text_columns = df.select_dtypes(include=['object']).columns.tolist()
if not text_columns:
    st.error("No text columns found in the uploaded data.")
    st.stop()
text_col = st.selectbox("Select the text column", text_columns)
df = df.dropna(subset=[text_col])
if df.empty:
    st.error("Selected column contains no valid data.")
    st.stop()

# Prepare data for modeling
df['doc_id'] = df[text_col].astype(str).apply(create_unique_id)
text_data = df[text_col].astype(str).tolist()
doc_ids = df[['doc_id']]
orig_df = df.copy()

# Step 3: Method selection
gmethod = st.selectbox("Select modeling method", ["Unsupervised", "Zero-Shot"], index=0)

# Unsupervised Method
if gmethod == "Unsupervised":
    st.header("Unsupervised Configuration")
    seed = st.number_input("Seed for reproducibility (optional)", min_value=0, value=None, step=1)
    lang = st.selectbox("Language", ['English', 'Multilingual'])
    topic_mode = st.selectbox("Topic count handling", ['Auto', 'Specific Number'])
    nr_topics = None if topic_mode == 'Auto' else st.number_input("Number of topics", min_value=1, step=1)
    outlier_red = st.checkbox("Apply Outlier Reduction?", value=True)
    if outlier_red:
        threshold = st.slider("c-TF-IDF threshold for outliers", 0.0, 1.0, 0.1)
    use_oa = st.checkbox("Use GPT-4o for labels?")
    oa_key = st.text_input("OpenAI API Key", type='password') if use_oa else None
    if st.button("Run Unsupervised Model"):
        with st.spinner("Running unsupervised topic modeling..."):
            s = seed or random.randint(1, 10000)
            emb = SentenceTransformer('all-MiniLM-L6-v2')
            umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=s)
            vect = CountVectorizer(stop_words='english', min_df=1, max_df=0.9, ngram_range=(1,3))
            rep_model = {"Unique Keywords": KeyBERTInspired()}
            if use_oa and oa_key:
                try:
                    client = openai.OpenAI(api_key=oa_key)
                    prompt = "I have a topic described by keywords: [KEYWORDS]..."
                    rep_model["GPT Topic Label"] = OpenAIRep(client=client, model='gpt-4o', prompt=prompt, chat=True, nr_docs=10, delay_in_seconds=3)
                except Exception as e:
                    st.error(f"OpenAI init failed: {e}")
            else:
                try:
                    gen = pipeline('text2text-generation', model='google/flan-t5-base')
                    rep_model["T2T Topic Label"] = TextGeneration(gen)
                except Exception as e:
                    st.error(f"Text2Text init failed: {e}")
            model = BERTopic(
                representation_model=rep_model,
                umap_model=umap_model,
                embedding_model=emb,
                vectorizer_model=vect,
                top_n_words=10,
                nr_topics=nr_topics,
                language=lang.lower(),
                calculate_probabilities=True,
                verbose=True
            )
            topics, probs = model.fit_transform(text_data)
            if outlier_red and topics is not None:
                topics = model.reduce_outliers(text_data, topics, strategy='c-tf-idf', threshold=threshold)
                topics = model.reduce_outliers(text_data, topics, strategy='distributions')
                model.update_topics(text_data, topics=topics)
            display_unsupervised(model, text_data, doc_ids, orig_df)

# Zero-Shot Method
else:
    st.header("Zero-Shot Configuration")
    predefined = st.text_area("Predefined topics (comma-separated)")
    topics_list = [t.strip() for t in predefined.split(',') if t.strip()]
    if topics_list:
        sim = st.slider("Minimum similarity for zero-shot", 0.0, 1.0, 0.85)
        min_size = st.number_input("Minimum topic size", min_value=1, max_value=100, value=5, step=1)
        seed_z = st.number_input("Seed for reproducibility (optional)", min_value=0, value=None, step=1)
        use_oa_z = st.checkbox("Use GPT-4o for labels? (Zero-Shot)")
        oa_key_z = st.text_input("OpenAI API Key", type='password') if use_oa_z else None
        if st.button("Run Zero-Shot Model"):
            with st.spinner("Running zero-shot topic modeling..."):
                try:
                    emb_z = SentenceTransformer('thenlper/gte-small')
                    umap_z = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine', random_state=seed_z or random.randint(1,10000))
                    rep_z = {"Unique Keywords": KeyBERTInspired()}
                    if use_oa_z and oa_key_z:
                        try:
                            cli_z = openai.OpenAI(api_key=oa_key_z)
                            prompt_z = "Given topic keywords: [KEYWORDS]..."
                            rep_z["GPT Topic Label"] = OpenAIRep(client=cli_z, model='gpt-4o', prompt=prompt_z, chat=True, nr_docs=10, delay_in_seconds=3)
                        except Exception as e:
                            st.error(f"OpenAI init failed: {e}")
                    model_z = BERTopic(
                        representation_model=rep_z,
                        umap_model=umap_z,
                        embedding_model=emb_z,
                        min_topic_size=min_size,
                        zeroshot_topic_list=topics_list,
                        zeroshot_min_similarity=sim
                    )
                    topics_z, _ = model_z.fit_transform(text_data)
                    info_z = model_z.get_topic_info()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Identified Topics")
                        st.dataframe(info_z)
                    with col2:
                        st.subheader("Document-Topic Probabilities")
                        if set(topics_z) - {-1}:
                            docs_info = model_z.get_document_info(text_data)
                            probs_z = model_z.transform(text_data)
                            prob_df = pd.DataFrame({'Topic': probs_z[0], 'Probability': probs_z[1]})
                            docs_df = pd.concat([docs_info[['Document']], prob_df], axis=1)
                            st.dataframe(docs_df)
                        else:
                            st.warning("No valid topics found.")
                except Exception as e:
                    st.error(f"Zero-Shot error: {e}")
