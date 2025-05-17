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

# Utility functions
def create_unique_id(text): return hashlib.md5(text.encode()).hexdigest()
def extract_text_df(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()
    return df
def create_download_link(df, fname, txt):
    csv = df.to_csv(index=False)
    st.download_button(txt, csv, file_name=fname)

def display_unsupervised(BERTmodel, text_data, doc_ids, original_csv):
    # Hierarchy & map
    hier = BERTmodel.hierarchical_topics(text_data)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(BERTmodel.visualize_hierarchy(hierarchical_topics=hier), config=config)
    with c2: st.plotly_chart(BERTmodel.visualize_topics(), config=config)
    # Topic info & doc probs
    ti, dp = st.columns(2)
    with ti: st.dataframe(BERTmodel.get_topic_info().drop(columns=[c for c in ['Name','Representation'] if c in BERTmodel.get_topic_info().columns], errors='ignore'))
    with dp:
        di = BERTmodel.get_document_info(text_data)
        di['doc_id'] = doc_ids['doc_id'].tolist()
        st.dataframe(di.drop(columns=[c for c in ['Name','Top_n_words','Representative Docs','Representative_document'] if c in di.columns], errors='ignore'))
    create_download_link(original_csv, 'original_with_ids.csv', 'Download CSV with IDs')

# Shared config
config = {'toImageButtonOptions':{'format':'png','filename':'custom_image','height':1000,'width':1400,'scale':1}}

# 1. Upload
uploaded = st.file_uploader("Upload a CSV file", type=['csv'])
if uploaded:
    df = extract_text_df(uploaded)
    # 2. Select text column
    cols = df.select_dtypes(include='object').columns.tolist()
    text_col = st.selectbox("Select text column", cols)
    df = df.dropna(subset=[text_col])
    if df.empty:
        st.error("No valid text data found.")
    else:
        # Prepare data
        df['doc_id'] = df[text_col].astype(str).apply(create_unique_id)
        text_data = df[text_col].astype(str).tolist()
        doc_ids = df[['doc_id']]
        original_csv = df.copy()
        # 3. Choose method
        method = st.selectbox("Select modeling method", ['Unsupervised','Zero-Shot'], index=0)
        if method=='Unsupervised':
            # Unsupervised params
            st.subheader("Unsupervised Configuration")
            seed = st.number_input("Seed (optional)", min_value=0, value=None, step=1)
            lang = st.selectbox("Language",['English','Multilingual'])
            option = st.selectbox("Topic count handling",['Auto','Specific Number'])
            nr_topics = None if option=='Auto' else st.number_input("Number of topics",1)
            out_red = st.checkbox("Apply Outlier Reduction?", True)
            if out_red:
                thr = st.slider("c-TF-IDF threshold",0.0,1.0,0.1)
            use_oa = st.checkbox("Use GPT-4o for labels?")
            oa_key = st.text_input("OpenAI API Key", type='password') if use_oa else None
            if st.button("Run Unsupervised Model"):
                # run
                s = seed or random.randint(1,10000)
                st.write(f"Using seed: {s}")
                emb = SentenceTransformer('all-MiniLM-L6-v2')
                umap_m = UMAP(n_neighbors=10,n_components=5,min_dist=0.0,metric='cosine',random_state=s)
                vect = CountVectorizer(stop_words='english',min_df=1,max_df=0.9,ngram_range=(1,3))
                rep = {"Unique Keywords":KeyBERTInspired()}
                if use_oa and oa_key:
                    try:
                        cli = openai.OpenAI(api_key=oa_key)
                        prompt = "I have a topic described by keywords: [KEYWORDS]..."
                        rep["GPT Topic Label"] = OpenAIRep(client=cli,model='gpt-4o',prompt=prompt,chat=True,nr_docs=10,delay_in_seconds=3)
                    except Exception as e:
                        st.error(f"OpenAI init failed: {e}")
                else:
                    try:
                        gen = pipeline('text2text-generation',model='google/flan-t5-base')
                        rep["T2T Topic Label"] = TextGeneration(gen)
                    except Exception as e:
                        st.error(f"T2T init failed: {e}")
                model = BERTopic(representation_model=rep,umap_model=umap_m,embedding_model=emb,vectorizer_model=vect,top_n_words=10,nr_topics=nr_topics,language=lang.lower(),calculate_probabilities=True)
                topics,probs = model.fit_transform(text_data)
                if out_red and topics is not None:
                    new_t = model.reduce_outliers(text_data,topics,strategy='c-tf-idf',threshold=thr)
                    new_t = model.reduce_outliers(text_data,new_t,strategy='distributions')
                    model.update_topics(text_data,topics=new_t)
                display_unsupervised(model,text_data,doc_ids,original_csv)
        else:
            # Zero-Shot branch
            st.subheader("Zero-Shot Configuration")
            pre = st.text_area("Predefined topics (comma-separated)")
            topics_list = [t.strip() for t in pre.split(',') if t.strip()]
            if not topics_list:
                st.error("Enter at least one predefined topic.")
            else:
                sim = st.slider("Min similarity",0.0,1.0,0.85)
                min_size = st.number_input("Min topic size",1,100,5)
                seed_z = st.number_input("Seed (optional)",0,value=None,step=1)
                s_z = seed_z or random.randint(1,10000)
                st.write(f"Using seed: {s_z}")
                use_oa_z = st.checkbox("Use GPT-4o for labels? (Zero-Shot)")
                oa_key_z = st.text_input("OpenAI API Key", type='password') if use_oa_z else None
                if st.button("Run Zero-Shot Model"):
                    try:
                        emb_z = SentenceTransformer('thenlper/gte-small')
                        umap_z = UMAP(n_neighbors=10,n_components=5,min_dist=0.0,metric='cosine',random_state=s_z)
                        rep_z = {"Unique Keywords":KeyBERTInspired()}
                        if use_oa_z and oa_key_z:
                            try:
                                cli_z = openai.OpenAI(api_key=oa_key_z)
                                lbl = "Given the topic described by keywords: [KEYWORDS]..."
                                rep_z["GPT Topic Label"] = OpenAIRep(client=cli_z,model='gpt-4o',prompt=lbl,chat=True,nr_docs=10,delay_in_seconds=3)
                            except Exception as e:
                                st.error(f"OpenAI init failed: {e}")
                        model_z = BERTopic(representation_model=rep_z,umap_model=umap_z,embedding_model=emb_z,min_topic_size=min_size,zeroshot_topic_list=topics_list,zeroshot_min_similarity=sim)
                        topics_z,_ = model_z.fit_transform(text_data)
                        info_z = model_z.get_topic_info()
                        probs_z = model_z.transform(text_data) if set(topics_z)-{-1} else ([],[])
                        prob_df = pd.DataFrame({'Topic':probs_z[0],'Probability':probs_z[1]})
                        docs_df = model_z.get_document_info(text_data)
                        docs_df = pd.concat([docs_df[['Document']],prob_df],axis=1)
                        col1,col2 = st.columns(2)
                        with col1: st.dataframe(info_z)
                        with col2: st.dataframe(docs_df)
                    except Exception as e:
                        st.error(f"Zero-Shot error: {e}")
