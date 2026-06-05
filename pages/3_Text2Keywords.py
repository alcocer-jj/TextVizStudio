import streamlit as st
import pandas as pd
import hashlib
from PyPDF2 import PdfReader
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import nltk


st.set_page_config(
    page_title="Text2Keywords",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 20


# ---------------------------------------------------------------------------
# Cached loaders
# Run ONCE per app process / per unique input, shared across all sessions
# ---------------------------------------------------------------------------

@st.cache_resource
def get_gsheets_client():
    """Authenticate and return a gspread client. Cached for life of the app."""
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], scope
    )
    return gspread.authorize(creds)


@st.cache_resource
def get_stopwords(language: str):
    """
    Return a stopword list for the given UI language.

    Uses NLTK's curated stopword lists. NLTK ships lists for English, French,
    Spanish, Italian, Portuguese, and Arabic — these all map cleanly here.

    Chinese is NOT supported by NLTK and would require word segmentation
    (e.g., jieba) for stopword removal to be meaningful, since our current
    Chinese text cleaning produces character-level tokens. Returns None for
    Chinese, meaning no stopword removal (the same as the previous behavior
    for non-English languages — except previously it was unintentional).

    Cached per language across all users; the NLTK corpus is downloaded
    once per app process if not already present.
    """
    nltk_mapping = {
        "English":    "english",
        "French":     "french",
        "Spanish":    "spanish",
        "Italian":    "italian",
        "Portuguese": "portuguese",
        "Arabic":     "arabic",
    }
    if language not in nltk_mapping:
        return None

    try:
        from nltk.corpus import stopwords
        return stopwords.words(nltk_mapping[language])
    except LookupError:
        # Corpus not downloaded yet; do it now (one-time, ~5 MB)
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return stopwords.words(nltk_mapping[language])


@st.cache_data(show_spinner=False)
def extract_pdf_text_cached(file_bytes: bytes, file_name: str):
    """
    Extract text from a single PDF. Cached on (file_bytes, file_name) so the
    same file uploaded twice in a session only parses once.

    Uses list+join (O(n) vs O(n^2) for repeated +=) and guards against
    pages that return None (scanned/image-only pages).
    """
    reader = PdfReader(BytesIO(file_bytes))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return file_name, " ".join(pages)


@st.cache_data(show_spinner=False)
def extract_csv_text_cached(file_bytes: bytes, file_name: str):
    """Extract text from a single CSV file. Cached on (file_bytes, file_name)."""
    df = pd.read_csv(BytesIO(file_bytes))
    columns_lower = [col.lower() for col in df.columns]
    if 'text' not in columns_lower:
        return file_name, None  # signal missing column to caller
    actual_column_name = df.columns[columns_lower.index('text')]
    text = " ".join(df[actual_column_name].dropna().astype(str).tolist())
    return file_name, text


# ---------------------------------------------------------------------------
# Text cleaning (unchanged - methodology preserved exactly)
# ---------------------------------------------------------------------------

def clean_text(text, selected_language="English"):
    if selected_language == "English":
        return clean_text_english(text)
    elif selected_language == "French":
        return clean_text_french(text)
    elif selected_language == "Spanish":
        return clean_text_spanish(text)
    elif selected_language == "Italian":
        return clean_text_italian(text)
    elif selected_language == "Portuguese":
        return clean_text_portuguese(text)
    elif selected_language == "Arabic":
        return clean_text_arabic(text)
    else:
        return text


def clean_text_english(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_french(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", text).strip()
    return text


def clean_text_spanish(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text).strip()
    return text


def clean_text_italian(text):
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù\s]", " ", text).strip()
    return text


def clean_text_portuguese(text):
    text = text.lower()
    text = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", text).strip()
    return text


def clean_text_arabic(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Analysis functions (methodology unchanged)
# ---------------------------------------------------------------------------

def analyze_custom_keywords(text_data, keywords):
    keyword_freq = {"Features": keywords}
    for file_name, text in text_data:
        keyword_counts = []
        for keyword in keywords:
            matches = re.findall(keyword.lower(), text.lower())
            keyword_counts.append(len(matches))
        keyword_freq[file_name] = keyword_counts
    return pd.DataFrame(keyword_freq)


def discover_ngrams(text_data, top_n, language="English"):
    """
    Discover top-n unigrams, bigrams, and trigrams per document.

    Stopword removal now uses an NLTK list matched to `language`. Previously
    this function hardcoded stop_words='english' regardless of the user's
    language selection, which silently broke stopword removal for all
    non-English text. See get_stopwords() for language coverage.
    """
    stopword_list = get_stopwords(language)

    ngram_results = []
    ngram_ranges = [(1, 1), (2, 2), (3, 3)]
    ngram_labels = ['Unigrams', 'Bigrams', 'Trigrams']

    for file_name, text in text_data:
        for ngram_range, label in zip(ngram_ranges, ngram_labels):
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=stopword_list)
            try:
                ngram_counts = vectorizer.fit_transform([text])
            except ValueError:
                # CountVectorizer raises "empty vocabulary" when this document
                # has no terms for the n-gram range (blank text, or only
                # stopwords/punctuation). Skip it rather than crashing the run.
                continue
            ngram_sum = ngram_counts.sum(axis=0).A1
            ngram_names = vectorizer.get_feature_names_out()
            ngram_freq = pd.DataFrame({'N-gram': ngram_names, 'Frequency': ngram_sum})
            ngram_freq = ngram_freq.sort_values(by='Frequency', ascending=False).head(top_n)
            ngram_freq['Document'] = file_name
            ngram_freq['Ngram_Type'] = label
            ngram_results.append(ngram_freq)

    if not ngram_results:
        # No usable n-grams in any document. Return an empty, correctly-typed
        # frame so callers can show a message instead of pd.concat raising
        # "No objects to concatenate".
        return pd.DataFrame(columns=['N-gram', 'Frequency', 'Document', 'Ngram_Type'])

    combined_df = pd.concat(ngram_results, ignore_index=True)
    return combined_df


def generate_wordcloud(df, colormap, term_column='N-gram', frequency_column='Frequency'):
    if df.empty:
        return None
    freq_dict = dict(zip(df[term_column], df[frequency_column]))
    # WordCloud normalizes by the maximum frequency, so a dict whose values are
    # all zero (e.g. custom keywords that never appear in the text) triggers a
    # ZeroDivisionError. Keep only finite, strictly-positive weights and bail
    # out cleanly if nothing remains, letting the caller show a message.
    freq_dict = {
        term: float(freq)
        for term, freq in freq_dict.items()
        if pd.notna(freq) and float(freq) > 0
    }
    if not freq_dict:
        return None
    kwargs = dict(width=3840, height=2160, background_color="white")
    if colormap:
        kwargs["colormap"] = colormap
    return WordCloud(**kwargs).generate_from_frequencies(freq_dict)


# ---------------------------------------------------------------------------
# Display + ZIP helpers
# ---------------------------------------------------------------------------

def _render_wordcloud_to_buffer(wordcloud_obj):
    """
    Render a WordCloud to a Streamlit display AND a BytesIO PNG buffer.
    Uses explicit fig/ax pattern + plt.close(fig) for safe cleanup.
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.imshow(wordcloud_obj, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf


def display_custom_keyword_results(keyword_df, color_scheme, colormap_options):
    tab1, tab2 = st.tabs(["DataFrame", "Custom Keyword Word Cloud"])

    with tab1:
        st.subheader("Custom Keywords Analysis Results")
        st.dataframe(keyword_df)

    image_buffer = None
    with tab2:
        st.subheader("Custom Keyword Word Cloud")
        # Sum each keyword's counts across every document column (all columns
        # except "Features") so the cloud reflects the whole corpus rather than
        # only the first uploaded file.
        doc_cols = [c for c in keyword_df.columns if c != 'Features']
        wc_df = keyword_df[['Features']].copy()
        wc_df['Total'] = keyword_df[doc_cols].sum(axis=1)
        wordcloud_image = generate_wordcloud(
            wc_df,
            colormap_options[color_scheme],
            term_column='Features',
            frequency_column='Total',
        )
        if wordcloud_image:
            image_buffer = _render_wordcloud_to_buffer(wordcloud_image)
        else:
            st.info(
                "No word cloud to display: none of your custom keywords were "
                "found in any of the documents, so every frequency is zero."
            )

    # Bundle results into a ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(
            'keyword_analysis_results.csv',
            keyword_df.to_csv(index=False).encode('utf-8'),
        )
        if image_buffer is not None:
            zip_file.writestr('custom_keyword_wordcloud.png', image_buffer.read())
    zip_buffer.seek(0)

    st.download_button(
        label="Download All Outputs (ZIP)",
        data=zip_buffer,
        file_name="custom_keyword_analysis.zip",
        mime="application/zip",
    )


def display_combined_ngram_results(combined_df, color_scheme, colormap_options):
    wordcloud_images = {}

    if combined_df.empty:
        st.info(
            "No n-grams could be extracted from the uploaded text. This usually "
            "means the documents were empty or contained only stopwords. Try a "
            "different file, a lower frequency threshold, or another language setting."
        )
        return

    tab_labels = ["DataFrame", "Unigram Word Cloud", "Bigram Word Cloud", "Trigram Word Cloud"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.subheader("Combined N-gram Analysis Results")
        st.dataframe(combined_df)
        st.download_button(
            label="Download Combined N-gram Data (CSV)",
            data=combined_df.to_csv(index=False).encode('utf-8'),
            file_name="combined_ngram_analysis.csv",
            mime="text/csv",
        )

    ngram_types = combined_df['Ngram_Type'].unique()
    for ngram_type in ngram_types:
        with tabs[list(ngram_types).index(ngram_type) + 1]:
            df = combined_df[combined_df['Ngram_Type'] == ngram_type]
            st.subheader(f"{ngram_type} Word Cloud")
            wordcloud = generate_wordcloud(
                df,
                colormap_options[color_scheme],
                term_column='N-gram',
                frequency_column='Frequency',
            )
            if wordcloud:
                wordcloud_images[ngram_type] = _render_wordcloud_to_buffer(wordcloud)
            else:
                st.info(f"No {ngram_type.lower()} word cloud to display for this selection.")

    if wordcloud_images:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(
                'combined_ngram_analysis.csv',
                combined_df.to_csv(index=False).encode('utf-8'),
            )
            for ngram_type, image_data in wordcloud_images.items():
                zip_file.writestr(f'{ngram_type.lower()}_wordcloud.png', image_data.read())
        zip_buffer.seek(0)

        st.download_button(
            label="Download All Outputs (ZIP)",
            data=zip_buffer,
            file_name="ngram_analysis_outputs.zip",
            mime="application/zip",
        )


# ---------------------------------------------------------------------------
# Sidebar feedback form (uses cached gsheets client)
# ---------------------------------------------------------------------------
try:
    client = get_gsheets_client()
    sheet = client.open("TextViz Studio Feedback").sheet1

    st.sidebar.markdown("### **Feedback**")
    feedback = st.sidebar.text_area(
        "Experiencing bugs/issues? Have ideas to better the application tool?",
        placeholder="Leave feedback or error code here"
    )

    if st.sidebar.button("Submit"):
        if feedback:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sheet.append_row(["Text2Keywords:", feedback, timestamp])
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
st.sidebar.markdown(
    "For full documentation and future updates to the appliction, "
    "check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)"
)
st.sidebar.markdown("")


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align: center'>Text2Keywords: Keyword & Phrase Visualization</h1>",
    unsafe_allow_html=True
)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
**Text2Keywords** is an interactive tool designed for keyword frequency and n-gram discovery from text data. 
Upload your CSV or PDF files for analysis and choose between custom keyword input or automatic discovery of the 
most frequent unigrams, bigrams, and trigrams. The tool supports text cleaning in multiple languages, including 
English, French, Spanish, Italian, Portuguese, and Arabic. Visualize your results with keyword frequency 
data, generate word clouds, and download all outputs, including the analysis data and visualizations, in a convenient 
ZIP format. Tailor your analysis to suit your needs, whether for custom keyword extraction or automatic n-gram 
analysis.
""")

st.markdown("")
st.markdown("")


# ---------------------------------------------------------------------------
# Wizard state initialization
# ---------------------------------------------------------------------------
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None


def advance_to(step: int):
    """Advance the wizard. Idempotent — never goes backward on its own."""
    st.session_state.wizard_step = max(st.session_state.wizard_step, step)


# ---------------------------------------------------------------------------
# Progress indicator
# ---------------------------------------------------------------------------
step = st.session_state.wizard_step
checkmark = lambda n: "✅" if step > n else ("🔵" if step == n else "⬜")
st.markdown(
    f"**Progress:** &nbsp; {checkmark(1)} Upload &nbsp;·&nbsp; "
    f"{checkmark(2)} Preprocessing &nbsp;·&nbsp; "
    f"{checkmark(3)} Method &nbsp;·&nbsp; "
    f"{checkmark(4)} Visualization & Run"
)
st.markdown("")


# ---------------------------------------------------------------------------
# STEP 1 · Import Data
# ---------------------------------------------------------------------------
st.subheader("Step 1 · Import Data", divider=True)

uploaded_files = st.file_uploader(
    "Upload CSV or PDF files",
    type=["csv", "pdf"],
    accept_multiple_files=True,
    key="file_uploader",
)
st.warning("For CSV files, ensure that the text data is in a column named 'text'.")

if uploaded_files:
    # Size cap check
    oversized = [f for f in uploaded_files if f.size > MAX_FILE_SIZE_MB * 1024 * 1024]
    if oversized:
        names = ", ".join(f"{f.name} ({f.size / 1024 / 1024:.1f} MB)" for f in oversized)
        st.error(
            f"⚠️ The following files exceed the {MAX_FILE_SIZE_MB} MB per-file limit "
            f"and must be reduced before analysis: {names}"
        )
    else:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} file(s) ready.")
        if st.button("Continue → Preprocessing", key="continue_1"):
            advance_to(2)


# ---------------------------------------------------------------------------
# STEP 2 · Preprocessing
# ---------------------------------------------------------------------------
if st.session_state.wizard_step >= 2:
    st.subheader("Step 2 · Preprocessing", divider=True)
    language_option = st.selectbox(
        "Select the language of your text:",
        ("English", "French", "Spanish", "Italian", "Portuguese", "Arabic"),
        key="language_option",
    )
    if st.button("Continue → Analysis Method", key="continue_2"):
        advance_to(3)


# ---------------------------------------------------------------------------
# STEP 3 · Analysis Method
# ---------------------------------------------------------------------------
if st.session_state.wizard_step >= 3:
    st.subheader("Step 3 · Analysis Method", divider=True)

    analysis_option = st.radio(
        "How would you like to perform the analysis?",
        ("Input Custom Keywords", "Discover Automatically"),
        key="analysis_option",
    )

    if analysis_option == "Input Custom Keywords":
        st.warning(
            "**Instructions:** Enter one keyword or regular expression per line. For example: \n"
            "- **word** finds exact matches of the word 'word'. \n"
            "- **word(s|ing|ed)**: Finds 'word', 'words', 'wording', and 'worded'. \n"
            "- **\\d+** finds any sequence of digits (e.g., 123)."
        )
        custom_keywords = st.text_area(
            "Enter your custom keywords (one per line)",
            height=150,
            key="custom_keywords",
        )
    else:
        top_n = st.number_input(
            "Select how many top terms to discover for each n-gram type",
            min_value=1, max_value=100, value=10,
            key="top_n",
        )

    if st.button("Continue → Visualization", key="continue_3"):
        advance_to(4)


# ---------------------------------------------------------------------------
# STEP 4 · Visualization & Run
# ---------------------------------------------------------------------------
if st.session_state.wizard_step >= 4:
    st.subheader("Step 4 · Visualization & Run", divider=True)

    colormap_options = {
        "Default": None,
        "Monochromatic Blue": "Blues",
        "Monochromatic Green": "Greens",
        "Warm Tones": "autumn",
        "Cool Tones": "cool",
        "Pastel Colors": "Pastel1",
        "Grayscale": "gray",
        "Earth Tones": "terrain",
        "High Contrast": "Set1",
        "Colorblind-Friendly": "tab10",
    }
    color_scheme = st.selectbox(
        "Select WordCloud Color Scheme",
        options=list(colormap_options.keys()),
        key="color_scheme",
    )

    analyze_button = st.button("Run Analysis", type="primary", key="run_analysis")

    # ------------------------------------------------------------------
    # Analysis execution
    # ------------------------------------------------------------------
    if analyze_button:
        if not st.session_state.uploaded_files:
            st.error("Please upload at least one CSV or PDF file before running the analysis.")
        else:
            with st.spinner("Analyzing data..."):
                text_data = []

                pdf_files = [f for f in st.session_state.uploaded_files
                             if f.type == "application/pdf"]
                csv_files = [f for f in st.session_state.uploaded_files
                             if f.type == "text/csv"]

                # Extract PDFs (cached per file)
                for f in pdf_files:
                    try:
                        name, text = extract_pdf_text_cached(f.getvalue(), f.name)
                        text_data.append((name, text))
                    except Exception as e:
                        st.error(f"Error processing PDF {f.name}: {e}")

                # Extract CSVs (cached per file)
                for f in csv_files:
                    try:
                        name, text = extract_csv_text_cached(f.getvalue(), f.name)
                        if text is None:
                            st.error(f"CSV file {name} must contain a 'text' column.")
                        else:
                            text_data.append((name, text))
                    except Exception as e:
                        st.error(f"Error processing CSV {f.name}: {e}")

                # Clean text per selected language
                text_data = [
                    (name, clean_text(text, selected_language=language_option))
                    for name, text in text_data
                ]

                # Branch on analysis method
                if analysis_option == "Input Custom Keywords":
                    if not custom_keywords:
                        st.error("Please input custom keywords to perform the analysis.")
                    else:
                        keywords = custom_keywords.splitlines()
                        keyword_df = analyze_custom_keywords(text_data, keywords)
                        display_custom_keyword_results(keyword_df, color_scheme, colormap_options)
                else:
                    ngram_results = discover_ngrams(text_data, top_n, language=language_option)
                    display_combined_ngram_results(ngram_results, color_scheme, colormap_options)
