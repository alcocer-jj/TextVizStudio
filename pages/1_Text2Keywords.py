import streamlit as st
import pandas as pd
import hashlib  # To create unique identifiers
from PyPDF2 import PdfReader
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile


st.set_page_config(
    page_title="Text2Keywords",
    layout="wide"
)

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://alcocer-jj.github.io)")


# Sidebar: Title and description
#st.title("Text2Keywords: Keyword & Phrase Visualization")
st.markdown("<h1 style='text-align: center'>Text2Keywords: Keyword & Phrase Visualization</h1>", unsafe_allow_html=True)


st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
Analyze keyword frequency and discover n-grams from your text data. You can upload multiple CSV or PDF files
for analysis. Input custom keywords, or let the app automatically discover the most frequent unigrams, bigrams, 
and trigrams. Download your analysis results in a convenient ZIP format with all outputs.
""")

st.markdown("")
st.markdown("")

# Initialize session state to keep track of uploaded data
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
    st.session_state.analysis_data = None

# Function to create unique identifiers for each document
def create_unique_id(text):
    return hashlib.md5(text.encode()).hexdigest()

# Extract text from PDF files
def extract_text_from_pdfs(files):
    all_texts = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        all_texts.append((file.name, text))  # Store file name and text as a tuple
    return all_texts

# Extract text from CSV files
def extract_text_from_csvs(files):
    all_texts = []
    for file in files:
        df = pd.read_csv(file)
        if 'text' in df.columns:
            text = " ".join(df['text'].dropna().tolist())
            all_texts.append((file.name, text))  # Store file name and text as a tuple
        else:
            st.error(f"CSV file {file.name} must contain a 'text' column.")
    return all_texts

# Preprocessing functions for different languages
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
    elif selected_language == "Chinese":
        return clean_text_chinese(text)
    elif selected_language == "Arabic":
        return clean_text_arabic(text)
    else:
        return text

# Function to clean English text
def clean_text_english(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to clean French text
def clean_text_french(text):
    text = text.lower()
    text = re.sub(r"[^a-zàâäéèêëîïôùûüç\s]", " ", text).strip()
    return text

# Function to clean Spanish text
def clean_text_spanish(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text).strip()
    return text

# Function to clean Italian text
def clean_text_italian(text):
    text = text.lower()
    text = re.sub(r"[^a-zàèéìòù\s]", " ", text).strip()
    return text

# Function to clean Portuguese text
def clean_text_portuguese(text):
    text = text.lower()
    text = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", text).strip()
    return text

# Function to clean Chinese text
def clean_text_chinese(text):
    text = re.sub(r"[^\u4e00-\u9fff\s]", " ", text)  # Remove non-Chinese characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to clean Arabic text
def clean_text_arabic(text):
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)  # Remove non-Arabic characters
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.subheader("Import Data", divider=True)

# File uploader to handle CSV or PDF files
uploaded_files = st.file_uploader("Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True)
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

st.warning("For CSV files, ensure that the text data is in a column named 'text'.")

# Move language selection to the main page
language_option = st.selectbox(
    "Select the language of your text:",
    ("English", "French", "Spanish", "Italian", "Portuguese", "Chinese", "Arabic")
)

# Instructions for custom keyword input, with examples
st.subheader("Setting Parameters", divider=True)

# Give users a choice between inputting custom keywords or automatic discovery
analysis_option = st.radio("How would you like to perform the analysis?", ("Input Custom Keywords", "Discover Automatically"))

# If user selects custom keywords, show the input box for entering keywords
custom_keywords = None
if analysis_option == "Input Custom Keywords":
    st.warning("**Instructions:** Enter one keyword or regular expression per line. For example: \n - **word** finds exact matches of the word 'word'. \n - **word(s|ing|ed)**: Finds 'word', 'words', 'wording', and 'worded'. \n - **\\d+** finds any sequence of digits (e.g., 123).")   
    custom_keywords = st.text_area("Enter your custom keywords (one per line)", height=150)
else:
    top_n = st.number_input("Select how many top terms to discover for each n-gram type", min_value=1, max_value=100, value=10)

# Restrict colormap options for the word cloud
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
    "Colorblind-Friendly": "tab10"
}

# WordCloud color scheme selection
color_scheme = st.selectbox("Select WordCloud Color Scheme", options=list(colormap_options.keys()))

st.header("Analysis", divider=True)

# Analysis button
analyze_button = st.button("Run Analysis")

# Function to generate word clouds and return as a WordCloud object (no BytesIO handling in this part)
def generate_wordcloud(df, colormap, term_column='N-gram', frequency_column='Frequency'):
    # Create the word cloud from the N-gram and Frequency columns
    if colormap:
        wordcloud = WordCloud(width=3840, height=2160, background_color="white", colormap=colormap).generate_from_frequencies(dict(zip(df[term_column], df[frequency_column])))
    else:
        wordcloud = WordCloud(width=3840, height=2160, background_color="white").generate_from_frequencies(dict(zip(df[term_column], df[frequency_column])))
    
    return wordcloud

# Function to display and download the results with custom keywords and word clouds
def display_custom_keyword_results(keyword_df):
    # Create tabs for DataFrame and WordClouds
    tab1, tab2 = st.tabs(["DataFrame", "Custom Keyword Word Cloud"])
    
    # Display DataFrame in the first tab
    with tab1:
        st.subheader("Custom Keywords Analysis Results")
        st.dataframe(keyword_df)

    # Display Custom Keyword Word Cloud in the second tab
    with tab2:
        st.subheader("Custom Keyword Word Cloud")
        wordcloud_image = generate_wordcloud(keyword_df, colormap_options[color_scheme], term_column='Features', frequency_column=keyword_df.columns[1])
        
        # Display the word cloud using matplotlib
        plt.figure(figsize=(16, 9), dpi=300)
        plt.imshow(wordcloud_image, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)  # Display in Streamlit

        # Save the word cloud image to a BytesIO buffer for ZIP download
        image_buffer = BytesIO()
        plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
        image_buffer.seek(0)
        plt.close()

    # Create ZIP file with DataFrame and word cloud image
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add DataFrame CSV to the ZIP
        csv_data = keyword_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr('keyword_analysis_results.csv', csv_data)
        
        # Add WordCloud PNG to the ZIP
        zip_file.writestr('custom_keyword_wordcloud.png', image_buffer.read())

    zip_buffer.seek(0)
    
    # Provide download button for the ZIP file
    st.download_button(
        label="Download All Outputs (ZIP)",
        data=zip_buffer,
        file_name="custom_keyword_analysis.zip",
        mime="application/zip"
    )

# Function to create a ZIP file containing the combined DataFrame and word cloud images
def create_zip_with_outputs(combined_df, wordcloud_images):
    zip_buffer = BytesIO()

    # Create the ZIP file
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add the combined DataFrame as a CSV file
        csv_data = combined_df.to_csv(index=False).encode('utf-8')
        zip_file.writestr('combined_ngram_analysis.csv', csv_data)

        # Add each word cloud image to the ZIP
        for ngram_type, image_data in wordcloud_images.items():
            zip_file.writestr(f'{ngram_type.lower()}_wordcloud.png', image_data.read())

    # Ensure the buffer is ready for download
    zip_buffer.seek(0)
    return zip_buffer

# Function to analyze custom keywords and generate the required DataFrame
def analyze_custom_keywords(text_data, keywords):
    keyword_freq = { "Features": keywords }
    # Collect the frequency of each keyword in each document
    for file_name, text in text_data:
        keyword_counts = []
        for keyword in keywords:
            matches = re.findall(keyword.lower(), text.lower())  # Case-insensitive matching
            count = len(matches)
            keyword_counts.append(count)
        keyword_freq[file_name] = keyword_counts
    keyword_df = pd.DataFrame(keyword_freq)
    return keyword_df

# Modified display_combined_ngram_results function to handle ZIP creation and download
def display_combined_ngram_results(combined_df):
    # Create a dictionary to store the word cloud images for ZIP download
    wordcloud_images = {}

    # Create tabs for displaying the combined DataFrame and individual word clouds
    tab_labels = ["DataFrame", "Unigram Word Cloud", "Bigram Word Cloud", "Trigram Word Cloud"]
    tabs = st.tabs(tab_labels)

    # Display the combined DataFrame in the first tab
    with tabs[0]:
        st.subheader("Combined N-gram Analysis Results")
        st.dataframe(combined_df)

        # Optionally, provide a download button for the combined DataFrame
        csv_data = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Combined N-gram Data (CSV)",
            data=csv_data,
            file_name="combined_ngram_analysis.csv",
            mime="text/csv"
        )

    # Display word clouds for unigrams, bigrams, and trigrams in separate tabs and save the images
    ngram_types = combined_df['Ngram_Type'].unique()
    for ngram_type in ngram_types:
        with tabs[list(ngram_types).index(ngram_type) + 1]:
            df = combined_df[combined_df['Ngram_Type'] == ngram_type]
            st.subheader(f"{ngram_type} Word Cloud")

            # Generate the word cloud object
            wordcloud = generate_wordcloud(df, colormap_options[color_scheme], term_column='N-gram', frequency_column='Frequency')
            
            if wordcloud:
                # Display the word cloud using Matplotlib
                plt.figure(figsize=(16, 9), dpi=300)
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)  # Display the plot in Streamlit

                # Save the word cloud image to a BytesIO buffer for ZIP download
                image_buffer = BytesIO()
                plt.savefig(image_buffer, format='png', bbox_inches='tight', pad_inches=0)
                image_buffer.seek(0)
                plt.close()

                # Store the image in the dictionary for ZIP download
                wordcloud_images[ngram_type] = image_buffer

    # Create a ZIP file containing the combined DataFrame and the word cloud images
    if wordcloud_images:
        zip_file = create_zip_with_outputs(combined_df, wordcloud_images)

        # Provide a download button for the ZIP file
        st.download_button(
            label="Download All Outputs (ZIP)",
            data=zip_file,
            file_name="ngram_analysis_outputs.zip",
            mime="application/zip"
        )
        
# Function to automatically discover top n terms for unigrams, bigrams, and trigrams
def discover_ngrams(text_data, top_n):
    # Dictionary to store the top n results for unigrams, bigrams, and trigrams per document
    ngram_results = []

    # Define n-gram ranges (1 for unigrams, 2 for bigrams, 3 for trigrams)
    ngram_ranges = [(1, 1), (2, 2), (3, 3)]
    ngram_labels = ['Unigrams', 'Bigrams', 'Trigrams']

    for file_name, text in text_data:
        for ngram_range, label in zip(ngram_ranges, ngram_labels):
            vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
            ngram_counts = vectorizer.fit_transform([text])
            
            # Sum the counts of each n-gram
            ngram_sum = ngram_counts.sum(axis=0).A1
            ngram_names = vectorizer.get_feature_names_out()

            # Create a DataFrame with n-gram names and their respective counts, sorted by frequency
            ngram_freq = pd.DataFrame({'N-gram': ngram_names, 'Frequency': ngram_sum})
            ngram_freq = ngram_freq.sort_values(by='Frequency', ascending=False).head(top_n)

            # Add document name and n-gram type to the DataFrame
            ngram_freq['Document'] = file_name
            ngram_freq['Ngram_Type'] = label

            # Append the result to the list
            ngram_results.append(ngram_freq)

    # Combine all results into a single DataFrame
    combined_df = pd.concat(ngram_results, ignore_index=True)
    return combined_df


# Run the analysis when the user clicks the button
if analyze_button:
    if not st.session_state.uploaded_files:
        st.error("Please upload at least one CSV or PDF file before running the analysis.")
    else:
        with st.spinner("Analyzing data..."):
            text_data = []

            # Separate PDFs and CSVs
            pdf_files = [file for file in st.session_state.uploaded_files if file.type == "application/pdf"]
            csv_files = [file for file in st.session_state.uploaded_files if file.type == "text/csv"]

            # Extract text from files
            if pdf_files:
                try:
                    pdf_texts = extract_text_from_pdfs(pdf_files)
                    text_data.extend(pdf_texts)
                except Exception as e:
                    st.error(f"Error processing PDF files: {str(e)}")

            if csv_files:
                try:
                    csv_texts = extract_text_from_csvs(csv_files)
                    text_data.extend(csv_texts)
                except Exception as e:
                    st.error(f"Error processing CSV files: {str(e)}")

            # Clean text based on language option
            text_data = [(file_name, clean_text(text, selected_language=language_option)) for file_name, text in text_data]

            if analysis_option == "Input Custom Keywords":
                if not custom_keywords:
                    st.error("Please input custom keywords to perform the analysis.")
                else:
                    keywords = custom_keywords.splitlines()
                    keyword_df = analyze_custom_keywords(text_data, keywords)
                    display_custom_keyword_results(keyword_df)
            else:
                # Automatic discovery of unigrams, bigrams, and trigrams
                ngram_results = discover_ngrams(text_data, top_n)
                display_combined_ngram_results(ngram_results)




