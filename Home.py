import streamlit as st 

st.set_page_config(
    page_title="TextViz Studio",
    layout="wide"
)


st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

#st.title("Welcome to TextViz Studio")
st.markdown("<h1 style='text-align: center'>Welcome to TextViz Studio</h1>", unsafe_allow_html=True)


st.text("")    
st.text("")
st.text("")
st.text("")
st.text("")
    
st.subheader("Bridging the Gap Between Data Science and Social Science")
                
st.text("")
st.text("")

    
st.markdown("""
                TextViz Studio is an all-in-one platform designed to simplify complex text analysis for social scientists 
                and researchers. Delving into programming and data science can be daunting, especially when our primary focus as 
                social scientists is on the substantive aspects of our work. That's why I've set to streamline a suite of existing tools 
                in a more intuitive way that eliminates the need for coding expertise, allowing users to uncover deep insights from 
                textual data effortlessly. Ultimately, the vision of this project is to democratize data science tools for the social
                science community, making them more accessible and user-friendly. I aim to remove technical barriers by providing
                user-friendly interfaces that require no coding knowledge, empowering users to focus on what truly matters—analyzing and
                interpreting the substantive content of their research.
                """)

st.text("")
st.text("")


st.markdown("### **Application Tools**")
    
st.text("")
    
st.markdown(" #### Text2Keywords: Keyword & Phrase Visualization")
        
st.markdown("""    
    Unlock the core themes of your documents with ease. Text2Keywords extracts meaningful keywords and N-grams from text files, including PDFs and CSVs. 
    By analyzing word frequencies and patterns, it highlights the most significant terms and phrases, helping you summarize and comprehend large volumes 
    of text—ideal for research, content analysis, and document reviews.
    """)    

expander = st.expander("**Key Features**")
expander.write('''
    - PDF Text Extraction: Seamlessly extract text from PDF documents for analysis.
    - Keyword Extraction: Identify the most frequent words or keywords in your text.
    - N-gram Analysis: Discover common phrases through N-gram (word combinations) analysis.
    - Word Cloud Visualization: Generate customizable word clouds to visualize word frequencies.
    - Customizable Parameters: Adjust N-gram ranges and frequency thresholds to suit your needs.
    - Batch Processing: Upload and analyze multiple PDFs or CSV files simultaneously.
    - Export Results: Download your analysis, including keyword counts and word clouds, for further use.
''')

    
st.text("")

st.markdown(" #### Text2Topics: Large Language Topic Modeling")
    
st.markdown("""
    Dive deeper into your textual data with advanced topic modeling. Text2Topics utilizes cutting-edge natural language processing (NLP) techniques like 
    [BERTopic](https://maartengr.github.io/BERTopic/index.html) to identify and group similar themes within large text corpora. Leveraging transformer-based models, it embeds text into vector space 
    for accurate and nuanced topic discovery—perfect for unearthing latent themes in research articles, survey responses, or social media posts.
    """)

expander = st.expander("**Key Features**")
expander.write('''
    - Advanced Topic Modeling: Extract topics using BERTopic, a topic modeling technique based on transformer models capable of capturing context-aware relationships between words and sentences.
    - Interactive Visualization: Explore discovered topics and their relationships visually.
    - OpenAI Integration: Leverage OpenAI's GPT-4o model for enhanced text representation and generation.
    - Customizable Parameters: Tailor model settings such as the number of topics to generate and modify.
    - Multiple Representations: Experiment with different topic representations for clarity.
    - Text Summarization: Generate concise summaries of topics and key insights.
    - Exportable Results: Download topics and summaries for reporting or further analysis.
''')



    
st.text("")
st.text("")


st.markdown("### **Future Plans**")
        
expander = st.expander("**Application Tools**")
expander.write('''
    - **Text2Sentiment:** NLP tool to conduct both sentiment and emotion analyses on text from PDFs and CSVs, and display them in tables and visuals breaking them down.
    - **Data2Viz:** Exploratory data tool that will allow users to display relationships with customizable plots (e.g., line, bar, kernal, box, etc.).
    - **Stats2Viz:** Statistical data tool that will allow users to conduct regressions, inference, clustering/classification, time-series, and causal inference while producing output tables and plots.
''')    

expander = st.expander("**Current Applications**")
expander.write('''
    - Text2Topics: 
        - Allow users to select their embedding model of choice or stick to the default one.
        - Integrate LLAMA 3.2 as baseline LLM representation model for topic labeling instead of Transformer models.
        - Integrate option for users to conduct dynamic topic modeling (across time) and zero-shot topic modeling (when you have existing topics in mind).
''')    

st.text("")
st.text("")

    
st.markdown("### **About the Developer**")

st.text("")
    
st.markdown("""
    
    TextViz Studio was developed by [José J. Alcocer](https://alcocer-jj.github.io/), a fifth-year Ph.D. candidate in Political Science and International Relations at the 
    University of Southern California. Specializing in computational social science and political methodology, they focus on leveraging 
    machine learning and natural language processing to address complex questions in social science—particularly those related to 
    race and ethnicity. As a graduate research associate at the Security and Political Economy Lab (SPEC), they contribute to projects 
    that utilize advanced NLP models to analyze communication patterns and their impacts on communities. With experience working as 
    a survey statistician at the U.S. Census Bureau and as a computational social scientist at The Center for Election Science, they bring 
    a wealth of expertise in data analysis and research methods. Their passion for bridging the gap between data science and social science
    drives their commitment to facilitating tools that empower researchers to explore and understand their data more effectively.
    """)


    
    






