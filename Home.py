import streamlit as st 

st.set_page_config(
    page_title="TextViz Studio",
    layout="wide"
)



st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.sidebar.markdown("")

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
                TextViz Studio is an all-in-one platform designed to simplify complex computational analysis for social scientists 
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

st.markdown(" #### StatsDashboard: Statistical Data Exploration & Visualization")

st.markdown("""
    StatsDashboard provides an intuitive, user-friendly interface for comprehensive statistical analysis and visualization. Designed to simplify complex data 
    analysis tasks, StatsDashboard empowers researchers and social scientists to explore, visualize, and interpret their data without needing programming skills. 
    By supporting diverse statistical tests, including T-tests, Chi-Square, ANOVA, and correlation analysis, StatsDashboard offers the tools to investigate 
    relationships within data. Its customizable visualizations, proportion tables, and weighted calculations make it ideal for in-depth examination of survey 
    responses, demographic distributions, and experimental results. StatsDashboard’s flexibility, combined with easy data uploads and export options, ensures 
    a seamless analytical experience tailored for effective, data-driven insights.
""")

expander = st.expander("**Key Features**")
expander.write('''
    - Data Subsetting & Filtering: Easily refine your dataset by selecting specific rows and columns for analysis.
    - Proportion Tables & Weighted Analysis: Generate proportion tables with optional weighted calculations for accurate survey data interpretation.
    - Comprehensive Statistical Tests: Perform T-tests, Chi-Square, ANOVA, and more to analyze relationships within your data.
    - Descriptive Statistics & Correlation Analysis: Gain detailed insight into your dataset through summary statistics and correlation matrices.
    - Versatile Visualization Tools: Create histograms, scatter plots, line plots, regression plots, and box plots with options for customization.
    - Aggregation Functions: Apply sum, mean, count, and other aggregation functions to streamline data summaries.
    - High-Quality Export Options: Download visualizations in high-resolution PNG or interactive HTML formats for reporting and sharing.
''')    

st.text("")

st.markdown("#### StatsModeling: Interactive Statistical Modeling & Estimation")

st.markdown("""
StatsModeling provides a flexible, user-friendly interface for building, estimating, and comparing regression models across a wide range of statistical types. 
Designed to simplify complex regression workflows, the app empowers users to specify multiple models side-by-side, control advanced settings, and obtain clear, 
interpretable results—without writing code. Whether working with linear, binary, count, or panel data, StatsModeling supports a full suite of estimators and 
diagnostic options, making it suitable for researchers, students, and applied analysts alike.

From uploading data to exporting polished tables, every step is built around an interactive experience. Users can define their models with full control over 
predictors, fixed effects, panel structures, weights, and interaction terms, while also customizing standard errors and output formats. With support for 
LaTeX, StatsModeling helps streamline the path from data analysis to presentation, offering a complete regression environment in the browser.
""")

expander = st.expander("**Key Features**")
expander.write('''
- Parallel Model Estimation: Configure and estimate up to four models simultaneously for easy comparison.
- Broad Estimator Support: Choose from 14 estimators, including OLS, logit/probit, Poisson, negative binomial, zero-inflated, ordered, and mixed-effects models.
- Custom Model Inputs: Define dependent variables, predictors, interaction terms, fixed effects, panel IDs, and analytic weights with ease.
- Dynamic Standard Errors: Select homoskedastic, robust (HC0/HC1), or clustered standard errors depending on your research design.
- Advanced Controls: Customize solver methods, toggle exponentiated coefficients, and configure zero-inflation link functions.
- Export-Ready Outputs: Download regression results as formatted LaTeX tables.
- Guided Usability: Includes descriptions and tooltips for model types and error structures to assist users of all experience levels.
''')    
        
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
    - Exportable Results: Download topics and summaries for reporting or further analysis.
''')

st.text("")

st.markdown(" #### Text2Sentiment: Sentiment & Emotion Analysis")

st.markdown("""
    Uncover the emotional and sentiment-driven insights hidden within your text. Text2Sentiment analyzes text data for sentiment polarity and emotional content across multiple languages. 
    Whether you're analyzing survey responses, social media posts, or research data, this tool provides a clear breakdown of both sentiment (positive, negative, neutral) and emotions (joy, anger, sadness, etc.), 
    making it ideal for understanding user opinions or emotional trends in textual data.
""")

expander = st.expander("**Key Features**")
expander.write('''
    - Multilingual Support: Analyze text in multiple languages.
    - Sentiment Analysis: Detect positive, negative, and neutral sentiments in your text data.
    - Emotion Analysis: Identify emotions such as joy, anger, fear, and more with detailed emotion breakdowns.
    - Visualization Tools: Generate visualizations like bar charts for sentiment and emotion distributions.
    - Customizable Models: Choose between different dictionary models and a multi-lingual LLM for granular insights.
    - Export Results: Download your sentiment and emotion analysis for further exploration.
''')
    
st.text("")
st.text("")

st.markdown("### **About the Developer**")

st.text("")
    
st.markdown("""
            TextViz Studio was developed by [José J. Alcocer](https://alcocer-jj.github.io/), Ph.D., M.P.P., 
            an incoming Applied Research Statistician at Harvard Law School. He specializes in legislative institutions, 
            political methodology, and computational social science. His work bridges the gap between social science and 
            data science, leveraging statistical modeling, machine learning, natural language processing, and causal inference 
            to explore complex political and social questions—particularly those related to race and ethnicity.

            With over seven years of research experience across academic, nonprofit, and government settings, 
            José brings expertise in quantitative analysis, survey methodology, experimental design, and data visualization. 
            His commitment to accessible and rigorous methods drives the development of tools that help researchers 
            analyze and interpret their data with clarity and confidence.
            """)


    
    






