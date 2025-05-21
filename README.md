# TextViz Studio

## Overview

TextViz Studio is an intuitive web-based platform designed to simplify text analysis for social scientists and researchers. This Python-based application performs all analyses 'under-the-hood' using existing libraries so that users experience a coding-free environment to explore text data and extract meaningful insights with ease. By bridging the gap between data science and social science, the goal of TextViz Studio is to democratize access to advanced text analysis tools, empowering users to focus on content rather than code.


## Application Tools

### StatsDashboard: Statistical Data Exploration & Visualization

StatsDashboard offers an accessible, all-in-one platform for statistical data analysis and visualization, tailored to meet the needs of researchers and social scientists. This module simplifies data subsetting, descriptive statistics, statistical testing, and visualization, making complex analysis more approachable for users without coding expertise.

- Data Subsetting & Filtering: Refine datasets by selecting specific rows and columns for focused analysis.
- Proportion Tables & Weighted Analysis: Generate proportion tables with optional weighted calculations, ideal for survey and demographic data.
- Comprehensive Statistical Tests: Perform T-tests, Chi-Square, ANOVA, and more to uncover relationships within your data.
- Descriptive Statistics & Correlation Analysis: Summarize data distributions and calculate correlations to understand variable interactions.
- Flexible Visualization Tools: Create histograms, scatter plots, line plots, regression plots, bar charts, and box plots, each with customizable themes, grouping, and aggregation functions.
- Export High-Quality Visualizations: Download plots in high-resolution PNG or interactive HTML formats for sharing and reporting.

### StatsModeling: Interactive Statistical Modeling & Estimation

StatsModeling is a flexible, browser-based application for building, estimating, and comparing a wide range of regression models. Designed for applied researchers, students, and analysts, this module streamlines model specification and diagnostics while offering robust customization options—all without the need for coding.

- Multi-Model Configuration: Specify and compare up to four regression models in parallel using an intuitive, expandable interface.
- Broad Estimator Support: Choose from 14 model types including OLS, logit/probit, Poisson, negative binomial (incl. zero-inflated), ordered logit/probit, and panel or mixed-effects models.
- Custom Model Inputs: Define dependent variables, predictors, interaction terms, fixed effects, panel identifiers, and analytic weights.
- Standard Error Controls: Select from homoskedastic, robust (HC0/HC1), or clustered standard errors to fit your research design.
- Exportable Results: Generate clean LaTeX tables for use in reports or publications.
- Built-In Guidance: Each model type and estimation option includes plain-language descriptions for accessible, informed use.

### Text2Keywords: Keyword & Phrase Visualization

This tool helps uncover key themes and patterns in text files, such as PDFs or CSVs. Users can analyze word frequencies and visualize their results through word clouds and keyword summaries.

- PDF Text Extraction: Extract text from PDF documents seamlessly.
- Keyword Extraction: Identify high-frequency keywords or phrases.
- N-gram Analysis: Discover common word combinations (N-grams) in your data.
- Word Cloud Visualization: Generate visual representations of word frequencies.
- Customizable Parameters: Adjust frequency thresholds and N-gram ranges.
- Batch Processing: Analyze multiple PDFs or CSV files at once.
- Export Results: Download keyword counts, word clouds, and summaries for further use.

### Text2Sentiment: Sentiment Discovery

Key Features

- Multilingual Support: Analyze text in over 50 languages
- Sentiment Analysis: Detect positive, negative, and neutral sentiments in your text data.
- Emotion Analysis: Identify emotions such as joy, anger, fear, and more with detailed emotion breakdowns.
- Visualization Tools: Generate visualizations like bar charts for sentiment and emotion distributions.
- Customizable Models: Choose between dictionary methods or a large language model trained for sentiment.
- Export Results: Download your analysis results for further exploration.

### Text2Topics: Large Language Topic Modeling

This advanced tool uses transformer-based models to perform topic modeling, revealing latent themes in text. It’s ideal for analyzing large corpora such as research papers, survey responses, or online discussions.

- Advanced Topic Modeling: Leverages BERTopic for unsupervised topic discoviery or zero-shot classification.
- Interactive Visualization: Visualize topics and their relationships interactively.
- OpenAI Integration: Employs OpenAI GPT-4o for enhanced topic representation as an option.
- Customizable Parameters: Adjust the number of topics.
- Multiple Representations: Compare different topic representations for clarity.
- Text Summarization: Summarize topics and generate concise reports.
- Exportable Results: Download topics, summaries, and visualizations for reporting.

## Updates and References

#### Update 1.2 (11/13/2024)

- **StatsDashboard**:
  - Added new **StatsDashboard** module for statistical data analysis and visualization.
  - Features include data subsetting, customizable proportion tables with weighted options, statistical tests (T-tests, Chi-Square, ANOVA), and a variety of visualization tools (histograms, scatter plots, line plots, etc.).
  - Includes high-quality export options for PNG and HTML visualizations.

#### Update 1.1 (11/05/2024)

- **Text2Sentiment:**
  - Added Korean and Turkish to [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) as additional languages for sentiment and emotion.
  - Replaced [nli-distilroberta-base](https://huggingface.co/cross-encoder/nli-distilroberta-base) with [XLM-RoBERTa-Twitter-Sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment) for multi-language support.

- **Text2Topics:**
  - Added Topic hierarchical plot capabilities for further exploration.

#### 1.0 (10/16/2024)

This section will be continuously changing as newer and more efficient resources continue to be released. As of October 16,2024, the following resources are used to perform the variety of analyses wrapped in the application overall:
- **Text2Topics:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for the sentence-embedding model in BERTopic, and [FLAN-T5 base](https://huggingface.co/google/flan-t5-base) for one of the representation models to produce topic labels.
- **Text2Sentiment:** The [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) as the default dictionary for sentiment and emotion classification of text ([Mohammad 2021](https://arxiv.org/abs/2005.11882)), the [VADER](https://github.com/cjhutto/vaderSentiment) lexicon ([Hutto & Gilbert 2014](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)) and the [nli-distilroberta-base](https://huggingface.co/cross-encoder/nli-distilroberta-base) natural language inference (NLI) model as additional options to classify sentiment from text.
