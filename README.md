# TextViz Studio

## Overview

TextViz Studio is an intuitive web-based platform designed to simplify text analysis for social scientists and researchers. This Python-based application performs all analyses 'under-the-hood' using existing libraries so that users experience a coding-free environment to explore text data and extract meaningful insights with ease. By bridging the gap between data science and social science, the goal of TextViz Studio is to democratize access to advanced text analysis tools, empowering users to focus on content rather than code.


## Application Tools

### Text2Keywords: Keyword & Phrase Visualization

This tool helps uncover key themes and patterns in text files, such as PDFs or CSVs. Users can analyze word frequencies and visualize their results through word clouds and keyword summaries.

- PDF Text Extraction: Extract text from PDF documents seamlessly.
- Keyword Extraction: Identify high-frequency keywords or phrases.
- N-gram Analysis: Discover common word combinations (N-grams) in your data.
- Word Cloud Visualization: Generate visual representations of word frequencies.
- Customizable Parameters: Adjust frequency thresholds and N-gram ranges.
- Batch Processing: Analyze multiple PDFs or CSV files at once.
- Export Results: Download keyword counts, word clouds, and summaries for further use.

### Text2Topics: Large Language Topic Modeling

This advanced tool uses transformer-based models to perform topic modeling, revealing latent themes in text. It’s ideal for analyzing large corpora such as research papers, survey responses, or online discussions.

- Advanced Topic Modeling: Leverages BERTopic for transformer-based topic discovery.
- Interactive Visualization: Visualize topics and their relationships interactively.
- OpenAI Integration: Employs OpenAI GPT-4o for enhanced topic representation as an option.
- Customizable Parameters: Adjust the number of topics.
- Multiple Representations: Compare different topic representations for clarity.
- Text Summarization: Summarize topics and generate concise reports.
- Exportable Results: Download topics, summaries, and visualizations for reporting.

## Future Updates

This web-based Python application will continue to be updated and host additional tools for text analysis and eventually beyond this area. The following are big application tools that will be added in the following months:

  - **Text2Sentiment:** NLP tool to conduct both sentiment and emotion analyses on text from PDFs and CSVs, and display them in tables and visuals breaking them down.
  - **Data2Viz:** Exploratory data tool that will allow users to display relationships with customizable plots (e.g., line, bar, kernal, box, etc.).
  - **Stats2Viz:** Statistical data tool that will allow users to conduct regressions, inference, clustering/classification, time-series, and causal inference while producing output tables and plots.

## References

This section will be continuously changing as newer and more efficient resources continue to be released. As of October 16,2024, the following resources are used to perform the variety of analyses wrapped in the application overall:
- **Text2Topics:** [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for the sentence-embedding model in BERTopic, and [FLAN-T5 base](https://huggingface.co/google/flan-t5-base) for one of the representation models to produce topic labels.
- **Text2Sentiment:** The [NRC Word-Emotion Association Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm) as the default dictionary for sentiment and emotion classification of text ([Mohammad 2021](https://arxiv.org/abs/2005.11882)), the [VADER](https://github.com/cjhutto/vaderSentiment) lexicon ([Hutto & Gilbert 2014](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)) and the [nli-distilroberta-base](https://huggingface.co/cross-encoder/nli-distilroberta-base) natural language inference (NLI) model as additional options to classify sentiment from text.
