import streamlit as st
from sympy import div
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import scipy.stats as stats
from io import BytesIO, StringIO
import zipfile
import gspread
from oauth2client.service_account import ServiceAccountCredentials


st.set_page_config(
    page_title="StatsDashboard",
    layout="wide"
)

# Authenticate with Google Sheets API using Streamlit Secrets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open("TextViz Studio Feedback").sheet1


# Feedback form in the sidebar
st.sidebar.markdown("### **Feedback**")
feedback = st.sidebar.text_area("Experiencing bugs/issues? Have ideas to better the application tool?", placeholder="Leave feedback or error code here")

# Submit feedback
if st.sidebar.button("Submit"):
    if feedback:
        sheet.append_row(["StatsDashboard: ", feedback])
        st.sidebar.success("Thank you for your feedback!")
    else:
        st.sidebar.error("Feedback cannot be empty!")

st.sidebar.markdown("")

st.sidebar.markdown("For full documentation and future updates to the appliction, check the [GitHub Repository](https://github.com/alcocer-jj/TextVizStudio)")

st.sidebar.markdown("")

st.sidebar.markdown("Citation: Alcocer, J. J. (2024). StatsViz Studio (Version 1.2) [Software]. Retrieved from https://textvizstudio.streamlit.app/")


# Sidebar: Title and description
#st.title("Text2Keywords: Keyword & Phrase Visualization")
st.markdown("<h1 style='text-align: center'>StatsDashboard</h1>", unsafe_allow_html=True)


st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")

st.markdown("""
**StatsDashboard** is an all-in-one tool for statistical data exploration, visualization, and analysis, 
designed to accommodate a wide range of data analysis needs. Upload your dataset in CSV format and gain 
immediate access to powerful tools for data subsetting, filtering, and dynamic column selection. Dive 
into comprehensive descriptive statistics, correlation analysis, and statistical tests such as 
T-tests, Chi-Square, ANOVA, Mann-Whitney, and Kruskal-Wallis, enabling robust hypothesis testing for 
both categorical and numeric data. Additionally, the module includes customizable proportion tables with 
optional weighted calculations—ideal for survey or demographic analysis.

The intuitive visualization suite offers a variety of plot types, including histograms, scatter plots, 
line plots, regression plots, bar charts, and box plots. Each visualization is fully customizable with 
options for grouping, theming, and titling, along with aggregation functions to streamline data summaries. 
For easy sharing and reporting, StatsDashboard allows you to export plots in high-quality PNG and interactive
HTML formats. Whether you’re performing exploratory data analysis, testing for statistical associations, 
or preparing detailed reports, StatsDashboard offers a flexible, user-friendly workflow for in-depth,
tailored analysis and visualization of your data.
""")

st.markdown("")
st.markdown("")

# Helper function for downloading Plotly figures
def get_plotly_download(fig, file_format="png", scale=3):
    if file_format == "png":
        buffer = BytesIO()
        fig.write_image(buffer, format="png", scale=scale)
        buffer.seek(0)
    elif file_format == "html":
        html_str = StringIO()
        fig.write_html(html_str)
        buffer = BytesIO(html_str.getvalue().encode("utf-8"))
        buffer.seek(0)
    return buffer

# Descriptive Statistics & Exploratory Analysis
st.header("Import Data", divider=True)

# Data Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Define a function to initialize session state
    def initialize_session_state():
        if 'selected_columns' not in st.session_state:
            st.session_state.selected_columns = data.columns.tolist()
        if 'numeric_filters' not in st.session_state:
            st.session_state.numeric_filters = {}
        if 'filter_mode' not in st.session_state:
            st.session_state.filter_mode = {}
        if 'categorical_filters' not in st.session_state:
            st.session_state.categorical_filters = {}

    # Initialize session state at the start
    initialize_session_state()

    st.header("Descriptive Statistics", divider=True)
    # Define layout with columns: filters on the left, DataFrame on the right
    col1, col2 = st.columns([1, 3])

    with col1:
        
        # Reset Button Logic
        if st.button("Reset Filters"):
            # Clear all session state values to reset the app
            st.session_state.clear()
            # Reinitialize session state to default values after reset
            initialize_session_state()
        
        # Column Selection (always displayed to control visible columns)
        selected_columns = st.multiselect(
            "Select columns to display", 
            options=data.columns, 
            default=st.session_state.selected_columns, 
            key="selected_columns"
        )

        # Dropdown to enable or disable Data Subset Filters
        subset_option = st.selectbox("Subset Data Options", ["No Subset", "Enable Subset"])

        # Update `selected_data` based on selected columns
        selected_data = data[selected_columns].copy()

        # Display subset options only if "Enable Subset" is selected
        if subset_option == "Enable Subset":
            st.header("Data Subset Filters")

            for col in selected_data.columns:
                if pd.api.types.is_numeric_dtype(selected_data[col]):
                    # For numeric columns: Single Value or Range options
                    st.write(f"#### Filter for Numeric Column: {col}")
                    filter_mode = st.selectbox(f"Filter type for {col}", ["Range", "Single Value"], key=f"filter_mode_{col}")
                    st.session_state.filter_mode[col] = filter_mode

                    if filter_mode == "Range":
                        # Two side-by-side text inputs for range filtering
                        from_col, to_col = st.columns(2)
                        with from_col:
                            from_value = st.number_input(f"From", key=f"from_{col}", value=float(selected_data[col].min()) if not pd.isna(selected_data[col].min()) else 0.0)
                        with to_col:
                            to_value = st.number_input(f"To", key=f"to_{col}", value=float(selected_data[col].max()) if not pd.isna(selected_data[col].max()) else 1.0)
                        
                        # Filter data based on the range
                        selected_data = selected_data[(selected_data[col] >= from_value) & (selected_data[col] <= to_value)]

                    elif filter_mode == "Single Value":
                        # Side-by-side layout for comparison operator and value input
                        operator_col, value_col = st.columns(2)
                        with operator_col:
                            operator = st.selectbox(f"Operator", ["<", ">", "="], key=f"operator_{col}")
                        with value_col:
                            single_val = st.number_input(f"Value", key=f"value_{col}", value=float(selected_data[col].median()) if not pd.isna(selected_data[col].median()) else 0.0)

                        # Apply the single value condition based on the selected operator
                        if operator == "<":
                            selected_data = selected_data[selected_data[col] < single_val]
                        elif operator == ">":
                            selected_data = selected_data[selected_data[col] > single_val]
                        elif operator == "=":
                            selected_data = selected_data[selected_data[col] == single_val]

                elif pd.api.types.is_object_dtype(selected_data[col]) or pd.api.types.is_categorical_dtype(selected_data[col]):
                    # For categorical columns: Multi-select option
                    st.write(f"#### Filter for Categorical Column: {col}")
                    unique_vals = selected_data[col].dropna().unique()
                    selected_vals = st.multiselect(
                        f"Select values for {col}:", 
                        options=unique_vals, 
                        default=unique_vals, 
                        key=f"filter_{col}"
                    )
                    st.session_state.categorical_filters[col] = selected_vals
                    selected_data = selected_data[selected_data[col].isin(selected_vals)]

    with col2:        

        # Define layout with columns: filters on the left, DataFrame on the right
        colSUM, colType = st.columns(2)
        with colSUM:
            # Display Summary Statistics below Data Preview
            st.write("### Summary Statistics")
            #st.write(selected_data.describe().T)
            described_data = selected_data.describe().T
            st.dataframe(described_data, height=200, use_container_width=True)

        with colType:
            # Display Data Type Information below Summary Statistics
            st.write("### Data Types")
            data_info = pd.DataFrame({
                "Column Name": selected_data.columns,
                "Data Type": selected_data.dtypes,
                "NA Count": selected_data.isna().sum()
            }).reset_index(drop=True)
            #st.write(data_info)
            st.dataframe(data_info, height=200, use_container_width=True)

        # Display the DataFrame (filtered if subset enabled, otherwise full data)
        st.write("### Data Preview")
        st.dataframe(selected_data, use_container_width=True)

        # Proportion Tables Section
        st.write("### Proportion Table")

        colProp, colPropTable = st.columns(2)
        with colProp:
            # User selects categorical variables for the proportion table
            cat_var1 = st.selectbox("Select first categorical variable", ["None"] + list(selected_data.select_dtypes(include=['object', 'category']).columns))
            cat_var2 = st.selectbox("Select second categorical variable", ["None"] + [col for col in selected_data.select_dtypes(include=['object', 'category']).columns if col != cat_var1])

            # Weight selection (only numeric columns are shown)
            weight_column = st.selectbox("Choose a weight column (optional)", ["None"] + list(selected_data.select_dtypes(include=['int64', 'float64']).columns))

            # Proportion table type selection
            normalize_option = st.selectbox("Normalize by", ["Total", "Row", "Column"], index=0)

            # Initialize proportion_table as None
            proportion_table = None

            # Generate and display proportion table if valid variables are selected
            if cat_var1 != "None" and cat_var2 != "None":
                if weight_column != "None":
                    # Apply weighted counts to create the proportion table
                    weighted_counts = pd.crosstab(
                        selected_data[cat_var1],
                        selected_data[cat_var2],
                        values=selected_data[weight_column],
                        aggfunc='sum'
                        )
        
                    # Normalize weighted counts based on user selection
                    if normalize_option == "Total":
                        proportion_table = weighted_counts / weighted_counts.sum().sum()
                    elif normalize_option == "Row":
                        proportion_table = weighted_counts.div(weighted_counts.sum(axis=1), axis=0)
                    elif normalize_option == "Column":
                        proportion_table = weighted_counts.div(weighted_counts.sum(axis=0), axis=1)
                else:
                    # Map the user selection to pandas-compatible normalize argument
                    normalize_map = {"Total": "all", "Row": "index", "Column": "columns"}
                    normalize_arg = normalize_map[normalize_option]

                    # Standard unweighted proportion table
                    proportion_table = pd.crosstab(selected_data[cat_var1], selected_data[cat_var2], normalize=normalize_arg)
                
        with colPropTable:        
            # Display and download proportion table if it was created
            if proportion_table is not None:
                st.dataframe(proportion_table)
            else:
                st.warning("Please select two categorical variables to generate the proportion table.")

    # Layout: Second Row - Correlation Analysis and Chi-Square Test
    st.header("Statistical Tests & Hypothesis Testing", divider=True)
    col3, col4, col5 = st.columns(3)
    # --- Left Column: Correlation Analysis ---
    with col3:
        with st.expander("Correlation Coefficients", expanded=False):
            st.write("Select two or more numeric variables to calculate the correlation coefficient.")
            
            # Select only numeric variables from `selected_data` for correlation analysis
            numeric_vars = selected_data.select_dtypes(include=['int64', 'float64']).columns
            selected_vars = st.multiselect("Select variables for correlation", numeric_vars)

            # Check if at least two variables are selected for correlation analysis
            if len(selected_vars) >= 2:
                # Choose correlation method
                correlation_type = st.selectbox(
                    "Select correlation method", 
                    ["Pearson (Ideal for normally distributed data)", 
                     "Spearman (Ideal for non-normally distributed data)", 
                     "Kendall (Ideal for small datasets or ordinal data)"]
                    )
                # Apply selected correlation method on `selected_data`
                if correlation_type.startswith("Pearson"):
                    correlation_matrix = selected_data[selected_vars].corr(method="pearson")
                elif correlation_type.startswith("Spearman"):
                    correlation_matrix = selected_data[selected_vars].corr(method="spearman")
                elif correlation_type.startswith("Kendall"):
                    correlation_matrix = selected_data[selected_vars].corr(method="kendall")

                # Display correlation matrix as a table
                st.write("Correlation Matrix")
                st.dataframe(correlation_matrix)

                # Plot correlation heatmap using Plotly
                heatmap_fig = px.imshow(
                    correlation_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap"
                )
                heatmap_fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
                st.plotly_chart(heatmap_fig)

                # Helper function to create a downloadable HTML file for the Plotly figure
                def get_plotly_download_html(fig):
                    html_str = StringIO()
                    fig.write_html(html_str)
                    buffer = BytesIO(html_str.getvalue().encode("utf-8"))
                    buffer.seek(0)
                    return buffer

                # Download option for the correlation heatmap as HTML
                heatmap_html = get_plotly_download_html(heatmap_fig)

                st.download_button(
                    label="Download Correlation Heatmap as HTML (Interactive)",
                    data=heatmap_html,
                    file_name="correlation_heatmap.html",
                    mime="text/html"
                )

    # --- Middle Column: t-tests ---
    # T-Test Analysis in col4
    with col4:
        with st.expander("T-Tests", expanded=False):
            st.write("Select one numeric variable and one categorical variable to perform the T-Test.")

            # Select numeric and categorical variables
            numeric_var = st.selectbox("Select numeric variable", ["None"] + list(selected_data.select_dtypes(include=['int64', 'float64']).columns))
            categorical_var = st.selectbox("Select categorical variable", ["None"] + list(selected_data.select_dtypes(include=['object', 'category']).columns))

            # Option for one-tailed or two-tailed test
            tail_option = st.selectbox("Select test type", ["Two-Tailed", "One-Tailed"])

            # Check if both a numeric and a categorical variable are selected
            if numeric_var != "None" and categorical_var != "None":
                # Check that the categorical variable has exactly two unique values (two-sample T-Test requirement)
                unique_groups = selected_data[categorical_var].dropna().unique()
                if len(unique_groups) == 2:
                    # Split the numeric data by the groups in the categorical variable
                    group1 = selected_data[selected_data[categorical_var] == unique_groups[0]][numeric_var]
                    group2 = selected_data[selected_data[categorical_var] == unique_groups[1]][numeric_var]

                    # Perform T-Test
                    t_stat, p_value = stats.ttest_ind(group1, group2, nan_policy='omit')

                    # Adjust p-value for one-tailed test if selected
                    if tail_option == "One-Tailed":
                        # Divide p-value by 2 for one-tailed test and check t-stat direction
                        p_value /= 2
                        # For a one-tailed test, p-value is valid if t-stat is in the direction of the alternative hypothesis
                        if t_stat < 0:
                            p_value = 1 - p_value

                    # Display T-Test results
                    st.write("### T-Test Results")
                    st.write(f"T-Statistic: {t_stat}")
                    st.write(f"P-Value: {p_value}")

                    # Interpretation based on p-value
                    if p_value < 0.05:
                        st.write(f"**Interpretation**: The means of `{numeric_var}` differ significantly between the groups in `{categorical_var}` (at the 5% significance level).")
                    else:
                        st.write(f"**Interpretation**: There is no significant difference in the means of `{numeric_var}` between the groups in `{categorical_var}`.")
                else:
                    st.write("Please select a categorical variable with exactly two unique groups for a two-sample T-Test.")

    # --- Right Column: Chi-Square Test for Independence ---
    with col5:
        with st.expander("Chi-Square Test of Independence", expanded=False):
            st.write("Select two or more categorical variables to perform pairwise Chi-Square Tests.")

            # Select multiple categorical variables from `selected_data`
            cat_vars = selected_data.select_dtypes(include=['object', 'category']).columns
            selected_vars = st.multiselect("Select categorical variables for Chi-Square Test", cat_vars)

            # Run Chi-Square Test for each pair of selected variables
            if len(selected_vars) >= 2:
                for i in range(len(selected_vars)):
                    for j in range(i + 1, len(selected_vars)):
                        var1, var2 = selected_vars[i], selected_vars[j]
                        st.write(f"### Chi-Square Test between `{var1}` and `{var2}`")

                        # Create a contingency table for the selected pair of variables
                        contingency_table = pd.crosstab(selected_data[var1], selected_data[var2])
                        st.write("Contingency Table:")
                        st.dataframe(contingency_table)

                        # Perform the Chi-Square test
                        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

                        # Display Chi-Square test results
                        st.write("### Chi-Square Test Results")
                        st.write(f"Chi-Square Statistic: {chi2}")
                        st.write(f"Degrees of Freedom: {dof}")
                        st.write(f"P-Value: {p}")

                        # Interpretation based on p-value
                        if p < 0.05:
                            st.write("**Interpretation**: There is a significant association between the variables.")
                        else:
                            st.write("**Interpretation**: There is no significant association between the variables.")


    # Define layout for ANOVA, Mann-Whitney, and Kruskal-Wallis tests
    col_anova, col_mann_whitney, col_kruskal_wallis = st.columns(3)

    # ANOVA Test in col_anova
    with col_anova:
        with st.expander("ANOVA Test", expanded=False):
            st.write("Select a numeric variable and a categorical variable with three or more groups.")

            # Select numeric and categorical variables
            numeric_var = st.selectbox("Select numeric variable for ANOVA", ["None"] + list(selected_data.select_dtypes(include=['int64', 'float64']).columns))
            categorical_var = st.selectbox("Select categorical variable for ANOVA", ["None"] + list(selected_data.select_dtypes(include=['object', 'category']).columns))

            # Check if a valid numeric and categorical variable are selected
            if numeric_var != "None" and categorical_var != "None":
                # Extract groups based on the categorical variable
                groups = [group[numeric_var].dropna() for name, group in selected_data.groupby(categorical_var)]
            
                if len(groups) >= 3:
                    # Perform ANOVA
                    f_statistic, p_value = stats.f_oneway(*groups)

                    # Display results
                    st.write("### ANOVA Test Results")
                    st.write(f"F-Statistic: {f_statistic}")
                    st.write(f"P-Value: {p_value}")

                    # Interpretation based on p-value
                    if p_value < 0.05:
                        st.write(f"**Interpretation**: There is a significant difference in `{numeric_var}` between the groups in `{categorical_var}`.")
                    else:
                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` across groups in `{categorical_var}`.")
                else:
                    st.write("Please select a categorical variable with three or more unique groups.")


    # Mann-Whitney Test in col_mann_whitney
    with col_mann_whitney:
        with st.expander("Mann-Whitney U Test", expanded=False):
            st.write("Select a numeric variable and a categorical variable with exactly two groups.")

            # Select numeric and categorical variables
            numeric_var = st.selectbox("Select numeric variable for Mann-Whitney", ["None"] + list(selected_data.select_dtypes(include=['int64', 'float64']).columns))
            categorical_var = st.selectbox("Select categorical variable for Mann-Whitney", ["None"] + list(selected_data.select_dtypes(include=['object', 'category']).columns))

            # Check if valid variables are selected
            if numeric_var != "None" and categorical_var != "None":
                unique_groups = selected_data[categorical_var].dropna().unique()
                if len(unique_groups) == 2:
                    # Extract the two groups
                    group1 = selected_data[selected_data[categorical_var] == unique_groups[0]][numeric_var]
                    group2 = selected_data[selected_data[categorical_var] == unique_groups[1]][numeric_var]

                    # Perform Mann-Whitney U Test
                    u_statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                    # Display results
                    st.write("### Mann-Whitney U Test Results")
                    st.write(f"U-Statistic: {u_statistic}")
                    st.write(f"P-Value: {p_value}")

                    # Interpretation
                    if p_value < 0.05:
                        st.write(f"**Interpretation**: The distribution of `{numeric_var}` differs significantly between the two groups in `{categorical_var}`.")
                    else:
                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution between the groups in `{categorical_var}`.")
            else:
                st.write("Please select a categorical variable with exactly two unique groups.")

    # Kruskal-Wallis Test in col_kruskal_wallis
    with col_kruskal_wallis:
        with st.expander("Kruskal-Wallis Test", expanded=False):
            st.write("Select a numeric variable and a categorical variable with three or more groups.")

            # Select numeric and categorical variables
            numeric_var = st.selectbox("Select numeric variable for Kruskal-Wallis", ["None"] + list(selected_data.select_dtypes(include=['int64', 'float64']).columns))
            categorical_var = st.selectbox("Select categorical variable for Kruskal-Wallis", ["None"] + list(selected_data.select_dtypes(include=['object', 'category']).columns))

            # Check if a valid numeric and categorical variable are selected
            if numeric_var != "None" and categorical_var != "None":
                # Extract groups based on the categorical variable
                groups = [group[numeric_var].dropna() for name, group in selected_data.groupby(categorical_var)]
            
                if len(groups) >= 3:
                    # Perform Kruskal-Wallis Test
                    h_statistic, p_value = stats.kruskal(*groups)

                    # Display results
                    st.write("### Kruskal-Wallis Test Results")
                    st.write(f"H-Statistic: {h_statistic}")
                    st.write(f"P-Value: {p_value}")

                    # Interpretation
                    if p_value < 0.05:
                        st.write(f"**Interpretation**: The distribution of `{numeric_var}` differs significantly across groups in `{categorical_var}`.")
                    else:
                        st.write(f"**Interpretation**: No significant difference in `{numeric_var}` distribution across groups in `{categorical_var}`.")
                else:
                    st.write("Please select a categorical variable with three or more unique groups.")

    # Visualization section with option to generate multiple plots
    st.header("Data Visualization", divider=True)
    num_plots = st.selectbox("Select the number of plots to generate", [1, 2, 3, 4], index=0)

    for i in range(num_plots):
        st.write(f"### Plot {i + 1}")
        left_col, right_col = st.columns([1, 3])

        with left_col:
            # Plot controls for each plot
            selected_var = st.selectbox(f"Select Y variable for Plot {i + 1}", selected_data.columns, key=f"y_{i}")
            secondary_var = st.selectbox(f"Select X variable for Plot {i + 1} (optional)", ["None"] + list(selected_data.columns), key=f"x_{i}")
            group_by_var = st.selectbox(f"Select a grouping variable for Plot {i + 1} (optional)", ["None"] + list(selected_data.columns), key=f"group_{i}")

            # Determine available plot types based on variable selection
            if secondary_var == "None":
                plot_options = ["Histogram Plot", "Bar Plot", "Box Plot"]  # Plots requiring only a primary variable
            else:
                plot_options = ["Histogram Plot", "Scatter Plot", "Line Plot", "Regression Plot", "Bar Plot", "Box Plot"]

            # Select plot type, filtering based on available options
            plot_type = st.radio(f"Select Plot Type for Plot {i + 1}", plot_options, index=0, key=f"type_{i}")

            # Display a warning if the plot type requires a secondary variable but it's not selected
            if plot_type in ["Scatter Plot", "Line Plot", "Regression Plot"] and secondary_var == "None":
                st.warning(f"{plot_type} requires both a Y and an X variable. Please select a secondary variable (X) for this plot type.")
    
            # Aggregation option for applicable plot types
            use_aggregation = st.checkbox(f"Apply Aggregation Function to Plot", key=f"agg_{i}")
            if use_aggregation:
                aggregation_function = st.selectbox(f"Select aggregation function for Plot", ["sum", "avg", "count", "min", "max"], key=f"agg_func_{i}")

            # Title customization option
            default_title = f"{plot_type} of {selected_var}" + (f" vs {secondary_var}" if secondary_var != "None" else "") + (f" grouped by {group_by_var}" if group_by_var != "None" else "")
            plot_title = st.text_input(f"Set title for Plot {i + 1}", value=default_title, key=f"title_{i}")

            # Select Plot Theme
            plot_theme = st.selectbox(f"Select Plot Theme for Plot {i + 1}", ["ggplot2", "seaborn", "simple_theme", "none"], key=f"theme_{i}")
            theme = {"ggplot2": "ggplot2", "seaborn": "plotly_white", "simple_theme": "simple_white", "none": None}.get(plot_theme)

            # Determine grouping variables and perform aggregation only if grouping_vars is not empty
            grouping_vars = [var for var in [secondary_var, group_by_var] if var != "None"]
    
            # Set default aggregated_data to selected_data in case no aggregation is applied
            aggregated_data = selected_data  # Default to non-aggregated data
    
            if use_aggregation:
                if grouping_vars:
                    # Apply aggregation only when there are grouping variables
                    if aggregation_function == "sum":
                        aggregated_data = selected_data.groupby(grouping_vars)[selected_var].sum().reset_index()
                    elif aggregation_function == "avg":
                        aggregated_data = selected_data.groupby(grouping_vars)[selected_var].mean().reset_index()
                    elif aggregation_function == "count":
                        aggregated_data = selected_data.groupby(grouping_vars)[selected_var].count().reset_index()
                    elif aggregation_function == "min":
                        aggregated_data = selected_data.groupby(grouping_vars)[selected_var].min().reset_index()
                    elif aggregation_function == "max":
                        aggregated_data = selected_data.groupby(grouping_vars)[selected_var].max().reset_index()
                else:
                    # Show warning if no grouping variables are selected but aggregation is enabled
                    st.warning("To use an aggregation function, please select a secondary variable (X) or a grouping variable.")
            else:
                # Use subsetted data if no aggregation is applied
                aggregated_data = selected_data

        # Generate and display the Plotly figure
        with right_col:
            # Conditional plot creation based on plot_type
            if plot_type == "Histogram Plot":
                fig = px.histogram(aggregated_data, x=selected_var, color=group_by_var if group_by_var != "None" else None, nbins=30, template=theme,
                                   title=plot_title)
            elif plot_type == "Scatter Plot" and secondary_var != "None":
                fig = px.scatter(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, template=theme,
                                 title=plot_title)
            elif plot_type == "Line Plot" and secondary_var != "None":
                fig = px.line(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, template=theme,
                              title=plot_title)
            elif plot_type == "Regression Plot" and secondary_var != "None":
                fig = px.scatter(aggregated_data, x=secondary_var, y=selected_var, color=group_by_var if group_by_var != "None" else None, trendline="ols", template=theme,
                                 title=plot_title)
            elif plot_type == "Bar Plot":
                fig = px.bar(aggregated_data, y=selected_var, x=secondary_var if secondary_var != "None" else None, color=group_by_var if group_by_var != "None" else None,
                             template=theme, title=plot_title)
                fig.update_layout(barmode='group' if group_by_var != "None" else 'relative')
            elif plot_type == "Box Plot":
                fig = px.box(aggregated_data, y=selected_var, x=secondary_var if secondary_var != "None" else None, color=group_by_var if group_by_var != "None" else None,
                             template=theme, title=plot_title)

            # Display the plot only if it was successfully created
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
            
            
            # Download buttons for high-quality PNG and HTML
            png_buffer = get_plotly_download(fig, file_format="png", scale=3)
            html_buffer = get_plotly_download(fig, file_format="html")

            st.download_button(
                label="Download as PNG (High Quality)",
                data=png_buffer,
                file_name=f"plot_{i + 1}.png",
                mime="image/png"
            )

            st.download_button(
                label="Download as HTML (Interactive)",
                data=html_buffer,
                file_name=f"plot_{i + 1}.html",
                mime="text/html"
            )