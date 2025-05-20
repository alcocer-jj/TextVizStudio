import streamlit as st
import pandas as pd
import chardet
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
from linearmodels.panel import PanelOLS, RandomEffects

st.set_page_config(
    page_title="TBD",
    layout="wide"
)
st.title("TBD")

# --- Data Upload and Preview ---
uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if not uploaded:
    st.info("Please upload a CSV file to get started.")
    st.stop()

@st.cache_data
def load_data(file):
    raw = file.read()
    encoding = chardet.detect(raw)["encoding"]
    file.seek(0)
    return pd.read_csv(file, encoding=encoding)

data = load_data(uploaded)
st.subheader("Data Preview")
st.dataframe(data.head(5))

# --- Dynamic Model Configuration ---
num_models = st.number_input(
    "How many models would you like to run?", min_value=1, max_value=5, value=1, step=1
)
model_configs = []
for i in range(num_models):
    with st.expander(f"Configure Model {i+1}", expanded=(i == 0)):
        dv = st.selectbox(
            "Dependent variable", data.columns, key=f"dv_{i}"
        )
        ivs = st.multiselect(
            "Independent variables", [c for c in data.columns if c != dv],
            key=f"ivs_{i}"
        )
        estimators = st.multiselect(
            "Estimators to run:",
            ["OLS", "LPM", "Fixed Effects", "Random Effects"],
            default=["OLS"], key=f"ests_{i}"
        )
        entity, time = None, None
        if any(e in estimators for e in ["Fixed Effects", "Random Effects"]):
            entity = st.selectbox("Panel: Entity ID", data.columns, key=f"ent_{i}")
            time = st.selectbox("Panel: Time ID", data.columns, key=f"time_{i}")
        se_type = st.selectbox(
            "Standard errors:",
            ["Standard", "White (HC0)", "Robust (HC1)", "Clustered"],
            key=f"se_{i}",
            help=(
                "Standard: assume homoskedasticity; "
                "White (HC0): heteroskedasticity-robust (HC0); "
                "Robust (HC1): heteroskedasticity-robust (HC1); "
                "Clustered: cluster at chosen level"
            )
        )
        cluster_var = None
        if se_type == "Clustered":
            cluster_var = st.selectbox(
                "Select cluster variable:", data.columns, key=f"cluster_{i}"
            )
        model_configs.append({
            "dv": dv,
            "ivs": ivs,
            "ests": estimators,
            "entity": entity,
            "time": time,
            "se": se_type,
            "cluster_var": cluster_var
        })

# --- Run Models ---
if st.button("Run All Models"):
    results = {}
    for idx, cfg in enumerate(model_configs, start=1):
        formula = f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"
        # Map SE choices to covariance kwargs
        covmap = {
            "Standard":    {"cov_type": "nonrobust"},
            "White (HC0)": {"cov_type": "HC0"},
            "Robust (HC1)": {"cov_type": "HC1"}
        }
        if cfg["se"] == "Clustered":
            groups = data[cfg["cluster_var"]]
            covmap[cfg["se"]] = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}

        # Fit OLS
        if "OLS" in cfg["ests"]:
            results[f"Model{idx}_OLS"] = (
                smf.ols(formula, data=data)
                   .fit(**covmap[cfg["se"]])
            )
        # Fit LPM (use same SE mapping)
        if "LPM" in cfg["ests"]:
            results[f"Model{idx}_LPM"] = (
                smf.ols(formula, data=data)
                   .fit(**covmap[cfg["se"]])
            )
        # Fit Fixed Effects
        if "Fixed Effects" in cfg["ests"]:
            panel = data.set_index([cfg["entity"], cfg["time"]])
            results[f"Model{idx}_FE"] = (
                PanelOLS.from_formula(formula, panel, entity_effects=True)
                        .fit(**covmap[cfg["se"]])
            )
        # Fit Random Effects
        if "Random Effects" in cfg["ests"]:
            panel = data.set_index([cfg["entity"], cfg["time"]])
            results[f"Model{idx}_RE"] = (
                RandomEffects.from_formula(formula, panel)
                             .fit(**covmap[cfg["se"]])
            )

    # --- Display Results ---
    for name, res in results.items():
        st.subheader(name)
        st.code(res.summary().as_text())

    # --- Combined Table for multiple statsmodels models ---
    statsmods = [res for res in results.values() if hasattr(res, 'params')]
    if len(statsmods) > 0:
        st.subheader("Combined Results Table")
        st.write("Rendered with Stargazer:")
        html = Stargazer(statsmods).render_html()
        st.components.v1.html(html, height=400)
        latex = Stargazer(statsmods).render_latex()
        st.download_button("Download LaTeX Table", latex, "regression_table.tex")
