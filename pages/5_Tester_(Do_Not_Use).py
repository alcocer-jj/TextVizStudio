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
        dv = st.selectbox("Dependent variable", data.columns, key=f"dv_{i}")
        ivs = st.multiselect(
            "Independent variables", [c for c in data.columns if c != dv], key=f"ivs_{i}"
        )
        estimators = st.multiselect(
            "Estimators to run:",
            ["OLS", "LPM", "Fixed Effects", "Random Effects"],
            default=["OLS"], key=f"ests_{i}"
        )
        entity, time = None, None
        if any(e in estimators for e in ["Fixed Effects", "Random Effects"]):
            st.markdown("**Panel identifiers**")
            entity = st.selectbox("Entity (panel unit)", data.columns, key=f"ent_{i}")
            time   = st.selectbox("Time (period)", data.columns, key=f"time_{i}")
        se_type = st.selectbox(
            "Standard errors:",
            ["Standard", "White (HC0)", "Robust (HC1)", "Clustered"],
            key=f"se_{i}",
            help=("Standard: homoskedastic; White (HC0): heteroskedasticity-robust; "
                  "Robust (HC1): alternate robust estimator; Clustered: cluster at chosen level")
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
        # Build formula for OLS/LPM
        formula = f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"

        # OLS/LPM covariance mapping (statsmodels style)
        ols_covmap = {
            "Standard":    {"cov_type": "nonrobust"},
            "White (HC0)": {"cov_type": "HC0"},
            "Robust (HC1)": {"cov_type": "HC1"},
            "Clustered":   {"cov_type": "cluster", "cov_kwds": {"groups": data[cfg['cluster_var']]}}
        }

        # Panel covariance mapping (linearmodels style)
        panel_covmap = {
            "Standard":    {"cov_type": "unadjusted"},
            "White (HC0)": {"cov_type": "robust"},
            "Robust (HC1)": {"cov_type": "robust"},
            "Clustered":   {}
        }
        if cfg["se"] == "Clustered":
            # cluster at entity or time
            if cfg["cluster_var"] == cfg["entity"]:
                panel_covmap["Clustered"] = {"cov_type": "clustered", "cluster_entity": True}
            elif cfg["cluster_var"] == cfg["time"]:
                panel_covmap["Clustered"] = {"cov_type": "clustered", "cluster_time": True}
            else:
                st.warning(
                    f"Panel models only support clustering by panel entity or time – "
                    f"defaulting to entity for Model {idx}."
                )
                panel_covmap["Clustered"] = {"cov_type": "clustered", "cluster_entity": True}

        # --- Fit OLS & LPM ---
        if "OLS" in cfg["ests"]:
            results[f"Model{idx}_OLS"] = (
                smf.ols(formula, data=data)
                   .fit(**ols_covmap[cfg["se"]])
            )
        if "LPM" in cfg["ests"]:
            results[f"Model{idx}_LPM"] = (
                smf.ols(formula, data=data)
                   .fit(**ols_covmap[cfg["se"]])
            )

        # --- Prepare Panel data for FE/RE ---
        if any(e in cfg["ests"] for e in ["Fixed Effects", "Random Effects"]):
            panel = data.set_index([cfg["entity"], cfg["time"]])
            y = panel[cfg["dv"]]
            X = panel[cfg["ivs"]]

        # --- Fit Fixed Effects ---
        if "Fixed Effects" in cfg["ests"]:
            fe_mod = PanelOLS(y, X, entity_effects=True)
            results[f"Model{idx}_FE"] = fe_mod.fit(**panel_covmap[cfg["se"]])

        # --- Fit Random Effects ---
        if "Random Effects" in cfg["ests"]:
            re_mod = RandomEffects(y, X)
            results[f"Model{idx}_RE"] = re_mod.fit(**panel_covmap[cfg["se"]])

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
