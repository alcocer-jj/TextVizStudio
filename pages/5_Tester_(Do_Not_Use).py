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

# --- Data Upload & Preview ---
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
num_models = st.number_input("How many models would you like to run?", min_value=1, max_value=5, value=1, step=1)
model_configs = []
for i in range(num_models):
    with st.expander(f"Configure Model {i+1}", expanded=(i == 0)):
        dv = st.selectbox("Dependent variable", data.columns, key=f"dv_{i}")
        ivs = st.multiselect("Independent variables", [c for c in data.columns if c != dv], key=f"ivs_{i}")
        # Interaction terms
        interaction_options = []
        if len(ivs) >= 2:
            pairs = [(ivs[a], ivs[b]) for a in range(len(ivs)) for b in range(a+1, len(ivs))]
            interaction_options = [f"{x}_x_{y}" for x, y in pairs]
        if interaction_options:
            selected_ints = st.multiselect("Add interaction terms", interaction_options, key=f"ints_{i}")
            for inter in selected_ints:
                x, y = inter.split("_x_")
                if inter not in data.columns:
                    data[inter] = data[x] * data[y]
                ivs.append(inter)
        estimators = st.multiselect(
            "Estimators to run:",
            ["OLS", "LPM", "Fixed Effects", "Random Effects", "Mixed Effects"],
            default=["OLS"], key=f"ests_{i}"
        )
        # Panel identifiers
        entity = time = None
        if any(m in estimators for m in ["Fixed Effects", "Random Effects"]):
            st.markdown("**Panel identifiers**")
            entity = st.selectbox("Entity (panel unit)", data.columns, key=f"ent_{i}")
            time = st.selectbox("Time (period)"    , data.columns, key=f"time_{i}")
        # Mixed Effects settings
        mixed_group = mixed_slopes = None
        if "Mixed Effects" in estimators:
            st.markdown("**Mixed Effects settings**")
            mixed_group  = st.selectbox("Grouping variable (random intercept)", data.columns, key=f"mg_{i}")
            mixed_slopes = st.multiselect("Random slopes (optional)", ivs, key=f"ms_{i}")
        # Standard errors
        se_type = st.selectbox(
            "Standard errors:",
            ["Standard", "White (HC0)", "Robust (HC1)", "Clustered"],
            key=f"se_{i}",
            help="Standard: homoskedastic; White: HC0; Robust: HC1; Clustered: choose var"
        )
        cluster_var = None
        if se_type == "Clustered":
            cluster_var = st.selectbox("Cluster variable:", data.columns, key=f"cl_{i}")
        model_configs.append({
            "dv": dv,
            "ivs": ivs,
            "ests": estimators,
            "entity": entity,
            "time": time,
            "mixed_group": mixed_group,
            "mixed_slopes": mixed_slopes,
            "se": se_type,
            "cluster_var": cluster_var
        })

# --- Run Models ---
if st.button("Run All Models"):
    results = {}
    for idx, cfg in enumerate(model_configs, start=1):
        formula = f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"
        # OLS/LPM covariance map
        ols_cov = {
            "Standard":    {"cov_type": "nonrobust"},
            "White (HC0)": {"cov_type": "HC0"},
            "Robust (HC1)": {"cov_type": "HC1"}
        }
        if cfg["se"] == "Clustered":
            ols_cov["Clustered"] = {
                "cov_type": "cluster",
                "cov_kwds": {"groups": data[cfg["cluster_var"]]}
            }
        # Fit OLS
        if "OLS" in cfg["ests"]:
            results[f"Model{idx}_OLS"] = smf.ols(formula, data=data).fit(**ols_cov[cfg["se"]])
        # Fit LPM
        if "LPM" in cfg["ests"]:
            results[f"Model{idx}_LPM"] = smf.ols(formula, data=data).fit(**ols_cov[cfg["se"]])
        # Panel data setup
        if any(m in cfg["ests"] for m in ["Fixed Effects", "Random Effects"]):
            panel = data.set_index([cfg["entity"], cfg["time"]])
            y = panel[cfg["dv"]]
            X = panel[cfg["ivs"]]
            # Panel covariance map
            panel_cov = {
                "Standard":    {"cov_type": "unadjusted"},
                "White (HC0)": {"cov_type": "robust"},
                "Robust (HC1)": {"cov_type": "robust"}
            }
            if cfg["se"] == "Clustered":
                if cfg["cluster_var"] == cfg["entity"]:
                    panel_cov["Clustered"] = {"cov_type": "clustered", "cluster_entity": True}
                elif cfg["cluster_var"] == cfg["time"]:
                    panel_cov["Clustered"] = {"cov_type": "clustered", "cluster_time": True}
                else:
                    st.warning(f"Clustering by {cfg['cluster_var']} not supported for panel; defaulting to entity.")
                    panel_cov["Clustered"] = {"cov_type": "clustered", "cluster_entity": True}
            # Fit Fixed Effects
            if "Fixed Effects" in cfg["ests"]:
                fe_mod = PanelOLS(y, X, entity_effects=True)
                results[f"Model{idx}_FE"] = fe_mod.fit(**panel_cov[cfg["se"]])
            # Fit Random Effects
            if "Random Effects" in cfg["ests"]:
                re_mod = RandomEffects(y, X)
                results[f"Model{idx}_RE"] = re_mod.fit(**panel_cov[cfg["se"]])
        # Fit Mixed Effects
        if "Mixed Effects" in cfg["ests"]:
            re_formula = None
            if cfg.get("mixed_slopes"):
                re_formula = "~ " + " + ".join(cfg["mixed_slopes"])
            mixed = MixedLM(formula, data, groups=data[cfg["mixed_group"]], re_formula=re_formula)
            results[f"Model{idx}_ME"] = mixed.fit()
    # --- Display Results ---
    for name, res in results.items():
        st.subheader(name)
        summ = res.summary() if callable(res.summary) else res.summary
        text = summ.as_text() if hasattr(summ, 'as_text') else str(summ)
        st.code(text)
    # Combined Stargazer table
    statsmods = [r for r in results.values() if hasattr(r, 'params')]
    if len(statsmods) > 0:
        st.subheader("Results Table")
        html = Stargazer(statsmods).render_html()
        st.components.v1.html(html, height=400)
        latex = Stargazer(statsmods).render_latex()
        st.download_button("Download LaTeX Table", latex, "regression_table.tex")
