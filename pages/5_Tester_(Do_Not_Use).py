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

# --- Model Configuration ---
num_models = st.number_input("How many models to run?", min_value=1, max_value=5, value=1, step=1)
model_configs = []
for i in range(num_models):
    with st.expander(f"Configure Model {i+1}", expanded=(i == 0)):
        dv = st.selectbox("Dependent variable", data.columns, key=f"dv_{i}")
        ivs = st.multiselect("Independent variables", [c for c in data.columns if c != dv], key=f"ivs_{i}")
        estimators = st.multiselect(
            "Estimators:", ["OLS", "LPM", "Fixed Effects", "Random Effects"],
            default=["OLS"], key=f"ests_{i}"
        )
        entity = time = None
        if any(m in estimators for m in ["Fixed Effects", "Random Effects"]):
            st.markdown("**Panel identifiers**")
            entity = st.selectbox("Entity (panel unit)", data.columns, key=f"ent_{i}")
            time = st.selectbox("Time (period)", data.columns, key=f"time_{i}")
        se_type = st.selectbox(
            "Standard errors:", ["Standard", "White (HC0)", "Robust (HC1)", "Clustered"],
            key=f"se_{i}", help="Standard: homoskedastic; White: HC0; Robust: HC1; Clustered: choose var"
        )
        cluster_var = None
        if se_type == "Clustered":
            cluster_var = st.selectbox("Cluster variable", data.columns, key=f"cluster_{i}")
        model_configs.append({"dv": dv, "ivs": ivs, "ests": estimators,
                               "entity": entity, "time": time,
                               "se": se_type, "cluster_var": cluster_var})

# --- Run & Display ---
if st.button("Run Models"):
    results = {}
    for idx, cfg in enumerate(model_configs, 1):
        formula = f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"

        # Build OLS/LPM covariance mapping per config
        ols_cov = {"Standard": {"cov_type": "nonrobust"},
                   "White (HC0)": {"cov_type": "HC0"},
                   "Robust (HC1)": {"cov_type": "HC1"}}
        if cfg['se'] == 'Clustered':
            # safe guard: cluster_var always set when se='Clustered'
            ols_cov['Clustered'] = {"cov_type": "cluster",
                                     "cov_kwds": {"groups": data[cfg['cluster_var']]}}

        # Fit OLS
        if 'OLS' in cfg['ests']:
            results[f"Model{idx}_OLS"] = smf.ols(formula, data=data).fit(**ols_cov[cfg['se']])
        # Fit LPM
        if 'LPM' in cfg['ests']:
            results[f"Model{idx}_LPM"] = smf.ols(formula, data=data).fit(**ols_cov[cfg['se']])

        # Prepare panel if needed
        if any(m in cfg['ests'] for m in ["Fixed Effects", "Random Effects"]):
            panel = data.set_index([cfg['entity'], cfg['time']])
            y = panel[cfg['dv']]
            X = panel[cfg['ivs']]

            # Panel covariance mapping per config
            panel_cov = {"Standard": {"cov_type": "unadjusted"},
                         "White (HC0)": {"cov_type": "robust"},
                         "Robust (HC1)": {"cov_type": "robust"}}
            if cfg['se'] == 'Clustered':
                # allow clustering by entity or time
                if cfg['cluster_var'] == cfg['entity']:
                    panel_cov['Clustered'] = {"cov_type": "clustered", "cluster_entity": True}
                elif cfg['cluster_var'] == cfg['time']:
                    panel_cov['Clustered'] = {"cov_type": "clustered", "cluster_time": True}
                else:
                    st.warning(f"Clustering panel on {cfg['cluster_var']} not supported; default entity.")
                    panel_cov['Clustered'] = {"cov_type": "clustered", "cluster_entity": True}

            # Fit Fixed Effects
            if 'Fixed Effects' in cfg['ests']:
                fe = PanelOLS(y, X, entity_effects=True)
                results[f"Model{idx}_FE"] = fe.fit(**panel_cov[cfg['se']])
            # Fit Random Effects
            if 'Random Effects' in cfg['ests']:
                re = RandomEffects(y, X)
                results[f"Model{idx}_RE"] = re.fit(**panel_cov[cfg['se']])

    # Show summaries
    for name, res in results.items():
        st.subheader(name)
        st.code(res.summary().as_text())

    # Combined Stargazer table if >1 statsmodels model
    statsmods = [r for r in results.values() if hasattr(r, 'params')]
    if len(statsmods) > 0:
        st.subheader("Combined Results Table")
        html = Stargazer(statsmods).render_html()
        st.components.v1.html(html, height=400)
        latex = Stargazer(statsmods).render_latex()
        st.download_button("Download LaTeX Table", latex, "regression_table.tex")