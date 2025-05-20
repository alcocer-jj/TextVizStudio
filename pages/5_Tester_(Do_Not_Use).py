import streamlit as st
import pandas as pd
import chardet
import statsmodels.formula.api as smf
import statsmodels.discrete.discrete_model as discrete
import statsmodels.discrete.count_model as count
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.mixed_linear_model import MixedLM
from linearmodels.panel import PanelOLS, RandomEffects
from stargazer.stargazer import Stargazer

# --- Standard error mappings ---
COMMON_STATS_SE = {
    "Standard":    {"cov_type": "nonrobust", "cov_kwds": {}},
    "White (HC0)": {"cov_type": "HC0",      "cov_kwds": {}},
    "Robust (HC1)":{"cov_type": "HC1",      "cov_kwds": {}},
    "Clustered":   {"cov_type": "cluster",  "cov_kwds": {}}  # groups assigned later
}
PANEL_SE = {
    "Standard":    {"cov_type": "unadjusted",     "cov_kwds": {}},
    "White (HC0)": {"cov_type": "robust",         "cov_kwds": {}},
    "Robust (HC1)":{"cov_type": "robust",         "cov_kwds": {}},
    "Clustered":   {"cov_type": "clustered",      "cov_kwds": {}}  # cluster_entity/time later
}
MIXED_SE = {
    "Standard":    {"cov_type": "nonrobust",      "cov_kwds": {}},
    "Robust (HC1)":{"cov_type": "robust",         "cov_kwds": {}}
}

# --- Estimator registry ---
ESTIMATOR_MAP = {
    # Linear & discrete/count/ordered
    "OLS":    {"func": lambda f,df: smf.ols(f,df),                    "panel": False, "mixed": False},
    "LPM":    {"func": lambda f,df: smf.ols(f,df),                    "panel": False, "mixed": False},
    "Logit":  {"func": lambda f,df: discrete.Logit.from_formula(f,df),"panel": False, "mixed": False},
    "Probit": {"func": lambda f,df: discrete.Probit.from_formula(f,df),"panel": False, "mixed": False},
    "Multinomial Logit": {"func": lambda f,df: discrete.MNLogit.from_formula(f,df),"panel": False, "mixed": False},
    "Poisson":           {"func": lambda f,df: count.Poisson.from_formula(f,df),                "panel": False, "mixed": False},
    "Negative Binomial": {"func": lambda f,df: count.NegativeBinomialP.from_formula(f,df),        "panel": False, "mixed": False},
    "Zero-Inflated Poisson":      {"func": lambda f,df: count.ZeroInflatedPoisson.from_formula(f,df),      "panel": False, "mixed": False},
    "Zero-Inflated NB": {"func": lambda f, df: count.ZeroInflatedNegativeBinomialP.from_formula(formula=f, data=df, exog_infl=df[[c.strip() for c in f.split('~')[1].split('+')]], inflation='logit', p=1), "panel": False, "mixed": False},
    "Ordered Logit":  {"func": lambda f,df: OrderedModel.from_formula(f,df,distr="logit"), "panel": False, "mixed": False},
    "Ordered Probit": {"func": lambda f,df: OrderedModel.from_formula(f,df,distr="probit"),"panel": False, "mixed": False},
    # Panel & mixed
    "Fixed Effects":  {"func": None, "panel": True,  "mixed": False},
    "Random Effects": {"func": None, "panel": True,  "mixed": False},
    "Mixed Effects":  {"func": None, "panel": False, "mixed": True},
    # Placeholders
    "Tobit":    {"func": None, "panel": False, "mixed": False},
    "Hurdle":   {"func": None, "panel": False, "mixed": False},
    "Heckman":  {"func": None, "panel": True,  "mixed": False}
}

# --- Supported SE types per estimator ---
SUPPORTED_SE = {}
for est, v in ESTIMATOR_MAP.items():
    if v["mixed"]:
        SUPPORTED_SE[est] = set(MIXED_SE.keys())
    elif v["panel"]:
        SUPPORTED_SE[est] = set(PANEL_SE.keys())
    elif v["func"]:
        SUPPORTED_SE[est] = set(COMMON_STATS_SE.keys())
    else:
        SUPPORTED_SE[est] = {"Standard"}

# --- Streamlit UI ---
st.set_page_config(page_title="TBD", layout="wide")
st.title("TBD")

# Data upload
uploaded = st.file_uploader("Upload your CSV data", type=["csv"])
if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()
@st.cache_data
def load_data(f):
    raw = f.read(); enc = chardet.detect(raw)["encoding"]; f.seek(0)
    return pd.read_csv(f, encoding=enc)

data = load_data(uploaded)
st.subheader("Data Preview"); st.dataframe(data.head(5))

# Model configuration
num_models = st.number_input("Number of models to run", min_value=1, max_value=5, value=1)
configs = []
for i in range(num_models):
    with st.expander(f"Configure Model {i+1}", expanded=(i==0)):
        dv = st.selectbox("Dependent variable", data.columns, key=f"dv_{i}")
        ivs = st.multiselect("Independent variables", [c for c in data.columns if c!=dv], key=f"ivs_{i}")
        # interactions
        if len(ivs)>=2:
            pairs=[(ivs[a],ivs[b]) for a in range(len(ivs)) for b in range(a+1,len(ivs))]
            opts=[f"{x}_x_{y}" for x,y in pairs]
            sel=st.multiselect("Add interaction terms", opts, key=f"ints_{i}")
            for term in sel:
                x,y=term.split("_x_"); data[term]=data[x]*data[y]
                if term not in ivs: ivs.append(term)
        # estimator selection
        ests = st.multiselect("Estimators", list(ESTIMATOR_MAP.keys()), default=["OLS"], key=f"ests_{i}")
        # panel identifiers
        ent=time=mg=None; ms=[]
        if any(ESTIMATOR_MAP[e]["panel"] for e in ests):
            st.markdown("**Panel identifiers**")
            ent=st.selectbox("Entity ID", data.columns, key=f"ent_{i}")
            time=st.selectbox("Time ID", data.columns, key=f"time_{i}")
        # mixed settings
        if "Mixed Effects" in ests:
            st.markdown("**Mixed Effects settings**")
            mg=st.selectbox("Grouping variable", data.columns, key=f"mg_{i}")
            ms=st.multiselect("Random slopes", ivs, key=f"ms_{i}")
        # dynamic SE options
        common_sets=[SUPPORTED_SE[e] for e in ests]
        se_opts=sorted(set.intersection(*common_sets)) if common_sets else []
        se_type=st.selectbox("Standard errors", se_opts, key=f"se_{i}")
        cl=None
        if se_type=="Clustered": cl=st.selectbox("Cluster variable", data.columns, key=f"cl_{i}")
        configs.append({"dv":dv,"ivs":ivs,"ests":ests,"ent":ent,"time":time,"mg":mg,"ms":ms,"se":se_type,"cl":cl})

# Run models
if st.button("Run Models"):
    results={}
    for idx,cfg in enumerate(configs,1):
        form=f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"
        stats_cov=COMMON_STATS_SE.get(cfg['se'],{})
        panel_cov=PANEL_SE.get(cfg['se'],{})
        mixed_cov=MIXED_SE.get(cfg['se'],{})
        for est in cfg['ests']:
            key=f"Model{idx}_{est.replace(' ','')}"
            try:
                if ESTIMATOR_MAP[est]["func"]:
                    # statsmodels families
                    mod=ESTIMATOR_MAP[est]["func"](form,data)
                    covargs=stats_cov.copy()
                    if cfg['se']=="Clustered": covargs['cov_kwds']={'groups':data[cfg['cl']]}
                    if est == "Zero-Inflated NB":
                        res = mod.fit(method='bfgs', maxiter=500, disp=0)
                    else:
                        res = mod.fit(**covargs)
                elif ESTIMATOR_MAP[est]["panel"]:
                    panel_df=data.set_index([cfg['ent'],cfg['time']])
                    y=panel_df[cfg['dv']]; X=panel_df[cfg['ivs']]
                    if est=="Fixed Effects": mod=PanelOLS(y,X,entity_effects=True)
                    else: mod=RandomEffects(y,X)
                    pargs=panel_cov.copy()
                    if cfg['se']=="Clustered":
                        if cfg['cl']==cfg['ent']: pargs['cov_kwds']={'cluster_entity':True}
                        else: pargs['cov_kwds']={'cluster_time':True}
                    res=mod.fit(**pargs)
                elif ESTIMATOR_MAP[est]["mixed"]:
                    rf=None
                    if cfg['ms']: rf="~ "+" + ".join(cfg['ms'])
                    mod=MixedLM(form,data,groups=data[cfg['mg']],re_formula=rf)
                    res=mod.fit(**mixed_cov)
                else:
                    raise NotImplementedError(f"{est} not supported yet")
                results[key]=res
            except Exception as e:
                st.error(f"{est} failed: {e}")
    # Display results
    statsmods=[r for r in results.values() if hasattr(r,'params')]
    for name,res in results.items():
        st.subheader(name)
        summ=res.summary() if callable(res.summary) else res.summary
        txt=summ.as_text() if hasattr(summ,'as_text') else str(summ)
        st.code(txt)
    if len(statsmods)>0:
        st.subheader("Combined Results Table")
        html=Stargazer(statsmods).render_html()
        st.components.v1.html(html,height=500)
        st.download_button("Download LaTeX Table",Stargazer(statsmods).render_latex(),"regression_table.tex")
