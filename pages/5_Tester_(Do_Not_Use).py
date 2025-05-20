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

# Estimator registry with constructors and kwargs
ESTIMATOR_MAP = {
    # Linear
    "OLS":    {"func": lambda f, df: smf.ols(f, df), "fit_kwargs": {}},
    "LPM":    {"func": lambda f, df: smf.ols(f, df), "fit_kwargs": {}},
    # Discrete
    "Logit":  {"func": lambda f, df: discrete.Logit.from_formula(f, df), "fit_kwargs": {}},
    "Probit": {"func": lambda f, df: discrete.Probit.from_formula(f, df), "fit_kwargs": {}},
    "Multinomial Logit": {"func": lambda f, df: discrete.MNLogit.from_formula(f, df), "fit_kwargs": {}},
    # Count
    "Poisson":           {"func": lambda f, df: count.Poisson.from_formula(f, df), "fit_kwargs": {}},
    "Negative Binomial": {"func": lambda f, df: count.NegativeBinomial.from_formula(f, df), "fit_kwargs": {}},
    "Zero-Inflated Poisson":      {"func": lambda f, df: count.ZeroInflatedPoisson.from_formula(f, df), "fit_kwargs": {}},
    "Zero-Inflated NB":           {"func": lambda f, df: count.ZeroInflatedNegativeBinomialP.from_formula(f, df), "fit_kwargs": {}},
    # Ordered
    "Ordered Logit":  {"func": lambda f, df: OrderedModel.from_formula(f, df, distr="logit"), "fit_kwargs": {}},
    "Ordered Probit": {"func": lambda f, df: OrderedModel.from_formula(f, df, distr="probit"), "fit_kwargs": {}},
    # Tobit & Hurdle & Heckman placeholders
    "Tobit":    {"func": lambda f, df: (_ for _ in ()).throw(NotImplementedError("Tobit not yet supported")), "fit_kwargs": {}},
    "Hurdle":   {"func": lambda f, df: (_ for _ in ()).throw(NotImplementedError("Hurdle model not yet supported")), "fit_kwargs": {}},
    "Heckman":  {"func": lambda f, df: (_ for _ in ()).throw(NotImplementedError("Heckman selection not yet supported")), "fit_kwargs": {}}
}

# Page config
st.set_page_config(page_title="TBD", layout="wide")
st.title("TBD")

# Data upload
df_file = st.file_uploader("Upload CSV data", type=["csv"])
if not df_file:
    st.info("Please upload a CSV to continue.")
    st.stop()

@st.cache_data
def load_data(f):
    raw = f.read()
    enc = chardet.detect(raw)["encoding"]
    f.seek(0)
    return pd.read_csv(f, encoding=enc)

data = load_data(df_file)
st.subheader("Data Preview")
st.dataframe(data.head(5))

# Model configuration
n = st.number_input("Number of models", min_value=1, max_value=5, value=1)
configs = []
for i in range(n):
    with st.expander(f"Configure Model {i+1}", expanded=(i==0)):
        dv = st.selectbox("Dependent var", data.columns, key=f"dv{i}")
        ivs = st.multiselect("Independent vars", [c for c in data.columns if c!=dv], key=f"ivs{i}")
        # interactions
        if len(ivs)>=2:
            pairs = [(ivs[a], ivs[b]) for a in range(len(ivs)) for b in range(a+1, len(ivs))]
            opts = [f"{x}_x_{y}" for x,y in pairs]
            sel = st.multiselect("Interaction terms", opts, key=f"ints{i}")
            for term in sel:
                x,y = term.split("_x_")
                data[term] = data[x]*data[y]
                ivs.append(term)
        ests = st.multiselect(
            "Estimators:", list(ESTIMATOR_MAP.keys()) + ["Fixed Effects","Random Effects","Mixed Effects"],
            default=["OLS"], key=f"ests{i}"  
        )
        # panel ids
        ent = time = None
        if any(m in ests for m in ["Fixed Effects","Random Effects"]):
            ent = st.selectbox("Entity ID", data.columns, key=f"ent{i}")
            time= st.selectbox("Time ID", data.columns, key=f"time{i}")
        # mixed
        mg = ms = None
        if "Mixed Effects" in ests:
            mg = st.selectbox("Mixed group var", data.columns, key=f"mg{i}")
            ms = st.multiselect("Random slopes", ivs, key=f"ms{i}")
        # SEs
        se = st.selectbox("SE type", ["Standard","White (HC0)","Robust (HC1)","Clustered"], key=f"se{i}")
        cl = None
        if se=="Clustered": cl = st.selectbox("Cluster var", data.columns, key=f"cl{i}")
        configs.append(dict(dv=dv, ivs=ivs, ests=ests, ent=ent, time=time, mg=mg, ms=ms, se=se, cl=cl))

# Run models
if st.button("Run Models"):
    results = {}
    for idx,cfg in enumerate(configs,1):
        formula = f"{cfg['dv']} ~ {' + '.join(cfg['ivs'])}"
        # OLS/LPM SE map
        ols_map = {"Standard":{"cov_type":"nonrobust"},"White (HC0)":{"cov_type":"HC0"},"Robust (HC1)":{"cov_type":"HC1"}}
        if cfg['se']=="Clustered": ols_map['Clustered']={"cov_type":"cluster","cov_kwds":{"groups":data[cfg['cl']]}}
        # dispatch
        for est in cfg['ests']:
            key=f"Model{idx}_{est.replace(' ','')}_"
            try:
                if est in ['OLS','LPM']:
                    mod = ESTIMATOR_MAP[est]['func'](formula,data)
                    res = mod.fit(**ols_map[cfg['se']])
                elif est in ESTIMATOR_MAP:
                    mod = ESTIMATOR_MAP[est]['func'](formula,data)
                    res = mod.fit(**ESTIMATOR_MAP[est]['fit_kwargs'])
                elif est=='Fixed Effects':
                    panel=data.set_index([cfg['ent'],cfg['time']]);y=panel[cfg['dv']];X=panel[cfg['ivs']]
                    fe=PanelOLS(y,X,entity_effects=True)
                    cov={'unadjusted'}
                    res=fe.fit(cov_type=('clustered' if cfg['se']=='Clustered' else 'robust' if cfg['se']!='Standard' else 'unadjusted'),cluster_entity=(cfg['cl']==cfg['ent']))
                elif est=='Random Effects':
                    panel=data.set_index([cfg['ent'],cfg['time']]);y=panel[cfg['dv']];X=panel[cfg['ivs']]
                    re=RandomEffects(y,X)
                    res=re.fit(cov_type=('clustered' if cfg['se']=='Clustered' else 'robust' if cfg['se']!='Standard' else 'unadjusted'),cluster_entity=(cfg['cl']==cfg['ent']))
                elif est=='Mixed Effects':
                    re_formula = None
                    if cfg['ms']: re_formula="~"+"+".join(cfg['ms'])
                    mixed=MixedLM(formula,data,groups=data[cfg['mg']],re_formula=re_formula)
                    res=mixed.fit()
                else:
                    raise ValueError(f"Unknown estimator {est}")
                results[key+est]=res
            except Exception as e:
                st.error(f"Estimator {est} failed: {e}")
    # display
    statsmods=[r for r in results.values() if hasattr(r,'params')]
    for name,r in results.items():
        st.subheader(name)
        summ=r.summary() if callable(r.summary) else r.summary
        txt=summ.as_text() if hasattr(summ,'as_text') else str(summ)
        st.code(txt)
    if len(statsmods)>0:
        st.subheader("Combined Table")
        html=Stargazer(statsmods).render_html()
        st.components.v1.html(html,height=400)
        st.download_button("Download LaTeX",Stargazer(statsmods).render_latex(),"table.tex")
