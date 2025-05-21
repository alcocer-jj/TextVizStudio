import streamlit as st
import streamlit_nested_layout
import pandas as pd
import chardet
import statsmodels.formula.api as smf
import statsmodels.discrete.discrete_model as discrete
import statsmodels.discrete.count_model as count
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.regression.mixed_linear_model import MixedLM
from linearmodels.panel import PanelOLS, RandomEffects
from stargazer.stargazer import Stargazer
import numpy as np


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
    "OLS": {
        "func": (lambda f, df, weights=None: smf.wls(f, df, weights=weights)
                 if weights is not None else smf.ols(f, df)),
        "panel": False, "mixed": False
    },
    "LPM": {
        "func": (lambda f, df, weights=None: smf.wls(f, df, weights=weights)
                 if weights is not None else smf.ols(f, df)),
        "panel": False, "mixed": False
    },
    "Logit": {
        "func": (lambda f, df, weights=None: discrete.Logit.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Probit": {
        "func": (lambda f, df, weights=None: discrete.Probit.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Multinomial Logit": {
        "func": (lambda f, df, weights=None: discrete.MNLogit.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Poisson": {
        "func": (lambda f, df, weights=None: count.Poisson.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Negative Binomial": {
        "func": (lambda f, df, weights=None: count.NegativeBinomialP.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Zero-Inflated Poisson": {
        "func": (lambda f, df, weights=None: count.ZeroInflatedPoisson.from_formula(f, df)),
        "panel": False, "mixed": False
    },
    "Zero-Inflated NB": {
        "func": (lambda f, df, weights=None: count.ZeroInflatedNegativeBinomialP.from_formula(
            formula=f,
            data=df,
            exog_infl=df[weights] if weights in df.columns else None,
            inflation="logit",
            p=1)),
        "panel": False, "mixed": False
    },
    "Ordered Logit": {
        "func": (lambda f, df, weights=None: OrderedModel.from_formula(f, df, distr="logit")),
        "panel": False, "mixed": False
    },
    "Ordered Probit": {
        "func": (lambda f, df, weights=None: OrderedModel.from_formula(f, df, distr="probit")),
        "panel": False, "mixed": False
    },
    "Fixed Effects":    {"func": None, "panel": True,  "mixed": False},
    "Random Effects":   {"func": None, "panel": True,  "mixed": False},
    "Mixed Effects":    {"func": None, "panel": False, "mixed": True}
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

# --- Models that support weights ---
WEIGHTABLE_MODELS = [key for key, val in ESTIMATOR_MAP.items() if val['func'] is not None]


# --- Streamlit UI ---
st.set_page_config(page_title="TBD", layout="wide")
st.title("TBD")

def detect_encoding(file):
    raw = file.read()
    result = chardet.detect(raw)
    encoding = result["encoding"]
    file.seek(0)
    return encoding

# Data upload
uploaded = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded:
    try:
        encoding = detect_encoding(uploaded)
        st.info(f"Detected file encoding: `{encoding}`")
        @st.cache_data
        def load_data(uploaded_file, file_encoding):
            return pd.read_csv(uploaded_file, encoding=file_encoding)
        data = load_data(uploaded, encoding)
        st.success("✔︎ File successfully loaded!")
        st.subheader("Data Preview"); st.dataframe(data.head(5))
    except Exception as e:
        st.error(f"Failed to read the CSV file: {e}")
if 'data' not in locals():
    st.stop()

# Model configuration
st.subheader("Model Configurations")
num_models = st.number_input(
    "Number of models to run",
    min_value=1,
    max_value=4,
    value=1,
    help="**Note:** To maintain a clean user interface and limit cloud memory, selection is set to a maximum of four concurrent models."
)
configs = []
for i in range(num_models):
    with st.expander(f"Configure Model {i+1}", expanded=(i==0)):
        dv = st.selectbox("Dependent variable", data.columns, key=f"dv_{i}")
        ivs = st.multiselect("Independent variables", [c for c in data.columns if c!=dv], key=f"ivs_{i}")
        # interactions
        if len(ivs)>=2:
            pairs=[(ivs[a],ivs[b]) for a in range(len(ivs)) for b in range(a+1,len(ivs))]
            opts=[f"{x}_x_{y}" for x,y in pairs]
            sel=st.multiselect("Interaction terms (optional)", opts, key=f"ints_{i}")
            for term in sel:
                x,y=term.split("_x_"); data[term]=data[x]*data[y]
                if term not in ivs: ivs.append(term)
        # estimator selection
        ests = st.multiselect("Estimators", list(ESTIMATOR_MAP.keys()), default=["OLS"], key=f"ests_{i}")
        with st.expander("About the Available Estimators", expanded=False):
            st.markdown("""
                        #### Predicting Continuous Outcomes
                        - **OLS (Ordinary Least Squares)**  
                            - **When to pick it:** You want to predict something like people’s height or test scores based on other factors, and most of your data points hover reasonably close to a line.  
                            - **Why it helps:** Very quick to run and easy to explain (“a one-unit change in X adds β units to Y”).  
                            - **Watch out:** If a few data points are way off the line (outliers) or the spread of errors varies wildly, your results can get skewed.  
                            - **How to read the result:** The coefficient β means “for each one-unit increase in X, the average of Y goes up by β.”

                        - **LPM (Linear Probability Model)**  
                            - **When to pick it:** You need a fast, first-cut estimate of a yes/no outcome (e.g., did someone vote or not?).  
                            - **Why it helps:** Coefficients translate directly into percentage-point changes (e.g., “X increases the chance of voting by 5 points”).  
                            - **Watch out:** Sometimes it predicts probabilities below 0% or above 100%.  
                            - **How to read the result:** A coefficient of 0.05 means a one-unit rise in X boosts the probability of “yes” by 5 percentage points.
                    
                        ---

                        #### Choosing Among Categories
                        - **Logit / Probit**  
                            - **When to pick it:** Your outcome is yes/no and you want predictions that always stay between 0% and 100%.  
                            - **Why it helps:** Keeps your predictions in bounds and handles uneven spread in the data.  
                            - **Watch out:** The raw coefficients aren’t in “percentage points”—you’ll need a quick post-estimate step to get marginal effects.  
                            - **How to read the result:** For Logit, β is a change in log-odds; exp(β) is the odds ratio. For Probit, β is a change in the underlying z-score; use average marginal effects to convert to probability changes.

                        - **Multinomial Logit**  
                            - **When to pick it:** You have three or more options without any natural order (e.g., favorite ice-cream flavor).  
                            - **Why it helps:** Models each choice against a baseline, letting you see how factors shift people toward option A vs. B vs. C.  
                            - **Watch out:** Assumes every new option affects all choices equally (IIA).  
                            - **How to read the result:** A coefficient β for choice j means a one-unit increase in X multiplies the odds of choosing j vs. the baseline by exp(β).

                        ---

                        #### Modeling Counts of Events
                        - **Poisson Regression**  
                            - **When to pick it:** You’re counting events (e.g., doctor visits per year) and your counts cluster around an average.  
                            - **Why it helps:** Fast, interpretable “rate” model (mean equals variance).  
                            - **Watch out:** If you see a lot more spread (variance) than the average, your standard errors will be too small.  
                            - **How to read the result:** β is the log change in the expected count; exp(β) is the multiplicative effect on the rate (e.g., exp(β)=1.2 means 20% more events).

                        - **Negative Binomial Regression**  
                            - **When to pick it:** Like Poisson, but your data show extra “lumpiness” (variance > mean).  
                            - **Why it helps:** Learns an extra dispersion parameter so the model can stretch to fit that extra spread.  
                            - **Watch out:** You need enough data to estimate that extra parameter well.  
                            - **How to read the result:** Same as Poisson: exp(β) gives the factor change in expected count per unit of X.

                        - **Zero-Inflated / Hurdle Models**  
                            - **When to pick it:** You see more zeros than a standard count model expects (e.g., lots of people never visit the ER).  
                            - **Why it helps:** Splits the process into “zero vs. non-zero” and “how many if non-zero,” giving you better fit.  
                            - **Watch out:** They’re more complex—make sure you have good predictors for why zeros occur.  
                            - **How to read the result:**  
                                - **Zero part:** exp(β₀) is the odds ratio of being an “excess zero.”  
                                - **Count part:** exp(β₁) is the rate ratio for non-zero counts.

                        ---

                        #### Ordering Things
                        - **Ordered Logit / Probit**  
                            - **When to pick it:** Your outcome has a clear order but no exact spacing (e.g., ratings like “low,” “medium,” “high”).  
                            - **Why it helps:** Respects the ranking without treating “high vs. medium” the same as “medium vs. low.”  
                            - **Watch out:** Assumes the effect of X is the same across all thresholds.  
                            - **How to read the result:** β shifts the log-odds (or z-score) of being at or above each threshold; use predicted probabilities to see category moves.

                        ---

                        #### Working with Panel Data
                        - **Fixed Effects**  
                            - **When to pick it:** You have multiple observations per entity (e.g., states over time) and want to ignore anything that doesn’t change over time.  
                            - **Why it helps:** Automatically removes unchanging traits (like a state’s geography).  
                            - **Watch out:** You can’t estimate effects of variables that never change.  
                            - **How to read the result:** Coefficients are the within-entity effect: a one-unit change in X for a given entity changes Y by β, holding that entity’s baseline constant.

                        - **Random Effects**  
                            - **When to pick it:** Same panel setup but you believe the unobserved traits aren’t correlated with your predictors.  
                            - **Why it helps:** Lets you include variables that don’t change over time.  
                            - **Watch out:** If that uncorrelated assumption fails, results can be biased—run a Hausman test.  
                            - **How to read the result:** Similar to OLS/GLM: β shows the average change in Y per unit X across entities, accounting for random intercept variation.

                        ---

                        #### Mixing It Up
                        - **Mixed Effects (Multilevel Models)**  
                            - **When to pick it:** Your data is nested (e.g., students within schools) and you want to model both overall trends and group-specific deviations.  
                            - **Why it helps:** Captures variation at each level (school vs. student).  
                            - **Watch out:** Model setup and convergence can be tricky—start simple before adding random slopes.  
                            - **How to read the result:**  
                                - **Fixed part:** β is the average effect of X across all groups.  
                                - **Random part:** The variance component tells you how much the intercepts (or slopes) vary from group to group.
                        """)   
        # Weights dropdown (conditional)
        weightable = any(e in WEIGHTABLE_MODELS for e in ests)
        if weightable:
            weight_options = [None] + list(data.columns)
            weight_var = st.selectbox("Weights variable (optional)",
            weight_options,
            index=0,
            format_func=lambda x: "Choose an option" if x is None else x,
            key=f"weights_{i}")
        else:
            weight_var = None    
        # Additional fixed effects
        disable_fe_ui = "Fixed Effects" in ests
        if disable_fe_ui:
            st.warning("⚠︎ Fixed effects (within) estimator handles entity/time FE internally. FE dummies disabled.")
        fe_vars = st.multiselect("Fixed effects (categorical variables) (optional)", options=[c for c in data.columns if c != dv and c not in ivs], key=f"fe_{i}", disabled=disable_fe_ui, help="**Note:** The `Fixed Effects` estimator is designed for panel data analysis, controlling for unobserved heterogeneity across entities. In contrast, this ‘fixed effects’ parameter refers to explicitly controlling for categorical variables (e.g., via dummy variables) in a regression model.")
        # ZINB-specific options
        zinb_infl_vars = []
        zinb_inflation = "logit"
        if "Zero-Inflated NB" in ests:
            st.markdown("**Zero-Inflation Settings**")
            zinb_infl_vars = st.multiselect("Zero-inflation variables", [c for c in data.columns if c != dv], key=f"zinb_vars_{i}")
            zinb_inflation = st.selectbox("Inflation link function", ["logit", "probit"], key=f"zinb_link_{i}")
            zinb_method = st.selectbox("Fitting method", ["bfgs", "lbfgs", "newton", "nm", "powell"], key=f"zinb_method_{i}")
            zinb_maxiter = st.number_input("Max iterations", min_value=500, max_value=1000000, value=500, step=10, key=f"zinb_maxiter_{i}")            
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
        if any(e in ["Zero-Inflated NB","Zero-Inflated Poisson"] for e in ests):
            se_opts=["Standard"]
        else:
            common_sets=[SUPPORTED_SE[e] for e in ests]
            se_opts=sorted(set.intersection(*common_sets)) if common_sets else []
        se_type=st.selectbox("Standard errors", se_opts, key=f"se_{i}", help="**Note:** Available standard error estimators are limited to those compatible with the selected estimation method.")
        with st.expander("Which Standard Error Estimator should I use?", expanded=False):
            st.markdown("""
                        **Standard**  
                        - Assumes every observation is independent and has the same variability.  
                        - Use when you believe your data errors are roughly “even” across all cases (e.g., no obvious outliers or groups).  

                        **White (HC0)**  
                        - Adjusts for situations where some points have more scatter than others (heteroskedasticity).  
                        - Good first fix if you see “funnel-shaped” residuals in a scatterplot.  

                        **Robust (HC1)**  
                        - Builds on HC0 but corrects for small sample bias.  
                        - Recommended when you have fewer observations (say, under a few hundred).  

                        **Clustered**  
                        - Accounts for “grouped” data (e.g., students within schools or repeated measures per person).  
                        - Treats all observations in the same cluster as potentially correlated—so your error bars widen appropriately.  
                        - Use when you suspect units within the same group share unmeasured shocks (like students all exposed to the same classroom).
                        """)
        cl=None
        if se_type=="Clustered": cl=st.selectbox("Cluster variable", data.columns, key=f"cl_{i}")        
        exp_output=False
        if any(e in ["Logit","Poisson","Negative Binomial","Zero-Inflated Poisson","Zero-Inflated NB"] for e in ests):
            exp_output=st.checkbox("Display exponentiated coefficients in output", key=f"exp_output_{i}")

        cfg={
            "dv":dv,
            "ivs":ivs,
            "ests":ests,
            "weights":weight_var,
            "ent":ent,
            "time":time,
            "mg":mg,
            "ms":ms,
            "se":se_type,
            "cl":cl,
            "fe_vars":fe_vars
        }
        if "Zero-Inflated NB" in ests:
            cfg.update({"zinb_vars":zinb_infl_vars,"zinb_link":zinb_inflation,"zinb_method":zinb_method,"zinb_maxiter":zinb_maxiter})
        if exp_output: cfg["exp_output"]=True
        configs.append(cfg)

# Run models
if st.button("Run Models"):
    results={}
    for idx,cfg in enumerate(configs,1):
        model_data=data.copy()
        # fixed effects dummies
        fe_dummies=[]
        if cfg.get("fe_vars") and "Fixed Effects" not in cfg["ests"]:
            for fe in cfg["fe_vars"]:
                dummies=pd.get_dummies(model_data[fe],prefix=fe,drop_first=True)
                model_data=model_data.join(dummies)
                fe_dummies.extend(dummies.columns)
            if len(fe_dummies)>100: st.warning("⚠️ Large number of dummy variables may cause memory or convergence issues.")
        rhs=cfg["ivs"]+fe_dummies
        form=f"{cfg['dv']} ~ {' + '.join(rhs)}"
        stats_cov=COMMON_STATS_SE.get(cfg["se"],{})
        panel_cov=PANEL_SE.get(cfg["se"],{})
        mixed_cov=MIXED_SE.get(cfg["se"],{})
        # weights Series
        weights=model_data[cfg['weights']] if cfg.get('weights') else None
        for est in cfg['ests']:
            key=f"Model{idx}_{est.replace(' ','')}"
            try:
                if ESTIMATOR_MAP[est]['func']:
                    covargs=stats_cov.copy()
                    if cfg['se']=='Clustered': covargs['cov_kwds']={'groups':model_data[cfg['cl']]}
                    # conditional weights
                    if est in WEIGHTABLE_MODELS and weights is not None:
                        mod=ESTIMATOR_MAP[est]['func'](form,model_data,weights=weights)
                    else:
                        mod=ESTIMATOR_MAP[est]['func'](form,model_data)
                    if est=="Zero-Inflated NB":
                        res=mod.fit(method=cfg.get("zinb_method","bfgs"),maxiter=cfg.get("zinb_maxiter",500),disp=1)
                    else:
                        res=mod.fit(**covargs)
                elif ESTIMATOR_MAP[est]['panel']:
                    panel_df=model_data.set_index([cfg['ent'],cfg['time']])
                    y=panel_df[cfg['dv']]
                    X=panel_df[cfg['ivs']]
                    mod=PanelOLS(y,X,entity_effects=True) if est=="Fixed Effects" else RandomEffects(y,X)
                    pargs=panel_cov.copy()
                    if cfg['se']=='Clustered':
                        pargs['cov_kwds']={'cluster_entity':True} if cfg['cl']==cfg['ent'] else {'cluster_time':True}
                    res=mod.fit(**pargs)
                elif ESTIMATOR_MAP[est]['mixed']:
                    rf="~ " + " + ".join(cfg['ms']) if cfg['ms'] else None
                    mod=MixedLM(form,model_data,groups=model_data[cfg['mg']],re_formula=rf)
                    res=mod.fit(**mixed_cov)
                else:
                    raise NotImplementedError(f"{est} not supported yet")
                results[key]=res
            except Exception as e:
                st.error(f"{est} failed: {e}")
    # display code unchanged
    exp_tables=[]
    statsmods=[r for r in results.values() if hasattr(r,'params')]
    for i,(name,res) in enumerate(results.items()):
        st.subheader(name)
        cfg_exp=configs[i].get('exp_output',False)
        model_type=str(type(res.model))
        supports_exp=any(k in model_type for k in ["Logit","Poisson","NegativeBinomial","ZeroInflated"])
        if cfg_exp and supports_exp:
            try:
                coef = res.params
                se = res.bse
                pvals = res.pvalues
                ci_lower = np.exp(coef - 1.96 * se)
                ci_upper = np.exp(coef + 1.96 * se)
                exp_coef = np.exp(coef)

                df = pd.DataFrame({
                    "exp(coef)": exp_coef.round(4),
                    "Std Err": se.round(4),
                    "95% CI Lower": ci_lower.round(4),
                    "95% CI Upper": ci_upper.round(4),
                    "P-value": pvals.round(4)
                })
                st.dataframe(df)
            except Exception as e:
                st.warning(f"Could not generate exponentiated output: {e}")
                summ = res.summary() if callable(res.summary) else res.summary
                txt = summ.as_text() if hasattr(summ, "as_text") else str(summ)
                st.code(txt)
        else:
            summ = res.summary() if callable(res.summary) else res.summary
            txt = summ.as_text() if hasattr(summ, "as_text") else str(summ)
            st.code(txt)

    # Stargazer for raw models only
    if any(not cfg.get("exp_output", False) for cfg in configs):
        raw_models = [r for i, r in enumerate(results.values()) if not configs[i].get("exp_output", False)]
        if len(raw_models) > 0:
            st.subheader("Results Table")
            html = Stargazer(raw_models).render_html()
            latex = Stargazer(raw_models).render_latex()
            st.download_button(
                label="Download LaTeX Table",
                data=latex,
                file_name="regression_table.tex")
            st.components.v1.html(html, height=3000, scrolling=False) 