"""
KINETIC CHALLENGE

# Run:
streamlit run streamlit_app.py --server.runOnSave true
"""

# ──────────────────────────────
# Imports
# ──────────────────────────────

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from churn_analysis import engineer_features

# ──────────────────────────────
# Config
# ──────────────────────────────

DATA_DIR   = Path("./data")
MODEL_PATH = Path("./outputs/model.joblib")
METRICS_FP = Path("./outputs/metrics.json")

# ──────────────────────────────
# Streamlit
# ──────────────────────────────

@st.cache_data(show_spinner=False)
def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    users  = pd.read_csv(DATA_DIR / "users.csv", parse_dates=["signup_date"])
    churn  = pd.read_csv(DATA_DIR / "churn_labels.csv")
    usage  = pd.read_csv(DATA_DIR / "usage_logs.csv", parse_dates=["date"])
    return users.merge(churn, on="user_id"), usage

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if METRICS_FP.exists():
        return json.loads(METRICS_FP.read_text(encoding="utf8"))
    return {}

# ──────────────────────────────
# Main
# ──────────────────────────────

def main():
    st.title("SaaS Churn Risk Explorer")

    users, usage = load_inputs()
    model        = load_model()
    metrics      = load_metrics()

    st.sidebar.header("Filters")

    country = st.sidebar.multiselect("Country",  sorted(users["country"].unique()))
    plan    = st.sidebar.multiselect("Plan",  sorted(users["plan_type"].unique()))

    min_dt = users["signup_date"].min().to_pydatetime()
    max_dt = users["signup_date"].max().to_pydatetime()
    date_range = st.sidebar.slider(
        "SignUp entre...",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
        format="YYYY-MM-DD",
    )

    mask  = users["signup_date"].between(pd.Timestamp(date_range[0]),
                                         pd.Timestamp(date_range[1]))
    if country:
        mask &= users["country"].isin(country)
    if plan:
        mask &= users["plan_type"].isin(plan)

    if mask.sum() == 0:
        st.warning("There are no users matching the selected filters...")
        return

    subset_users = users[mask]

    # ---------- Feature engineering & prediction ----------
    feat_all      = engineer_features(users, usage)
    feat_subset   = (
        feat_all.set_index("user_id")
        .loc[subset_users["user_id"]]
        .reset_index()
    )

    X = feat_subset[model.named_steps["preprocessor"].feature_names_in_]
    churn_prob = model.predict_proba(X)[:, 1]

    subset_users = subset_users.assign(churn_risk=churn_prob)

    # -------------- Table --------------
    st.write(f"### Filtered Users: {len(subset_users)}")
    st.dataframe(
        subset_users[
            ["user_id", "plan_type", "country", "signup_date", "churn_risk"]
        ].sort_values("churn_risk", ascending=False),
        height=400,
    )

    # ------------ Histogram -------------
    st.write("### Churn Risk Distribution:")
    fig = px.histogram(subset_users, x="churn_risk", nbins=20,
                       title="Probability of Churn:")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Global Metrics ----------
    if metrics:
        st.sidebar.markdown("### Global Metrics (test set):")
        st.sidebar.json(metrics, expanded=False)

if __name__ == "__main__":
    main()
