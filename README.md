# KINETIC CHALLENGE
# Churn Prediction Challenge – Deliverables

## Project Structure
```
.
├── churn_analysis.py              # End‑to‑end modelling script
├── streamlit_app.py               # Bonus dashboard
├── requirements.txt               # Python deps
├── outputs/                       # Artefacts and reports (created at runtime)
│   ├── eda/...
│   ├── metric.json
│   └── model.jonlib
│   └── feature_importance.png
│   └── roc_curve.png
│   └── shap_summary.png
└── docs/
    └── one_pager_improvements.md
```

## Approach & Assumptions
1. **Temporal split** – the newest 20 % of sign-ups are held out for testing to mimic future data drift.  
2. **Feature blocks**  
   * Lifetime aggregates (actions, minutes, documents, logins)  
   * Recent behaviour: 7-day rolling mean of logins (`avg_roll_logins_7d`)  
   * Recency: `days_since_signup`  
   * Demographics: `country`, `plan_type` (one-hot)  
3. **Class imbalance** – handled with `class_weight="balanced"` (RF / LogReg) or `scale_pos_weight` (XGBoost).  
4. **Model zoo** – Logistic Regression (baseline), Random-Forest (default), XGBoost (optional).  
5. **Explainability** – tree-based feature importances; optional SHAP summary when the library is available.

>
> **Assumptions**  
> * Behaviour in the first 90 days is most predictive; older users are assumed stabilised.  
> * Churn is a binary label (“canceled within the last 30 days”); we ignore downgrade-to-free events.  
> * Usage logs are complete and time-stamps are in UTC.
>

| Metric    | Value |
| --------- | ----: |
| Precision |  0.46 |
| Recall    |  0.60 |
| F1-score  |  0.52 |
| ROC-AUC   |  0.83 |

## How to Run

How to clone and run the project locally in VS Code?
Clone the repository!

```bash
# 1) Replace <user> with your GitHub username
git clone https://github.com/<user>/kinetic_challenge.git
cd kinetic_challenge

# 2) Open the folder in VS Code
code .

# 3) Python ≥3.10 + install deps (VS Code Terminal)
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt

# 4) Train model & generate artefacts
python churn_analysis.py --data-dir ./data --plots --shap --model rf

# 5) Launch dashboard
streamlit run streamlit_app.py --server.runOnSave true
```

## Next Production Iterations
See docs/one_pager_improvements.md for a concise plan covering:

- real-time behavioural signals,
- LLM-based user-behaviour summaries,
- drift monitoring and retraining cadence.

## Re-running / Customising

| Goal                        | Flag(s)                  |
| --------------------------- | ------------------------ |
| **Just metrics (no plots)** | omit `--plots`           |
| **Logistic regression**     | `--model logreg`         |
| **Train on synthetic data** | `--self-test`            |
| **Turn off SHAP**           | omit `--shap` (faster)   |
| **Change output directory** | `--outputs ./my_results` |


## Visuals
All EDA plots, ROC curves, and SHAP summaries are written to `outputs/`.

## Submission
Include:
* `churn_analysis.py`, `streamlit_app.py`, `requirements.txt`
* `outputs/model.joblib` (pre‑trained model)
* `docs/one_pager_improvements.md`
* Key PNGs inside `outputs/`

---
