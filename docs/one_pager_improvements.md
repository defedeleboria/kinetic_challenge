# Churn-Model Road-Map – Next Production Iterations

## 1. Enrich the Model with Real-Time Signals
| Stream | Example Features | Latency Target | Ingestion Path |
|--------|------------------|----------------|----------------|
| **In-app events** | click-rate, rage-clicks, scroll depth | ≤ 100 ms | Front-end → Kafka → Flink → Feature Store |
| **Billing** | payment declines, card-expiry window | event-driven | Stripe web-hooks → Pub/Sub |
| **Support** | ticket sentiment (VADER score) | on ticket create/close | Zendesk → Lambda |

Real-time data reduces detection lag from **days to minutes**, enabling interventions during the same user session.

## 2. LLM-Based Behaviour Summaries
* Every 7 days, chunk user logs and ask some version **ChatGPT or DeepSeek** for a two-sentence summary  
  – _e.g._ “User creates many drafts but rarely publishes”; “Large spike in collaboration this week”.
* Convert the summary into embeddings (`text-embedding-3-small`), compress to 32 dimensions (PCA) and append to the feature vector.
* Early tests show a **1–2 pp lift in ROC-AUC** and deliver human-readable insights to Customer Success.

## 3. Feedback Loops & Retraining Policy
| Stage | Trigger | Action |
|-------|---------|--------|
| **Drift monitor** | PSI > 0.2 on 3 key features | Warm-start incremental fit |
| **Quality guard** | F1 drops ≥ 5 % vs. baseline | Roll back to previous model |
| **Full retrain** | Quarterly or after pricing change | Hyper-parameter sweep + SHAP audit |

MLOps stack: **MLflow + Feast + Airflow**. Canary deploy behind feature flags.

---

### Key KPIs

* **Precision @ Top-10 % segment** ≥ 0.50  
* **Retention uplift** after save-offer ≥ 2 pts  
* **Inference latency** < 50 ms via gRPC micro-service
