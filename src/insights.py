# insights.py
# Rule-based business action layer — converts churn probability into actionable recommendations.
#
# Design choice: intentionally rule-based (not ML/SHAP) because:
#   - Deterministic output → easy to audit and explain to non-technical stakeholders
#   - Zero latency overhead → no extra model call or library required
#   - Business-aligned thresholds → rules can be tuned by the business team
#
# Called by api.py after model.predict_proba() returns a churn probability.
# Reads engineered features (computed in features.py) from the same row dict.

from typing import Dict


def generate_insights(row: Dict, churn_probability: float) -> Dict:
    insights = []

    # Segment flags — derived from engineered features in features.py
    high_value = row.get("total_charges", 0) > 4000        # Long-tenure, high-spend customer
    low_engagement = row.get("engagement_score", 0) < 6    # Few services, short tenure
    complaint_risk = row.get("complaint_proxy", 0) >= 2    # Two or more dissatisfaction signals

    # Rule 1: High-value customer about to leave → protect revenue with a direct offer
    # Threshold 0.7 (not 0.6) because discount is costly — only for near-certain churners
    if churn_probability >= 0.7 and high_value:
        insights.append("Offer retention discount to protect high-value customer")

    # Rule 2: Low-engagement customer at risk → pull them back into the product
    if churn_probability >= 0.6 and low_engagement:
        insights.append("Send re-engagement email campaign")

    # Rule 3: Dissatisfied customer at risk → fix the service problem before offering anything
    if churn_probability >= 0.6 and complaint_risk:
        insights.append("Route to priority support to address complaints")

    # Default: low churn signal — no action needed beyond routine monitoring
    if not insights:
        insights.append("Continue standard engagement and monitor")

    return {"insights": insights}
