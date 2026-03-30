from typing import Dict


def generate_insights(row: Dict, churn_probability: float) -> Dict:
    insights = []

    high_value = row.get("total_charges", 0) > 4000
    low_engagement = row.get("engagement_score", 0) < 6
    complaint_risk = row.get("complaint_proxy", 0) >= 2

    if churn_probability >= 0.7 and high_value:
        insights.append("Offer retention discount to protect high-value customer")
    if churn_probability >= 0.6 and low_engagement:
        insights.append("Send re-engagement email campaign")
    if churn_probability >= 0.6 and complaint_risk:
        insights.append("Route to priority support to address complaints")

    if not insights:
        insights.append("Continue standard engagement and monitor")

    return {"insights": insights}
