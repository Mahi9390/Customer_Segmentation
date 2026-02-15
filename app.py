# app.py  (Streamlit version)

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ────────────────────────────────────────────────
# Configuration & Model Loading
# ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.pkl")
    kmeans_model = joblib.load("kmeans.pkl")
    return scaler, kmeans_model

scaler, kmeans_model = load_artifacts()

FEATURES = [
    'income',
    'recency',
    'customer_tenure',
    'total_spending',
    'total_children',
    'numwebpurchases',
    'numcatalogpurchases',
    'numstorepurchases',
    'numwebvisitsmonth',
    'total_accepted_campaigns'
]

# Map cluster IDs to meaningful names (update according to your actual segments)
SEGMENT_NAMES = {
    0: "Low Engagement Customers",
    1: "High Engagement Customers",
    # Add more if your model has >2 clusters, example:
    # 2: "Premium Loyal Customers",
    # 3: "At-Risk Customers",
}

# ────────────────────────────────────────────────
# Prediction function
# ────────────────────────────────────────────────
def predict_segment(customer_input):
    # Create DataFrame from single customer input
    df = pd.DataFrame([customer_input])
    
    # Ensure all required columns exist
    X = df[FEATURES]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    cluster_id = kmeans_model.predict(X_scaled)[0]
    
    return int(cluster_id)

# ────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Predictor",
    page_icon="👥",
    layout="centered"
)

st.title("Customer Segmentation Predictor")
st.markdown("""
Predict which customer segment a person belongs to using K-Means clustering.  
Enter the required values below and click **Predict Segment**.
""")

# ── Input form ─────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Income", min_value=0.0, step=1000.0, format="%.0f")
    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    customer_tenure = st.number_input("Customer Tenure (months)", min_value=0, step=1)
    total_spending = st.number_input("Total Spending", min_value=0.0, step=100.0, format="%.2f")
    total_children = st.number_input("Total Children", min_value=0, max_value=10, step=1)

with col2:
    numwebpurchases = st.number_input("Web Purchases", min_value=0, step=1)
    numcatalogpurchases = st.number_input("Catalog Purchases", min_value=0, step=1)
    numstorepurchases = st.number_input("In-Store Purchases", min_value=0, step=1)
    numwebvisitsmonth = st.number_input("Web Visits per Month", min_value=0, step=1)
    total_accepted_campaigns = st.number_input("Accepted Campaigns (total)", min_value=0, max_value=6, step=1)

predict_button = st.button("Predict Segment", type="primary", use_container_width=True)

# ── Prediction & Result ────────────────────────────────────
if predict_button:
    # Collect input into dictionary
    customer = {
        'income': income,
        'recency': recency,
        'customer_tenure': customer_tenure,
        'total_spending': total_spending,
        'total_children': total_children,
        'numwebpurchases': numwebpurchases,
        'numcatalogpurchases': numcatalogpurchases,
        'numstorepurchases': numstorepurchases,
        'numwebvisitsmonth': numwebvisitsmonth,
        'total_accepted_campaigns': total_accepted_campaigns
    }

    with st.spinner("Predicting segment..."):
        try:
            cluster = predict_segment(customer)
            segment_name = SEGMENT_NAMES.get(cluster, f"Unknown Segment (Cluster {cluster})")

            st.success("Prediction complete!")
            
            st.subheader("Result")
            st.metric(
                label="Predicted Segment",
                value=segment_name,
                delta=f"Cluster {cluster}"
            )

            # Optional: show input summary
            with st.expander("Input values used"):
                st.json(customer)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check that all inputs are valid numbers and try again.")

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: K-Means clustering • Scaler & model loaded from joblib files")
