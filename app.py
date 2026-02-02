from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# -----------------------------
# Load artifacts
# -----------------------------
scaler = joblib.load("scaler.pkl")
kmeans_model = joblib.load("kmeans.pkl")

clustering_features = [
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

# ⚠️ Update this based on your computed logic
SEGMENT_NAMES = {
    0: "Low Engagement Customers",
    1: "High Engagement Customers"
}

# -----------------------------
# Prediction logic
# -----------------------------
def predict_cluster(customer_dict):
    df = pd.DataFrame([customer_dict])
    X = df[clustering_features]
    X_scaled = scaler.transform(X)
    cluster_id = kmeans_model.predict(X_scaled)[0]
    return int(cluster_id)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    segment_name = None

    if request.method == "POST":
        customer = {
            'income': float(request.form['income']),
            'recency': float(request.form['recency']),
            'customer_tenure': float(request.form['customer_tenure']),
            'total_spending': float(request.form['total_spending']),
            'total_children': float(request.form['total_children']),
            'numwebpurchases': float(request.form['numwebpurchases']),
            'numcatalogpurchases': float(request.form['numcatalogpurchases']),
            'numstorepurchases': float(request.form['numstorepurchases']),
            'numwebvisitsmonth': float(request.form['numwebvisitsmonth']),
            'total_accepted_campaigns': float(request.form['total_accepted_campaigns'])
        }

        prediction = predict_cluster(customer)
        segment_name = SEGMENT_NAMES[prediction]

    return render_template(
        "index.html",
        prediction=prediction,
        segment_name=segment_name
    )

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
