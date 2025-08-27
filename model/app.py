# app.py
from flask import Flask, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import io

app = Flask(__name__)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Load trained model
kmeans = joblib.load("model/kmeans_mall_model.pkl")

@app.route("/")
def home():
    return "<h2>Mall Customers Clustering App</h2><p>Go to <a href='/plot'>/plot</a> to see clustering results.</p>"

@app.route("/plot")
def plot_clusters():
    # Select features
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # Use pre-trained model
    df["Cluster"] = kmeans.predict(X)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="rainbow", s=50)
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments")

    # Save plot to memory
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return send_file(img, mimetype="image/png")

@app.route("/data")
def data():
    return df.to_html(classes="table table-striped", index=False)

if __name__ == "__main__":
    app.run(debug=True)