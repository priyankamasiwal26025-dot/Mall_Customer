# app.py

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Train KMeans (fixed k=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Save clustered data
df.to_csv("Mall_Customers_Clustered.csv", index=False)


@app.route("/")
def home():
    return "<h2>Welcome to Mall Customers Clustering App</h2>" \
           "<p>Visit <a href='/clusters'>/clusters</a> to see clustering results.</p>"


@app.route("/clusters")
def clusters():
    # Show first 10 rows of clustered dataset
    return df.head(10).to_html(classes="table table-striped")


@app.route("/plot")
def plot_clusters():
    # Plot clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="rainbow", s=50
    )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments")
    plot_path = "static/cluster_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return f"<h3>Customer Segments</h3><img src='/{plot_path}' width='600'>"


if __name__ == "__main__":
    # Ensure static directory exists
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)
