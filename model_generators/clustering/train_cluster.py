import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import KBinsDiscretizer
import joblib

# Paths
MODEL_PATH = "model_generators/clustering/clustering_model.pkl"
SCALER_PATH = "model_generators/clustering/clustering_scaler.pkl"
DATA_PATH = "dummy-data/vehicles_ml_dataset.csv"
FEATURES = ["estimated_income", "selling_price"]
INCOME_WEIGHT = 1.5


def build_features(df, scaler=None, fit=False):
    """Build discretized features for clustering."""
    income = df["estimated_income"].values.reshape(-1, 1)
    price = df["selling_price"].values.reshape(-1, 1)
    
    if fit:
        kbd_income = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        kbd_price = KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
        income_bins = kbd_income.fit_transform(income)
        price_bins = kbd_price.fit_transform(price)
        scaler = {"income": kbd_income, "price": kbd_price}
    else:
        income_bins = scaler["income"].transform(income)
        price_bins = scaler["price"].transform(price)
    
    # Weight income more heavily for better segmentation
    X = np.column_stack([income_bins * INCOME_WEIGHT, price_bins])
    return X, scaler


def get_cluster_labels(kmeans):
    """Get cluster labels sorted by income."""
    centers = kmeans.cluster_centers_
    sorted_indices = centers[:, 0].argsort()
    return {
        int(sorted_indices[0]): "Economy",
        int(sorted_indices[1]): "Standard", 
        int(sorted_indices[2]): "Premium"
    }


def predict_cluster(income, price):
    """Predict cluster for given income and price."""
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(MODEL_PATH)
    
    df_input = pd.DataFrame({"estimated_income": [income], "selling_price": [price]})
    X, _ = build_features(df_input, scaler=scaler, fit=False)
    
    cluster_id = int(kmeans.predict(X)[0])
    labels = get_cluster_labels(kmeans)
    return labels.get(cluster_id, "Unknown")


def evaluate_clustering_model():
    """Evaluate clustering model and return metrics."""
    df = pd.read_csv(DATA_PATH)
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(MODEL_PATH)
    X, _ = build_features(df, scaler=scaler, fit=False)

    # Predict clusters
    df["cluster_id"] = kmeans.predict(X)
    labels = get_cluster_labels(kmeans)
    df["client_class"] = df["cluster_id"].map(labels)

    # Calculate silhouette score
    silhouette = round(silhouette_score(X, df["cluster_id"]), 4)

    # Calculate CV for each cluster
    cv_data = []
    for cluster_name in ["Economy", "Standard", "Premium"]:
        cluster_data = df[df["client_class"] == cluster_name]
        for feature in FEATURES:
            mean_val = cluster_data[feature].mean()
            std_val = cluster_data[feature].std()
            cv = round((std_val / mean_val) * 100, 2) if mean_val != 0 else 0
            cv_data.append({
                "Cluster": cluster_name,
                "Feature": feature,
                "Mean": round(mean_val, 2),
                "Std": round(std_val, 2),
                "CV (%)": cv
            })
    
    cv_df = pd.DataFrame(cv_data)
    overall_cv = round(cv_df["CV (%)"].mean(), 2)
    
    # Create HTML tables
    cv_table = cv_df.to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        index=False
    )
    
    # Cluster summary
    summary = df.groupby("client_class")[FEATURES].mean().round(2)
    counts = df["client_class"].value_counts().reset_index()
    counts.columns = ["client_class", "count"]
    summary = summary.merge(counts, on="client_class")
    
    summary_table = summary.to_html(
        classes="table table-bordered table-striped table-sm",
        float_format="%.2f",
        index=False
    )

    return {
        "silhouette": silhouette,
        "cv": overall_cv,
        "cv_table": cv_table,
        "summary": summary_table
    }


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X, scaler = build_features(df, fit=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=100)
    kmeans.fit(X)
    
    # Save model and scaler
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Calculate and print score
    score = silhouette_score(X, kmeans.predict(X))
    print(f"Clustering model trained. Silhouette Score: {score:.4f}")
