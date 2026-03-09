import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Load models once
regression_model = joblib.load(
    "model_generators/regression/regression_model.pkl")
classification_model = joblib.load(
    "model_generators/classification/classification_model.pkl")
clustering_model = joblib.load(
    "model_generators/clustering/clustering_model.pkl")


def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "evaluations": {
            "regression": evaluate_regression_model(),
            "classification": evaluate_classification_model(),
            "clustering": evaluate_clustering_model()
        }
    }
    
    # Handle form submissions for all models
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        
        # Determine which form was submitted based on button name or hidden input
        form_type = request.POST.get("form_type", "")
        
        if form_type == "regression" or "Predict Market Price" in request.POST.get("submit", ""):
            prediction = regression_model.predict([[year, km, seats, income]])[0]
            context["price"] = prediction
            
        elif form_type == "classification" or "Predict Income Category" in request.POST.get("submit", ""):
            prediction = classification_model.predict([[year, km, seats, income]])[0]
            context["prediction"] = prediction
            
        elif form_type == "clustering" or "Run Combined Inference" in request.POST.get("submit", ""):
            # Step 1: Predict price
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            # Step 2: Predict cluster
            cluster_id = clustering_model.predict([[income, predicted_price]])[0]
            mapping = {
                0: "Economy",
                1: "Standard",
                2: "Premium"
            }
            context.update({
                "prediction": mapping.get(cluster_id, "Unknown"),
                "price": predicted_price
            })
    
    return render(request, "predictor/index.html", context)
