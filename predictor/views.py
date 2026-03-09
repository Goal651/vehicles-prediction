import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration, rwanda_vehicle_map
import joblib
from model_generators.clustering.train_cluster import evaluate_clustering_model, predict_cluster
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Load models once
regression_model = joblib.load("model_generators/regression/regression_model.pkl")
classification_model = joblib.load("model_generators/classification/classification_model.pkl")


def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")
    
    # Get model evaluations
    clustering_eval = evaluate_clustering_model()
    
    # CV metrics from clustering
    cv_metrics = {
        "income_cv": clustering_eval.get("cv", "N/A"),
        "price_cv": clustering_eval.get("cv", "N/A"),
        "average_cv": clustering_eval.get("cv", "N/A")
    }
    
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": rwanda_vehicle_map(df),
        "cv_metrics": cv_metrics,
        "evaluations": {
            "regression": evaluate_regression_model(),
            "classification": evaluate_classification_model(),
            "clustering": clustering_eval
        }
    }
    
    # Handle form submissions
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        form_type = request.POST.get("form_type", "")
        
        if form_type == "regression":
            context["price"] = regression_model.predict([[year, km, seats, income]])[0]
            
        elif form_type == "classification":
            context["prediction"] = classification_model.predict([[year, km, seats, income]])[0]
            
        elif form_type == "clustering":
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]
            context.update({
                "prediction": predict_cluster(income, predicted_price),
                "price": predicted_price
            })
    
    return render(request, "predictor/index.html", context)
