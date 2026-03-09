# Vehicle Analytics System - Django ML Project

## 📋 Overview
A comprehensive vehicle analytics system that integrates machine learning models with a Django web application. The system analyzes vehicle sales data to provide price predictions, income classification, and customer segmentation.

## 🎯 Features
- **Price Prediction**: Regression model to predict vehicle selling prices
- **Income Classification**: Classification model to predict customer income levels
- **Client Segmentation**: K-Means clustering for customer segmentation
- **Interactive Dashboard**: Web interface for real-time predictions
- **Data Exploration**: Built-in EDA capabilities

## 🛠️ Technology Stack
- **Backend**: Django 4.x
- **Machine Learning**: scikit-learn, pandas
- **Data Visualization**: matplotlib, seaborn, plotly
- **Frontend**: Bootstrap 5, HTML templates
- **Model Persistence**: joblib

## 📁 Project Structure
```
vehicles-prediction/
│
├── manage.py
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py              # Main URL configuration
│   └── wsgi.py
│
├── predictor/                # Main app
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   ├── views.py             # View functions for ML integration
│   ├── urls.py              # App URL routes
│   │
│   ├── templates/
│   │   └── predictor/
│   │       ├── index.html                    # Dashboard home
│   │       ├── regression_analysis.html       # Price prediction UI
│   │       ├── classification_analysis.html   # Income classification UI
│   │       └── clustering_analysis.html       # Customer segmentation UI
│
├── dummy-data/
│   └── vehicles_ml_dataset.csv    # Training dataset
│
├── model_generators/           # ML training scripts
│   ├── regression/
│   │   └── train_regression.py
│   │
│   ├── classification/
│   │   └── train_classifier.py
│   │
│   └── clustering/
│       └── train_cluster.py
│
├── regression_model.pkl          # Trained regression model
├── classification_model.pkl      # Trained classification model
├── clustering_model.pkl          # Trained clustering model
│
└── requirements.txt
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or newer
- pip (Python package manager)
- Virtual environment (recommended)
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Goal651/vehicles-prediction.git
cd vehicles-prediction
```

### Step 2: Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset
Place your dataset file (`vehicles_ml_dataset.csv`) in the `dummy-data/` folder.

Expected dataset columns:
- `year`: Vehicle manufacturing year
- `kilometers_driven`: Total kilometers driven
- `seating_capacity`: Number of seats
- `estimated_income`: Owner's estimated income
- `selling_price`: Vehicle selling price (for regression)
- `income_level`: Income category (for classification)
- `client_name`: Client identifier (for clustering)

### Step 5: Train ML Models
From the project root directory:
```bash
# Train regression model
python model_generators/regression/train_regression.py

# Train classification model
python model_generators/classification/train_classifier.py

# Train clustering model
python model_generators/clustering/train_cluster.py
```

### Step 6: Run Django Server
```bash
python manage.py runserver
```

### Step 7: Access the Application
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

## 📊 Model Details

### 1. Regression Model (Price Prediction)
- **Algorithm**: Random Forest Regressor
- **Features**: year, kilometers_driven, seating_capacity, estimated_income
- **Target**: selling_price
- **Evaluation**: R² Score

### 2. Classification Model (Income Level)
- **Algorithm**: Random Forest Classifier
- **Features**: year, kilometers_driven, seating_capacity, estimated_income
- **Target**: income_level
- **Evaluation**: Accuracy Score

### 3. Clustering Model (Customer Segmentation)
- **Algorithm**: K-Means Clustering
- **Features**: estimated_income, selling_price
- **Clusters**: 3 (Economy, Standard, Premium)
- **Evaluation**: Silhouette Score

## 🖥️ Usage Guide

### Navigation
1. **Data Exploration**: View dataset statistics and exploratory analysis
2. **Regression Analysis**: Predict vehicle selling prices
3. **Classification Analysis**: Predict income categories
4. **Clustering Analysis**: Customer segmentation with dual-model inference

### Making Predictions
1. Navigate to the desired analysis page
2. Fill in the vehicle specifications:
   - Model Year
   - Kilometers Driven
   - Number of Seats
   - Owner Income
3. Click submit to get predictions

## 📈 Exercise Tasks

### Task A: Rwanda Map Integration (20 marks)
Add a map visualization to the exploratory data analysis page showing:
- Rwanda district boundaries
- Number of vehicle clients in each district
- Use Plotly for interactive mapping

### Task B: Clustering Enhancement (10 marks)
The current Silhouette Score is 0.68:
1. Calculate the coefficient of variation and display it alongside the Silhouette Score (5 marks)
2. Refine the clustering model to achieve a Silhouette Score above 0.9 (5 marks)

## 🔧 Troubleshooting

### Common Issues and Solutions

1. **File not found errors**
   - Ensure you're running commands from the project root directory
   - Check that all file paths in the code match your actual structure

2. **Dataset not loading**
   - Verify the CSV file exists in `dummy-data/`
   - Check column names match those in the training scripts

3. **Model files not found**
   - Run training scripts first to generate .pkl files
   - Ensure .pkl files are in the project root

4. **Django import errors**
   - Activate virtual environment
   - Verify all packages are installed: `pip list`

## 📝 Requirements.txt
```
django>=4.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
plotly>=5.3.0
numpy>=1.21.0
```

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License
This project is for educational purposes as part of the Django/ML learning curriculum.

## 👥 Authors
- Created for students learning Python/Django and Machine Learning integration

## 🙏 Acknowledgments
- Django Documentation
- scikit-learn Documentation
- Bootstrap 5 Framework

---

**Happy Coding! 🚗📊🤖**