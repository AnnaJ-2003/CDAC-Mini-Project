# Dengue Prediction using a Decision Tree

Machine learning project based on the paper *"Dengue Prediction using Machine Learning Algorithms"* (IEEE R10-HTC 2020).

A Decision Tree is trained in a Jupyter Notebook and saved with joblib. The same saved model is then used by two web apps, one built with Flask and one with Django.

## What the website does
1. **Page 1, Analysis:** EDA charts, classification report, confusion matrix and the decision tree.
2. **Page 2, Prediction:** enter patient values and get a Positive or Negative result with probabilities.

## Project structure
```
dengue_project/
├── dengue_notebook.ipynb         # data analysis and model training
├── dengue_model.joblib           # saved decision tree
├── Dengue_clinical_dataset.csv   # dataset
├── flaskapp/                     # Flask website
├── django/                       # Django website
└── requirements.txt              # libraries needed
```

## How to run

**1. Set up**
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Flask app** (opens at http://127.0.0.1:5000)
```
cd flaskapp
python app.py
```

**3. Django app** (opens at http://127.0.0.1:8000)
```
cd django
python manage.py runserver
```

## Model details
- Algorithm: Decision Tree (Gini, max depth 4)
- Split: 70% training, 30% testing
- Target: Dengue Positive or Negative
- Test accuracy: about 99%, driven mostly by platelet count and WBC

## Dataset
Structured Clinical and Hematological Dataset for Early Dengue Diagnosis in Bangladesh (Mendeley Data, CC BY 4.0).

## Note
This is an educational project. It is not a medical diagnosis tool.