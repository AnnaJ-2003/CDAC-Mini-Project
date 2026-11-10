import os, joblib, pandas as pd
import matplotlib
matplotlib.use("Agg")                          # draw plots without opening a window
import matplotlib.pyplot as plt
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)

BASE = str(settings.BASE_DIR)
ROOT = str(settings.BASE_DIR.parent)       # parent folder with the joblib and CSV

# ---------- 1. Load the model saved from the notebook ----------
bundle = joblib.load(os.path.join(ROOT, "dengue_model.joblib"))
model, COLUMNS, CLASSES = bundle["model"], bundle["columns"], bundle["class_names"]
LOCATIONS = [c.replace("Location_", "") for c in COLUMNS if c.startswith("Location_")]
YESNO = ["Fever", "Headache", "Muscle_Pain", "Rash", "Vomiting"]

# ---------- 2. Rebuild the same data/split as the notebook ----------
df = pd.read_csv(os.path.join(ROOT, "Dengue_clinical_dataset.csv"))
y = (df["Outcome"] == "Positive").astype(int)
X = pd.get_dummies(df.drop(columns=["Id", "Outcome"]), columns=["Gender", "Location"]).astype(float)
X = X[COLUMNS]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
y_pred = model.predict(X_test)

# ---------- 3. Save the Page 1 pictures ----------
PLOTS = os.path.join(BASE, "predictor", "static", "predictor")
os.makedirs(PLOTS, exist_ok=True)

def save(name):
    plt.savefig(os.path.join(PLOTS, name + ".png"), dpi=100, bbox_inches="tight")
    plt.close()

df["Outcome"].value_counts().plot.bar(color=["#e54", "#4c9"])
plt.title("Class distribution"); plt.ylabel("Patients"); plt.xticks(rotation=0); save("eda_class")

fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, c in zip(axs.ravel(), ["Age", "Platelet Count", "WBC", "Duration_of_Fever"]):
    for label in ["Negative", "Positive"]:
        ax.hist(df.loc[df["Outcome"] == label, c], bins=20, alpha=.55, label=label)
    ax.set_title(c); ax.legend()
plt.tight_layout(); save("eda_numeric")

df.groupby("Outcome")[YESNO].mean().T.plot.bar(figsize=(8, 4))
plt.title("Symptom rate by outcome"); plt.xticks(rotation=0); save("eda_symptoms")

df.assign(y=y).groupby("Location")["y"].mean().sort_values().plot.barh(figsize=(7, 4))
plt.title("Dengue-positive rate by location"); save("eda_location")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASSES).plot(cmap="Blues")
save("confusion")

plt.figure(figsize=(20, 9))
plot_tree(model, feature_names=COLUMNS, class_names=CLASSES, filled=True, rounded=True, fontsize=9)
save("tree")

# ---------- 4. Classification report as simple rows for the template ----------
report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
ROWS = [{"name": k, "precision": f"{report[k]['precision']:.2f}", "recall": f"{report[k]['recall']:.2f}",
         "f1": f"{report[k]['f1-score']:.2f}", "support": int(report[k]["support"])}
        for k in CLASSES + ["macro avg", "weighted avg"]]
METRICS = {"n": len(df),
           "train_acc": f"{accuracy_score(y_train, model.predict(X_train)):.3f}",
           "test_acc": f"{accuracy_score(y_test, y_pred):.3f}"}