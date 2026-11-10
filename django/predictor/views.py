import pandas as pd
from django.shortcuts import render
from . import ml

SYMPTOMS = [("Fever", "Fever"), ("Headache", "Headache"), ("Muscle_Pain", "Muscle pain"),
            ("Rash", "Rash"), ("Vomiting", "Vomiting")]


def analysis(request):                      # Page 1
    return render(request, "predictor/analysis.html",
                  {"m": ml.METRICS, "rows": ml.ROWS})


def predict(request):                       # Page 2
    result, error = None, None
    symptoms = [{"name": n, "label": l, "value": request.POST.get(n, "0")} for n, l in SYMPTOMS]

    if request.method == "POST":
        p = request.POST
        try:
            row = {c: 0.0 for c in ml.COLUMNS}               # start with all zeros
            row["Age"] = float(p["Age"])
            row["Platelet Count"] = float(p["Platelet"])
            row["WBC"] = float(p["WBC"])
            row["Duration_of_Fever"] = float(p["Duration"])
            for s in ml.YESNO:
                row[s] = float(p[s])
            row["Gender_" + p["Gender"]] = 1.0               # one-hot, same as in the notebook
            row["Location_" + p["Location"]] = 1.0
            probs = ml.model.predict_proba(pd.DataFrame([row])[ml.COLUMNS])[0]
            result = {"label": ml.CLASSES[int(probs.argmax())],
                      "probs": [(c, round(float(v) * 100, 1)) for c, v in zip(ml.CLASSES, probs)]}
        except (KeyError, ValueError):
            error = "Please fill in all fields with valid numbers."

    return render(request, "predictor/predict.html",
                  {"result": result, "error": error, "locations": ml.LOCATIONS,
                   "symptoms": symptoms, "f": request.POST})