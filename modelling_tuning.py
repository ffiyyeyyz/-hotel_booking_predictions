import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

dagshub.init(
    repo_owner="ffiyyeyyz",
    repo_name="hotel_booking_predictions",
    mlflow=True
)

experiment_name = "Hotel_Booking_RF_Tuning"

try:
    mlflow.set_experiment(experiment_name)
except Exception:
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

df = pd.read_csv("hotel_bookings_preprocessing.csv")

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 30, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

with mlflow.start_run(run_name="RandomForest_Tuning"):

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 🔹 log params (hasil tuning)
    for param, value in grid.best_params_.items():
        mlflow.log_param(param, value)

    # 🔹 log metrics (SETARA AUTOLOG)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # 🔹 log model
    mlflow.sklearn.log_model(best_model, "model")

    # 🔹 confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # 🔹 classification report
    report = classification_report(y_test, y_pred)

    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)

    # 🔹 feature importance
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    feat_path = "artifacts/feature_importance.csv"
    feat_imp.to_csv(feat_path)

    mlflow.log_artifact(feat_path)


