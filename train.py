"""
train.py — Trains a simple RandomForest classifier on the Iris dataset,
logs params/metrics/seed/git_commit to MLflow, and saves the model artifact.

Usage:
    python train.py
"""

import subprocess
import hashlib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# ---- Reproducibility ----
SEED = 42

# ---- Get current git commit hash (for MLflow tag) ----
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
except Exception:
    git_commit = "unknown"

# ---- Get DVC-tracked dataset version (md5 hash from the .dvc file) ----
# This ties the MLflow run to the exact dataset version used for training,
# so reproducibility can be checked against both code (git) and data (dvc).
try:
    with open("file_list.csv.dvc") as f:
        dvc_file_contents = f.read()
    dvc_data_hash = hashlib.md5(dvc_file_contents.encode()).hexdigest()
except Exception:
    dvc_data_hash = "unknown"

# ---- Hyperparameters ----
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": SEED,
}

# ---- Load data ----
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ---- Train + log with MLflow ----
mlflow.set_experiment("aiops-capstone-reproducibility")

with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_param("seed", SEED)
    mlflow.set_tag("git_commit", git_commit)
    mlflow.set_tag("dvc_data_version", dvc_data_hash)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Git commit: {git_commit}")
    print(f"Accuracy: {acc:.4f}")