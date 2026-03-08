"""
MLflow demo - Applied AI teamwork
To show AI governance in practice with traceability + evidence.

Run a few simple experiments and log:
- params -> what we changed
- metrics -> accuracy, f1_macro
- artifacts -> confusion matrix + risk note + run summary
- tags -> lightweight governance metadata
"""

from __future__ import annotations

import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Small config to personalize
EXPERIMENT_NAME = "AI_Governance_MLflow_Demo"
TEAM_TAG = "Group O"  # group tag
OWNER_TAG = "Nuutti & Saska"  # name for accountability
ARTIFACT_DIR = "artifacts"


def get_data():
    """Built-in dataset to keep the demo simple (no external files)."""
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_confusion_matrix_png(y_true, y_pred, run_name: str) -> str:
    """Create a confusion matrix image for evidence."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix: {run_name}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out_path = os.path.join(ARTIFACT_DIR, f"confusion_{run_name}.png".replace(" ", "_"))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def run_and_log(run_name: str, model, params: dict, risk_note: str) -> None:
    """One run = one experiment. Train, evaluate, and log everything to MLflow."""
    X_train, X_test, y_train, y_test = get_data()

    with mlflow.start_run(run_name=run_name):
        # governance metadata
        mlflow.set_tag("team", TEAM_TAG)
        mlflow.set_tag("owner", OWNER_TAG)
        mlflow.set_tag("dataset", "sklearn: iris")
        mlflow.set_tag("purpose", "traceability_demo")
        mlflow.set_tag("review_status", "staging")
        mlflow.set_tag("approved", "false")
        mlflow.set_tag("timestamp", datetime.now().isoformat(timespec="seconds"))

        # params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # train + predict
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)

        # artifacts alias evidence
        cm_path = make_confusion_matrix_png(y_test, preds, run_name)
        mlflow.log_artifact(cm_path)

        risk_path = os.path.join(ARTIFACT_DIR, f"risk_note_{run_name}.txt")
        save_text(risk_path, risk_note)
        mlflow.log_artifact(risk_path)

        summary_path = os.path.join(ARTIFACT_DIR, f"run_summary_{run_name}.txt")
        save_text(
            summary_path,
            f"Run: {run_name}\n"
            f"Params: {params}\n"
            f"Metrics: accuracy={acc:.4f}, f1_macro={f1:.4f}\n"
            "Note: This is a small demo to show tracking and evidence.\n"
        )
        mlflow.log_artifact(summary_path)

        # For logging the model
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{run_name}: accuracy={acc:.3f}, f1_macro={f1:.3f}")


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Run 1: logistics regression
    run_and_log(
        run_name="logreg_baseline",
        model=LogisticRegression(C=1.0, max_iter=200),
        params={"model": "LogisticRegression", "C": 1.0, "max_iter": 200},
        risk_note=(
            "Risk note:\n"
            "- Toy dataset aka Iris, results don't generalize to real business cases.\n"
            "- Good baseline for transparency, but may miss more complex patterns.\n"
        ),
    )

    # Run 2: random forest v1
    run_and_log(
        run_name="random_forest_v1",
        model=RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        params={"model": "RandomForest", "n_estimators": 200, "max_depth": 5},
        risk_note=(
            "Risk note:\n"
            "- Often performs well, but is harder to explain than simpler models.\n"
            "- Accuracy/F1 don’t tell us anything about fairness or bias.\n"
        ),
    )

    # Run 3: random forest v2 + small parameter change
    run_and_log(
        run_name="random_forest_v2",
        model=RandomForestClassifier(n_estimators=200, max_depth=2, random_state=42),
        params={"model": "RandomForest", "n_estimators": 200, "max_depth": 5},
        risk_note=(
            "Risk note:\n"
            "- This run changes only one parameter (n_estimators), so this is mainly a comparison run.\n"
            "- Governance value -> we can see exactly what changed and what the impact was.\n"
        ),
    )


if __name__ == "__main__":
    main()
