# AI Governance & Responsible AI – MLflow Demo

This repository contains a small MLflow demo created for an Applied AI teamwork assignment.
The goal is to demonstrate AI governance in practice: traceability, documentation, and lightweight accountability.

## What the demo does
- Runs three simple ML experiments: Logistic Regression baseline + two Random Forest runs
- Logs parameters and metrics to MLflow: accuracy & macro F1
- Adds lightweight governance tags: e.g. owner, dataset, review status
- Stores artifacts as evidence:
  - confusion matrix image
  - risk note as txt
  - run summary as txt

## Requirements
- Python 3.9+
- Packages: mlflow, scikit-learn, pandas, matplotlib

## Setup and run (Windows / PowerShell)
1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies:
   python -m pip install --upgrade pip
   python -m pip install mlflow scikit-learn pandas matplotlib

3. Run the demo script:
   python .\teamwork_mlflow_demo.py

4. Start MLflow UI:
   python -m mlflow ui

5. Open in browser:
   http://127.0.0.1:5000
