# src/models/train.py

import pandas as pd
import glob
import argparse
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def load_data(parquet_folder):
    files = glob.glob(f"{parquet_folder}/*.parquet")
    df_list = [pd.read_parquet(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

def train_model(data, iteration_name):
    mlflow.set_tracking_uri("http://35.202.156.175:8100")
    mlflow.set_experiment("stock_movement_prediction")

    X = data[['rolling_avg_10', 'volume_sum_10']]
    y = data['target_5min_up']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=iteration_name):
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Save model
        mlflow.sklearn.log_model(clf, artifact_path="model")

        print(f"[{iteration_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed parquet folder")
    parser.add_argument("--name", required=True, help="Iteration name for MLflow")
    args = parser.parse_args()

    df = load_data(args.data)
    train_model(df, args.name)
