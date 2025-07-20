from feast_feature_loader import load_features_with_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow

if __name__ == "__main__":
    feast_repo = "feast/stock_project"
    entity_path = "feast/stock_project/data/entity_df.parquet"

    df = load_features_with_labels(entity_path, feast_repo)

    # Drop rows with missing labels or features
    df = df.dropna()

    X = df[["rolling_avg_10", "volume_sum_10"]]
    y = df["target"]

    # Split manually
    train_size = int(0.8 * len(df))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    with mlflow.start_run():
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        print(f"Test Accuracy: {acc:.4f}")
