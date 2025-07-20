from feast import FeatureStore
import pandas as pd

def load_features_with_labels(entity_df_path: str, feast_repo_path: str):
    store = FeatureStore(repo_path=feast_repo_path)

    # Load entity dataframe (timestamp + stock symbol)
    entity_df = pd.read_parquet(entity_df_path)

    # Ensure 'stock' is str and timestamp is datetime
    entity_df["stock"] = entity_df["stock"].astype(str)
    entity_df["timestamp"] = pd.to_datetime(entity_df["timestamp"])

    # Get feature data from Feast
    feature_vector = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "stock_features:rolling_avg_10",
            "stock_features:volume_sum_10"
        ],
    ).to_df()

    return feature_vector
