from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.data_format import ParquetFormat

stock_data = FileSource(
    path="data/feast_data.parquet",  # we’ll create this by merging
    timestamp_field="timestamp",

)

stock = Entity(name="stock", join_keys=["stock"])

stock_feature_view = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Int64),
    ],
    online=True,
    source=stock_data
)
