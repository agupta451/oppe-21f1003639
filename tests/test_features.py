import pandas as pd
import pytest

# Fixture to generate a dummy DataFrame similar to your real one
@pytest.fixture
def dummy_data():
    data = {
        "timestamp": pd.date_range("2024-01-01 09:15", periods=20, freq="min"),
        "close": [100 + i for i in range(20)],
        "volume": [1000 + 10*i for i in range(20)]
    }
    df = pd.DataFrame(data)
    return df

# rolling_avg_10: should match 10-min rolling mean of close price
def test_rolling_avg_10(dummy_data):
    df = dummy_data.copy()
    df["rolling_avg_10"] = df["close"].rolling(window=10, min_periods=10).mean()
    
    # Check values after index 9
    assert not df["rolling_avg_10"].iloc[9:].isnull().any(), "rolling_avg_10 has NaNs"
    assert abs(df["rolling_avg_10"].iloc[9] - sum(range(100, 110))/10) < 1e-5

# volume_sum_10: should match 10-min volume sum
def test_volume_sum_10(dummy_data):
    df = dummy_data.copy()
    df["volume_sum_10"] = df["volume"].rolling(window=10, min_periods=10).sum()

    assert not df["volume_sum_10"].iloc[9:].isnull().any(), "volume_sum_10 has NaNs"
    expected_sum = sum([1000 + 10*i for i in range(10)])
    assert abs(df["volume_sum_10"].iloc[9] - expected_sum) < 1e-5
