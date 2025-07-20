# src/preprocess.py

import pandas as pd
import numpy as np
import os
from pathlib import Path

def preprocess_stock_csv(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    df = df.sort_values('timestamp').reset_index(drop=True)

    # Ensure frequency is 1 minute
    df.set_index('timestamp', inplace=True)
    df = df.asfreq('1min')

    # Fill missing rows using previous values (ffill)
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')

    # Rolling features
    df['rolling_avg_10'] = df['close'].rolling(window=10, min_periods=10).mean()
    df['volume_sum_10'] = df['volume'].rolling(window=10, min_periods=10).sum()

    # Target: whether close price after 5 minutes is higher than now
    df['future_close'] = df['close'].shift(-5)
    df['target_5min_up'] = (df['future_close'] > df['close']).astype(int)

    df = df.drop(columns=['future_close'])

    # Drop initial rows with NaNs from rolling
    df = df.dropna()

    # Reset index
    df = df.reset_index()

    # Save to Parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Processed: {input_path} → {output_path}")

def run(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    for file in input_folder.glob("*.csv"):
        stock_name = file.stem
        out_file = output_folder / f"{stock_name}.parquet"
        preprocess_stock_csv(file, out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    run(args.input, args.output)
