import pandas as pd
import glob
import os

def merge_for_feast(input_folder, output_path):
    all_files = glob.glob(f"{input_folder}/*.parquet")
    dfs = []
    for file in all_files:
        df = pd.read_parquet(file)
        stock = os.path.basename(file).split(".")[0]
        df["stock"] = stock
        dfs.append(df)
    full_df = pd.concat(dfs)
    full_df.to_parquet(output_path, index=False)
    print(f"Feast-ready data written to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    merge_for_feast(args.input, args.output)
