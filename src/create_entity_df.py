import pandas as pd
import glob
import os

def prepare_entity_df(input_folder, output_path):
    dfs = []
    for file in glob.glob(f"{input_folder}/*.parquet"):
        df = pd.read_parquet(file)
        df["stock"] = os.path.basename(file).split(".")[0]
        df = df[["timestamp", "stock", "target"]]
        dfs.append(df)

    full_df = pd.concat(dfs)
    full_df.to_parquet(output_path, index=False)
    print(f"Entity DataFrame saved at {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prepare_entity_df(args.input, args.output)

