from pathlib import Path
import pandas as pd


def load_dataset(nb, split, data_dir="data", cluster_id=None, n=3):
    data_dir = Path(data_dir)

    # Build file names
    if cluster_id is None:
        X_path = data_dir / f"{nb}_X_{split}.parquet"
        y_path = data_dir / f"{nb}_y_{split}.parquet"
    else:
        X_path = data_dir / f"{nb}_X_{split}_cluster_{cluster_id}.parquet"
        y_path = data_dir / f"{nb}_y_{split}_cluster_{cluster_id}.parquet"

    # Load
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)

    # Print info
    label = f"{split}" if cluster_id is None else f"{split} | cluster {cluster_id}"

    print(f"\n{label}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nX head:")
    display(X.head(n))

    # print("\ny head:")
    # display(y.head(n))

    return X, y
