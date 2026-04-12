from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc,
)

sns.set(style="whitegrid")
sns.set_palette("deep")


def load_dataset(nb, split, data_dir="data", cluster_id=None, n=3):
    data_dir = Path(data_dir)

    # Build file names
    if cluster_id is None:
        X_path = data_dir / f"{nb}_X_{split}.parquet"
        y_path = data_dir / f"{nb}_y_{split}.parquet"
    else:
        X_path = data_dir / f"{nb}_X_{split}_c{cluster_id}.parquet"
        y_path = data_dir / f"{nb}_y_{split}_c{cluster_id}.parquet"

    # Load
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path).squeeze()

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


def evaluate_model(model, X_val, y_val, threshold=0.5):
    """
    Evaluate a binary classifier using probability outputs.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "AUC_PR": average_precision_score(y_val, y_prob),
        "F1": f1_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
    }
    return metrics


def evaluate_from_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "AUC_PR": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def plot_cluster_pr_curves(
    global_probs_cal,
    y_val,
    X_val,
    cluster_models,
    cluster_calibrators,
    cluster_metadata,
    X_val_full,
    cluster_col="cluster",
):
    """
    Plots Precision-Recall curves for each cluster with global and cluster-specific models.
    Each cluster gets its own subplot with PR AUC annotated.
    Uses a 2x2 grid for up to 4 clusters per figure.
    """
    cluster_ids = sorted(X_val_full[cluster_col].unique())
    n_clusters = len(cluster_ids)
    n_cols = 2
    n_rows = (n_clusters + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, cluster_id in enumerate(cluster_ids):
        ax = axes[i]
        idx = X_val_full[cluster_col] == cluster_id

        # Global model
        y_true = y_val.loc[idx]
        y_prob_global = global_probs_cal[idx]
        precision_g, recall_g, _ = precision_recall_curve(y_true, y_prob_global)
        auc_g = average_precision_score(y_true, y_prob_global)  # updated here
        sns.lineplot(
            x=recall_g, y=precision_g, ax=ax, label=f"Global (AUC={auc_g:.3f})"
        )
        ax.fill_between(recall_g, precision_g, alpha=0.1)

        # Cluster-specific model
        model = cluster_models[cluster_id]
        calibrator = cluster_calibrators[cluster_id]
        metadata = cluster_metadata[cluster_id]
        X_val_cluster = X_val.loc[idx, metadata["features"]]
        y_prob_cluster = calibrator.transform(model.predict_proba(X_val_cluster)[:, 1])
        precision_c, recall_c, _ = precision_recall_curve(y_true, y_prob_cluster)
        auc_c = average_precision_score(y_true, y_prob_cluster)  # updated here
        sns.lineplot(
            x=recall_c,
            y=precision_c,
            ax=ax,
            label=f"Cluster-specific (AUC={auc_c:.3f})",
        )
        ax.fill_between(recall_c, precision_c, alpha=0.1)

        ax.set_title(f"Cluster {cluster_id}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes if clusters < n_rows*n_cols
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def evaluate_with_profit(y_true, y_prob, threshold, C_call, B_sub):
    # decision rule
    y_pred = (y_prob >= threshold).astype(int)

    contacted = y_pred == 1

    # profit only from contacted users
    expected_profit = np.sum(y_prob[contacted] * B_sub - C_call)

    return {
        "AUC_PR": average_precision_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "% Contacted": contacted.mean() * 100,
        "Expected Profit": expected_profit,
    }
