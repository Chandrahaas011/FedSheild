"""ROADMAP.md Phase 2 gate deliverable: per-client class distribution at alpha in
{0.1, 0.5, 1.0, IID}.

    python -m experiments.make_partition_figures
"""
from pathlib import Path

from src.data.partition import partition_data, plot_client_distribution
from src.fl.dataset import load_processed

OUT_DIR = Path("results/figures")


def main():
    splits, metadata = load_processed()
    _, y_train = splits["train"]
    class_names = metadata["classes"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for alpha in [0.1, 0.5, 1.0, "iid"]:
        client_indices = partition_data(y_train, n_clients=10, alpha=alpha, seed=0)
        label = "IID" if alpha == "iid" else f"alpha={alpha}"
        out_path = OUT_DIR / f"partition_alpha_{alpha}.png"
        plot_client_distribution(
            client_indices, y_train, class_names, out_path,
            title=f"Per-client class distribution ({label}, n_clients=10)",
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
