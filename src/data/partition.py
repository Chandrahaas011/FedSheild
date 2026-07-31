"""Client data partitioning for federated simulation.

Two schemes:
  - IID: stratified random split across clients (same class proportions everywhere).
  - Dirichlet(alpha): per-class Dirichlet-distributed split (Hsu et al. 2019 / Yurochkin
    et al. 2019 non-IID partitioning). Smaller alpha => more label skew per client.
"""
import numpy as np


def partition_iid(y: np.ndarray, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        splits = np.array_split(idx_c, n_clients)
        for i, s in enumerate(splits):
            client_indices[i].append(s)
    return [rng.permutation(np.concatenate(idxs)) for idxs in client_indices]


def partition_dirichlet(y: np.ndarray, n_clients: int, alpha: float, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.repeat(alpha, n_clients))
        cut_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        splits = np.split(idx_c, cut_points)
        for i, s in enumerate(splits):
            client_indices[i].append(s)
    result = [rng.permutation(np.concatenate(idxs)) if idxs else np.array([], dtype=int) for idxs in client_indices]
    return result


def partition_data(y: np.ndarray, n_clients: int, alpha, seed: int = 42) -> list[np.ndarray]:
    """alpha=None or 'iid' -> IID split; else Dirichlet(alpha) non-IID split."""
    if alpha is None or (isinstance(alpha, str) and alpha.lower() == "iid"):
        return partition_iid(y, n_clients, seed)
    return partition_dirichlet(y, n_clients, float(alpha), seed)


def client_class_counts(client_indices: list[np.ndarray], y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.zeros((len(client_indices), n_classes), dtype=int)
    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        vals, c = np.unique(y[idx], return_counts=True)
        counts[i, vals] = c
    return counts


def plot_client_distribution(client_indices, y, class_names, out_path, title=""):
    import matplotlib.pyplot as plt

    counts = client_class_counts(client_indices, y, len(class_names))
    proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    palette = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(client_indices) * 0.5), 5))
    bottom = np.zeros(len(client_indices))
    for c_idx, c_name in enumerate(class_names):
        ax.bar(
            range(len(client_indices)), proportions[:, c_idx], bottom=bottom,
            label=c_name, color=palette[c_idx % len(palette)], width=0.85,
        )
        bottom += proportions[:, c_idx]
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Class proportion")
    ax.set_title(title)
    ax.set_xlim(-0.6, len(client_indices) - 0.4)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts
