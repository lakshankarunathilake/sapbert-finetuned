#!/usr/bin/env python3
"""
Visualize SAPBERT FAISS index with PCA -> t-SNE (or UMAP).

Assumes your index creator saved:
  - <index_path>.faiss
  - <index_path>_processed_data.csv
  - <index_path>_config.json  (optional)
  - <index_path>_metadata.pkl (optional)

Labels can be 'entity_id' (default), 'primary_alias', or any column in processed_data CSV.
"""

import os, json, pickle, argparse, math, logging, warnings
import numpy as np
import pandas as pd
import faiss
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("viz")

def read_index(base_path: str):
    path = base_path if base_path.endswith(".faiss") else base_path + ".faiss"
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    idx = faiss.read_index(path)
    # --- NEW: ensure IVF has a direct map so reconstruct(key) works ---
    core = idx
    # If wrapped in IDMap2 or PreTransform, unwrap to the IVF core
    if isinstance(core, faiss.IndexIDMap2):
        core = core.index
    if isinstance(core, faiss.IndexPreTransform):
        core = core.index

    if isinstance(core, faiss.IndexIVF):
        # Build direct map from current inverted lists
        core.make_direct_map()  # <-- key line
        # (optional) persist it so you don't have to rebuild next time:
        # faiss.write_index(idx, path)  # uncomment to overwrite file with direct map baked in

    return idx

def try_load_processed_csv(base_path: str):
    csv_path = base_path + "_processed_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

def try_load_config(base_path: str):
    cfg_path = base_path + "_config.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def try_load_metadata(base_path: str):
    meta_path = base_path + "_metadata.pkl"
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return None

def reconstruct_batchwise(index, ids, batch_size=1000):
    """
    Reconstruct vectors from FAISS index with progress tracking.
    Works for IVF/Flat/HNSW when direct map is available.
    """
    d = index.d
    X = np.zeros((len(ids), d), dtype=np.float32)

    # tqdm gives a progress bar that updates in real time
    for i in tqdm(range(0, len(ids), batch_size), desc="Reconstructing vectors"):
        batch_ids = ids[i:i + batch_size]
        for j, idx in enumerate(batch_ids):
            x = np.zeros(d, dtype=np.float32)
            try:
                index.reconstruct(int(idx), x)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Reconstruction failed at index {idx}. "
                    f"Ensure direct map is built (core.make_direct_map())."
                ) from e
            X[i + j] = x
    return X

def maybe_normalize(X, do_norm):
    if do_norm:
        faiss.normalize_L2(X)
    return X

def safe_tsne(Z, seed, perplexity, n_iter, metric):
    import sklearn
    from inspect import signature

    # clamp perplexity to valid range
    max_perp = max(5, (len(Z) - 1) // 3)
    perp = min(perplexity, max_perp)

    base_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        metric=metric,
        init="pca",
        random_state=seed,
        verbose=1,
        # method="barnes_hut",  # optionally force BH; sklearn will choose automatically
    )

    # Some environments use 'n_iter', others (rarely) use 'max_iter'
    ctor = TSNE.__init__
    params = set(signature(ctor).parameters.keys())

    if "n_iter" in params:
        base_kwargs["n_iter"] = n_iter
    elif "max_iter" in params:
        base_kwargs["max_iter"] = n_iter  # fallback name
    # else: no iter control available; let defaults apply

    # 'n_jobs' exists only in sklearn >= 1.4
    try:
        major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    except Exception:
        major, minor = (0, 0)
    if "n_jobs" in params and (major, minor) >= (1, 4):
        base_kwargs["n_jobs"] = -1

    # Filter kwargs to only what this TSNE actually accepts
    filtered = {k: v for k, v in base_kwargs.items() if k in params}

    return TSNE(**filtered).fit_transform(Z)


def try_umap(Z, seed, n_neighbors, min_dist, metric):
    try:
        import umap
    except Exception as e:
        raise RuntimeError(
            "UMAP is not installed. Run `pip install umap-learn` or use --algo tsne"
        ) from e
    return umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=seed
    ).fit_transform(Z)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_path", required=True, help="Path prefix to index (without .faiss)")
    ap.add_argument("--label_field", type=str, default="entity_id",
                    help="Column in *_processed_data.csv used for coloring (e.g., entity_id, primary_alias)")
    ap.add_argument("--max_points", type=int, default=10000, help="Subsample size")
    ap.add_argument("--pca_dim", type=int, default=50, help="PCA dims before t-SNE/UMAP (0 to skip)")
    ap.add_argument("--algo", choices=["tsne","umap"], default="tsne", help="Projection algorithm")
    ap.add_argument("--tsne_perplexity", type=int, default=30)
    ap.add_argument("--tsne_iter", type=int, default=1000)
    ap.add_argument("--umap_neighbors", type=int, default=30)
    ap.add_argument("--umap_min_dist", type=float, default=0.05)
    ap.add_argument("--metric", type=str, default="euclidean", help="Distance metric")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize reconstructed vectors (safe if unsure)")
    ap.add_argument("--topk_legend", type=int, default=15, help="Max categories shown in legend")
    ap.add_argument("--out_png", type=str, default=None, help="Output PNG path")
    args = ap.parse_args()

    base = args.index_path
    out_png = args.out_png or (base + f"_{args.algo}.png")

    # 1) Load artifacts
    index = read_index(base)
    cfg = try_load_config(base) or {}
    pdf = try_load_processed_csv(base)
    meta = try_load_metadata(base)
    ntotal = index.ntotal
    log.info(f"Index ntotal={ntotal}, dim={index.d}, trained={getattr(index, 'is_trained', None)}")

    # 2) Decide sample positions (index insertion order == FAISS positions in your creator)
    rng = np.random.default_rng(args.seed)
    if ntotal == 0:
        raise RuntimeError("Empty index.")
    if ntotal > args.max_points:
        pos = rng.choice(ntotal, size=args.max_points, replace=False)
    else:
        pos = np.arange(ntotal)
    pos.sort()

    # 3) Reconstruct vectors
    log.info("Reconstructing vectors (can take a bit for IVF/HNSW)...")
    X = reconstruct_batchwise(index, pos)
    if args.normalize:
        X = maybe_normalize(X, True)

    # 4) Labels
    labels = None
    label_title = args.label_field
    if pdf is not None and args.label_field in pdf.columns:
        # processed_data rows align with index positions (index_id in your code)
        try:
            labels = pdf.iloc[pos][args.label_field].astype(str).values
            log.info(f"Using labels from column '{args.label_field}' ({len(set(labels))} unique in sample).")
        except Exception as e:
            warnings.warn(f"Failed to align labels: {e}")
    else:
        log.warning(f"No processed_data CSV or missing column '{args.label_field}'. Plot will be unlabeled.")

    # 5) PCA (optional)
    Z = X
    if args.pca_dim and X.shape[1] > args.pca_dim:
        log.info(f"Running PCA -> {args.pca_dim}D")
        Z = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(X)

    # 6) 2D projection
    log.info(f"Running {args.algo.upper()}...")
    if args.algo == "tsne":
        Y = safe_tsne(Z, args.seed, args.tsne_perplexity, args.tsne_iter, args.metric)
    else:
        Y = try_umap(Z, args.seed, args.umap_neighbors, args.umap_min_dist, args.metric)

    # 7) Plot
    plt.figure(figsize=(8.5, 8.5))
    if labels is None:
        plt.scatter(Y[:, 0], Y[:, 1], s=4, alpha=0.7)
        plt.title(f"{args.algo.upper()} of SAPBERT embeddings (no labels)")
    else:
        ser = pd.Series(labels)
        # order by frequency so legend shows the most common first
        ordered = ser.value_counts().index.tolist()
        shown = set()
        for lab in ordered:
            mask = (ser.values == lab)
            plt.scatter(Y[mask, 0], Y[mask, 1], s=4, alpha=0.75, label=lab if len(shown) < args.topk_legend else None)
            if len(shown) < args.topk_legend: shown.add(lab)
        if shown:
            plt.legend(markerscale=3, frameon=False, fontsize=8, title=label_title, loc="best")
        plt.title(f"{args.algo.upper()} of SAPBERT embeddings (colored by {label_title})")

    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    log.info(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
