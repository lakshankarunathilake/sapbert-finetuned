# BC5CDR Reranker (AdapterHub + FAISS)

A lightweight, **reranker-only** training and evaluation pipeline for the **BC5CDR** concept linking task using:

* **AdapterHub** adapters on top of any Hugging Face Transformer model
* **AdapterTrainer** for efficient adapter-only finetuning
* **FAISS** for fast top‑K retrieval against a MeSH entity index

This repo trains a **listwise reranker** that scores (query, candidate_alias) pairs and optimizes a **group softmax** loss per query over the retrieved candidates. It also provides **batched evaluation** with multiple metrics (Top‑1, Top‑5, MRR, Recall@K).

---

## ✨ Highlights

* Fixed labels/label key bug; correct use of `label` (not `labels`)
* Proper device placement and no‑grad encoding for speed
* **10–50× faster** batched evaluation
* Metrics: **Top‑1**, **Top‑5**, **MRR**, **Retrieval Recall@K**
* **Progress bars** for all long‑running stages
* Robust error handling
* Pluggable **FAISS** index for entity retrieval

---

## 🧱 Requirements

```bash
pip install torch transformers adapter-transformers datasets tqdm faiss-cpu
# or, for GPUs with FAISS support
# pip install faiss-gpu
```

> **Note:** You also need a prebuilt FAISS MeSH entity index (see **FAISS Index Layout** below).

---

## 📦 Dataset

* Loads **BC5CDR** from Hugging Face via `datasets.load_dataset("bigbio", "bc5cdr")`.
* Uses `train`, `validation`, and `test` splits.
* Mentions are collected from `entities` and passages are concatenated to form document context when available.

---

## 🗂️ FAISS Index Layout

Pass `--faiss_index_path /path/to/index_without_extension`. The code expects the following files to exist:

```
/path/to/index_without_extension.faiss
/path/to/index_without_extension_metadata.pkl
/path/to/index_without_extension_config.json
```

Where:

* **`.faiss`**: the serialized FAISS index
* **`_metadata.pkl`**: Python dict mapping integer ids → entity info (e.g., `entity_id`, `primary_alias`, `processed_text`, `aliases`, `all_aliases`)
* **`_config.json`**: metadata such as `index_type`, `model_name`, `max_length`, `created_at`

> Retrieval returns top‑K **indices**, which are then mapped to aliases/IDs via `_metadata.pkl`.

---

## 🔧 Models & Adapters

* **Base model**: any `transformers` checkpoint (e.g., `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`).
* **Retriever adapter**: a frozen bi‑encoder adapter (e.g., `hub:AdapterHub/sapbert_retriever`). Loaded and **kept frozen** to encode queries for FAISS search.
* **Reranker adapter**: a **new** or **pretrained** adapter stack with a 1‑logit head (binary score). Trained **only** on the adapter parameters using `AdapterTrainer`.

---

## 🔎 Query Construction

Two modes via `--query_mode`:

* `mention` — raw mention text
* `context` — builds a context‑aware query from document text: `…left window… [MENTION] mention [/MENTION] …right window…` (controlled by `--context_window`, default 64 tokens on each side)

---

## 🚀 Quickstart

### Train

```bash
python bc5cdr_rerank_trainer.py \
  --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --retriever_adapter hub:AdapterHub/sapbert_retriever \
  --faiss_index_path /Users/lakshankarunathilake/PycharmProjects/sapbert/utils/NEL/indexes/mesh_adapter \
  --rerank_adapter_name link_rerank \
  --output_dir ./out/rerank_link \
  --k 50 --epochs 2 --per_device_train_batch_size 2 --lr 5e-5 \
  --query_mode context --context_window 64
```

### Evaluate (no training)

```bash
python bc5cdr_rerank_trainer.py \
  --base_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --retriever_adapter hub:AdapterHub/sapbert_retriever \
  --faiss_index_path ./indexes/mesh_entities_index \
  --rerank_adapter_load ./out/rerank_link \
  --k 50 --evaluate_only --query_mode context
```

---

## 🏗️ What the Script Does (Step‑by‑Step)

1. **Load dataset** (`bigbio/bc5cdr`).
2. **Load FAISS index** (and its metadata/config) via `FAISSRetriever`.
3. **Load frozen retriever** + tokenizer + mean‑pooler, and encode queries.
4. **Retrieve top‑K candidates** from FAISS for each query.
5. **Assemble listwise training data**: for every query with the gold entity in top‑K, build [gold alias + negatives].
6. **Build reranker model**: add or load an adapter (`--rerank_adapter_name`) with a 1‑logit head, activate/train only the adapter.
7. **Train** with `AdapterTrainer` using a **custom listwise loss** (group softmax per `query_id`).
8. **Evaluate** on the test set using **batched scoring** and report: Retrieval Recall@K, Top‑1, Top‑5, **MRR**.

---

## ⚙️ Command‑Line Arguments

| Arg                             | Type / Default   | Description                                                                                          |
| ------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| `--base_model`                  | **required**     | HF model id or path for both retriever and reranker backbones.                                       |
| `--retriever_adapter`           | **required**     | Path or `hub:<org/name>` to a **frozen** bi‑encoder adapter used to encode queries for FAISS search. |
| `--faiss_index_path`            | **required**     | Path **without extension** to the FAISS index triplet (see **FAISS Index Layout**).                  |
| `--rerank_adapter_name`         | `link_rerank`    | Name under which the reranker adapter stack is created/activated.                                    |
| `--rerank_adapter_load`         | `None`           | Directory of a **pretrained** reranker adapter (loads adapter + head).                               |
| `--output_dir`                  | `./out/reranker` | Where to save the adapter after training.                                                            |
| `--k`                           | `50`             | Number of FAISS candidates per query.                                                                |
| `--max_length`                  | `256`            | Max length for tokenizer when scoring pairs.                                                         |
| `--query_mode`                  | `context`        | `mention` or `context`.                                                                              |
| `--context_window`              | `64`             | Tokens taken left/right of the mention to build the context query.                                   |
| `--epochs`                      | `2`              | Number of training epochs.                                                                           |
| `--per_device_train_batch_size` | `2`              | Training batch size per device.                                                                      |
| `--per_device_eval_batch_size`  | `4`              | (HF) Used by trainer if you later enable eval.                                                       |
| `--eval_batch_size`             | `32`             | Batch size for **batched reranking during evaluation**.                                              |
| `--lr`                          | `5e-5`           | Learning rate.                                                                                       |
| `--weight_decay`                | `0.0`            | Weight decay for AdamW.                                                                              |
| `--seed`                        | `13`             | RNG seed.                                                                                            |
| `--evaluate_only`               | flag             | Skip training; only run evaluation steps.                                                            |
| `--train_split`                 | `validation`     | Which split to use for training: `train` or `validation`.                                            |

---

## 📤 Outputs

* `--output_dir` contains the saved reranker **adapter** and head after training.
* Final console summary prints:

  * Dataset coverage (# with gold in top‑K)
  * Retrieval **Recall@K**
  * Reranker **Top‑1**, **Top‑5**, **MRR**

---

## 📈 Metrics Explained

* **Retrieval Recall@K**: fraction of mentions where the gold entity is present in FAISS top‑K candidates (before reranking).
* **Top‑1 / Top‑5**: accuracy after reranking among candidates.
* **MRR**: mean reciprocal rank of the gold entity after reranking.

> The evaluation skips a mention if its gold concept is **not** found in top‑K; metrics are normalized by the number of evaluated mentions.

---

## 🔍 Tips & Good Practices

* **GPU recommended** for training; CPU is fine for indexing/inference if time is not critical.
* Keep the retriever **frozen** and well‑aligned with your FAISS index (same base model/tokenizer).
* For better recall@K, consider improving the **retriever**/index rather than the reranker.
* Start with `--train_split validation` to iterate fast, then switch to `--train_split train` for full training.
* Use `--query_mode context` for disambiguation in noisy biomedical text.

---

## 🧪 Reproducibility

* Script sets `torch.manual_seed(--seed)`.
* FAISS search uses L2‑normalized embeddings and cosine similarity.
* Mean pooling over token embeddings is used for query encoding.

---

## 🛠️ Troubleshooting

* **FAISS not available**: install `faiss-cpu` or `faiss-gpu`.
* **Index load errors**: check the three expected files and that paths match. Ensure `_metadata.pkl` keys are integer indices mapping to entity info dicts.
* **Gold not in top‑K**: counts against retrieval recall and is **skipped** for reranker metrics; increase `--k` or improve the retriever/index.
* **OOM during evaluation**: reduce `--eval_batch_size` or `--max_length`.
* **Adapter head missing**: if creating a new adapter, the script adds a single‑logit classification head automatically.

---

## 📚 Acknowledgments

* Built on **Hugging Face** `transformers`, **AdapterHub** `adapter-transformers`, **datasets`, and **FAISS**.
* BC5CDR dataset provided via the **bigbio** community dataset hub.

---

## 📄 License

This code is provided as‑is, under the same license as this repository (add your license file and reference here).

---

## 🔗 Citation

If you use this code or its results in academic work, please cite the appropriate resources for BC5CDR, FAISS, AdapterHub, and the base model you used.
