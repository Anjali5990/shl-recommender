"""
Build-time step (run once, not at request time):
Embeds every catalog item's `search_text` with a sentence-transformer and
saves the matrix to disk. The API loads this .npy at startup -- it never
computes embeddings live, which keeps /chat fast and avoids a slow cold
start pulling model weights during a request.

Run this after clean.py, and re-run it whenever cleaned_catalog.json changes:
    python app/ingestion/clean.py
    python app/ingestion/embed_catalog.py

Requires internet access (downloads the model from Hugging Face the first
time). If you're in an offline/restricted environment, retrieval.py will
automatically fall back to TF-IDF-only lexical search -- the service still
works, it just loses semantic/synonym matching until this step is run
somewhere with network access.
"""
import json
from pathlib import Path

import numpy as np

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "cleaned_catalog.json"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "vector_store"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer

    catalog = json.loads(CATALOG_PATH.read_text())
    texts = [item["search_text"] for item in catalog]
    ids = [item["id"] for item in catalog]

    print(f"Embedding {len(texts)} catalog items with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=True
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "catalog_dense.npy", embeddings)
    (OUT_DIR / "catalog_ids.json").write_text(json.dumps(ids))
    (OUT_DIR / "model_name.txt").write_text(MODEL_NAME)

    print(f"Saved embeddings {embeddings.shape} -> {OUT_DIR / 'catalog_dense.npy'}")


if __name__ == "__main__":
    main()
