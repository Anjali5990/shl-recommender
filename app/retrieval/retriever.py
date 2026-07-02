"""
Retrieval layer.

Design: three signals combined per candidate, not one:

1. Dense semantic similarity (sentence-transformer embeddings, cosine sim).
   Catches synonyms/paraphrase: "Spring framework" -> "Java Frameworks".
   Loaded from a pre-built .npy (see ingestion/embed_catalog.py). If that
   file isn't present (e.g. offline environment), this signal is skipped
   and we fall back to TF-IDF + name-matching only -- the service still
   works, just with weaker synonym handling.

2. TF-IDF lexical similarity (scikit-learn, fit in-memory at startup, no
   downloads). Catches exact skill/product tokens dense embeddings can
   sometimes dilute in a short query ("Docker", "HIPAA", "Rust").

3. Exact/fuzzy catalog name matching against the latest user turn. This is
   what grounds "compare X vs Y" and "add Y to the shortlist" -- if the user
   names a real catalog item, it must appear in the candidate set even if
   its embedding similarity to the whole conversation is mediocre.

The retriever only returns *candidates* -- it never decides what to
recommend. The agent layer picks from these candidates and the app then
validates the LLM's picks against this same catalog before returning them,
so a hallucinated name/URL can never reach the user.
"""
import difflib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "cleaned_catalog.json"
VECTOR_DIR = Path(__file__).parent.parent.parent / "data" / "vector_store"


class Retriever:
    def __init__(self):
        self.catalog = json.loads(CATALOG_PATH.read_text())
        self.by_id = {item["id"]: item for item in self.catalog}
        self.texts = [item["search_text"] for item in self.catalog]
        self.names_lower = [item["name"].lower() for item in self.catalog]

        # TF-IDF: always available, no external dependency
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

        # Dense embeddings: optional, loaded if the build step was run
        self.dense = None
        self.dense_model = None
        dense_path = VECTOR_DIR / "catalog_dense.npy"
        ids_path = VECTOR_DIR / "catalog_ids.json"
        if dense_path.exists() and ids_path.exists():
            saved_ids = json.loads(ids_path.read_text())
            if saved_ids == [item["id"] for item in self.catalog]:
                self.dense = np.load(dense_path)
                self._try_load_model()
            else:
                print("WARNING: catalog_ids.json doesn't match current catalog "
                      "order -- re-run embed_catalog.py. Skipping dense retrieval.")
        else:
            print("INFO: no dense embeddings found at data/vector_store/. "
                  "Run app/ingestion/embed_catalog.py for semantic retrieval. "
                  "Falling back to TF-IDF + name matching only.")

    def _try_load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = (VECTOR_DIR / "model_name.txt").read_text().strip()
            self.dense_model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"WARNING: could not load sentence-transformer model at "
                  f"request time ({e}). Dense query embedding disabled; "
                  f"catalog-side dense vectors will be ignored this run.")
            self.dense = None

    _STOPWORDS = {
        "the", "and", "for", "are", "what", "who", "how", "why", "with",
        "this", "that", "does", "did", "not", "add", "our", "you", "your",
        "test", "tests", "assessment", "assessments", "use", "used", "new",
        "between", "difference", "compare", "vs", "need", "want", "hire",
        "hiring", "role", "job", "candidate", "candidates", "level", "yes",
        "can", "will", "should", "please", "also", "any", "all", "have",
        "has", "had", "was", "were", "been", "from", "into", "than",
    }

    def _name_matches(self, query: str, limit: int = 5, cutoff: float = 0.72):
        """Fuzzy-match catalog names literally mentioned in the query text.

        Checks substring containment in BOTH directions: either the query
        contains a full catalog name, or an acronym-like token in the query
        ("OPQ32r", "GSA") is itself a substring of a longer official catalog
        name ("Occupational Personality Questionnaire OPQ32r"). Common
        English words are excluded from the short-token check so "the" or
        "and" don't spuriously match every name that happens to contain
        those three letters somewhere.
        """
        query_lower = query.lower()
        hits = set()
        for i, name in enumerate(self.names_lower):
            if len(name) <= 3:
                continue
            if name in query_lower or self._short_form_in(query, name):
                hits.add(i)
        candidates = difflib.get_close_matches(
            query_lower, self.names_lower, n=limit, cutoff=cutoff
        )
        for c in candidates:
            hits.add(self.names_lower.index(c))
        return hits

    def _short_form_in(self, query: str, name_lower: str) -> bool:
        """Token from the ORIGINAL-case query is acronym-like (has a digit,
        or is 2+ uppercase letters) and appears in the catalog name."""
        for raw_token in query.replace(",", " ").replace("?", " ").split():
            token = raw_token.strip(".:;()")
            token_lower = token.lower()
            if len(token_lower) < 3 or token_lower in self._STOPWORDS:
                continue
            looks_like_acronym = any(ch.isdigit() for ch in token) or (
                sum(1 for ch in token if ch.isupper()) >= 2
            )
            if looks_like_acronym and token_lower in name_lower:
                return True
        return False

    def retrieve(self, query_text: str, latest_user_message: str, top_k: int = 20):
        """
        query_text: representation of the whole conversation so far (used for
                    semantic + lexical similarity)
        latest_user_message: just the newest turn (used for name matching --
                    we only want to force-include items the user *just*
                    named, not everything ever mentioned in a long thread)
        """
        n = len(self.catalog)
        scores = np.zeros(n, dtype=np.float32)

        tfidf_q = self.tfidf.transform([query_text])
        tfidf_scores = cosine_similarity(tfidf_q, self.tfidf_matrix)[0]
        scores += 0.4 * tfidf_scores

        if self.dense is not None and self.dense_model is not None:
            q_emb = self.dense_model.encode([query_text], normalize_embeddings=True)
            dense_scores = cosine_similarity(q_emb, self.dense)[0]
            scores += 0.6 * dense_scores
        else:
            # no dense signal available -- lean fully on lexical
            scores += 0.6 * tfidf_scores

        ranked_idx = np.argsort(-scores)[:top_k].tolist()

        name_hit_idx = self._name_matches(latest_user_message)
        combined = list(dict.fromkeys(list(name_hit_idx) + ranked_idx))  # union, name hits first

        results = []
        for i in combined[: max(top_k, len(name_hit_idx))]:
            item = dict(self.catalog[i])
            item["_score"] = float(scores[i])
            item["_name_matched"] = i in name_hit_idx
            results.append(item)
        return results

    def find_by_name(self, name: str):
        """Exact/near-exact lookup, used to validate LLM output against catalog.

        This is the grounding gate: the agent may only present an item to
        the user if find_by_name (or find_by_url) resolves it to a real
        catalog record. Order of checks, strictest first:
        1. exact (case-insensitive) name match
        2. substring containment either direction (short forms/acronyms)
        3. fuzzy match as a last resort, high cutoff to avoid false positives
        """
        name_lower = name.strip().lower()
        if not name_lower:
            return None
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # here `name` is presumed to BE an assessment name (not a whole
        # sentence), so plain substring containment is safe in both
        # directions -- no stopword risk like in the free-text query case
        if len(name_lower) >= 4:
            for i, catalog_name in enumerate(self.names_lower):
                if len(catalog_name) > 3 and (
                    catalog_name in name_lower or name_lower in catalog_name
                ):
                    return self.catalog[i]
        matches = difflib.get_close_matches(name_lower, self.names_lower, n=1, cutoff=0.85)
        if matches:
            idx = self.names_lower.index(matches[0])
            return self.catalog[idx]
        return None

    def extract_mentioned_items(self, text: str):
        """Return catalog items whose exact name appears as a substring of
        `text`. Used to reconstruct what the agent already recommended in
        earlier turns -- since the API is stateless, the only record of the
        prior shortlist is the assistant's own past reply text, so we
        re-derive it by scanning that text against the real catalog rather
        than trusting the LLM to remember it accurately."""
        text_lower = text.lower()
        found = []
        for item in self.catalog:
            if len(item["name"]) > 3 and item["name"].lower() in text_lower:
                found.append(item)
        return found

    def find_by_url(self, url: str):
        for item in self.catalog:
            if item["url"].rstrip("/") == url.rstrip("/"):
                return item
        return None


_retriever_singleton = None


def get_retriever() -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever()
    return _retriever_singleton
