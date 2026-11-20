"""Stage 3: embed reviews with all-MiniLM-L6-v2 and store them in a local ChromaDB.

MiniLM reads at most 256 word pieces, and many laptop reviews are longer, so each review
is split into chunks of a few sentences (<= CHUNK_WORDS words). Search runs over chunks
and is then grouped back to reviews, because summaries cite whole reviews by review_id.
"""

import re

from .config import CHROMA_DIR, EMBED_MODEL

COLLECTION = "reviews"
CHUNK_WORDS = 80
_SENT = re.compile(r"(?<=[.!?])\s+")
_model = None


class _OnnxMiniLM:
    """all-MiniLM-L6-v2 through ChromaDB's ONNX runtime build (no PyTorch needed, so the
    deployed app stays small). Same vectors as sentence-transformers (cosine 1.000)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._ef = ONNXMiniLM_L6_V2()

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np

        vecs = np.vstack([np.asarray(self._ef(texts[i:i + batch_size]), dtype=np.float32)
                          for i in range(0, len(texts), batch_size)])
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def embedder():
    global _model
    if _model is None:
        assert EMBED_MODEL.endswith("all-MiniLM-L6-v2")
        _model = _OnnxMiniLM()
    return _model


def chunk(title, text, max_words=CHUNK_WORDS):
    """Split a review into chunks of whole sentences; the title opens the first chunk."""
    chunks, cur = [], f"{title}." if title else ""
    for sent in _SENT.split(text):
        if cur and len((cur + " " + sent).split()) > max_words:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def collection(reset=False):
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build(reviews, batch_size=64):
    """(Re)build the index from a reviews DataFrame (review_id, product_id, rating, title, text)."""
    col = collection(reset=True)
    ids, docs, metas = [], [], []
    for r in reviews.itertuples():
        title = r.title if isinstance(r.title, str) else ""
        for k, c in enumerate(chunk(title, r.text)):
            ids.append(f"{r.review_id}#{k}")
            docs.append(c)
            metas.append({"review_id": r.review_id, "product_id": r.product_id, "rating": int(r.rating)})
    vectors = embedder().encode(docs, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=True)
    step = 2000
    for i in range(0, len(ids), step):
        col.add(ids=ids[i:i + step], documents=docs[i:i + step],
                metadatas=metas[i:i + step], embeddings=vectors[i:i + step].tolist())
    return col.count()


def ensure(reviews):
    """Build the index if it is missing or empty (for example on a fresh deployment)."""
    try:
        n = collection().count()
    except Exception:
        n = 0
    return n if n else build(reviews)


def search(query, product_id=None, k=10, where=None, n_chunks=None):
    """Return up to k reviews most similar to the query, best chunk per review.

    Each hit: {"review_id", "score" (cosine similarity), "chunk" (best matching text)}.
    """
    col = collection()
    filters = dict(where or {})
    if product_id:
        filters["product_id"] = product_id
    if len(filters) > 1:
        filters = {"$and": [{key: val} for key, val in filters.items()]}
    q = embedder().encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q, n_results=n_chunks or k * 4, where=filters or None)
    best = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rid, score = meta["review_id"], 1 - dist
        if rid not in best or score > best[rid]["score"]:
            best[rid] = {"review_id": rid, "score": round(score, 4), "chunk": doc}
    return sorted(best.values(), key=lambda h: -h["score"])[:k]
