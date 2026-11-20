"""UNIT 3 MODULE: memory + vector retrieval + feedback-based adaptation.

- Short-term memory : the LangGraph state for the current request (graph.py)
- Long-term memory  : SQLite tables `memory` and `feedback`, persisted across sessions
- Vector retrieval  : symptom -> specialty lookup in a Chroma vector store (LangChain + Ollama embeddings), TF-IDF fallback
- Feedback adaptation: accepted/rejected proposals change future ranking
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import db
import llm

# ---------------------------------------------------------------- vector store (RAG)
# Short documents (one per symptom group) so short patient phrases match well.
KB = [
    ("cardiology", "chest pain"), ("cardiology", "palpitations, racing or irregular heartbeat"),
    ("cardiology", "shortness of breath, high blood pressure, angina, heart problems"),
    ("neurology", "headache, migraine, dizziness"), ("neurology", "seizures, numbness, tingling, tremor"),
    ("neurology", "memory loss, stroke symptoms"),
    ("dermatology", "skin rash, itching, eczema"), ("dermatology", "acne, hair loss, psoriasis, skin allergy"),
    ("orthopedics", "bone fracture, joint pain, knee pain"), ("orthopedics", "back pain, sprain, shoulder, arthritis, spine"),
    ("pediatrics", "child, baby, infant, newborn, toddler"), ("pediatrics", "vaccination, growth check, fever in children"),
    ("general", "fever, cold, cough, flu"), ("general", "weakness, body ache, stomach infection, general checkup"),
]

CONFIG = {"min_sim": 0.5}  # cosine-similarity threshold for the Chroma backend (tunable in the sidebar)


class VectorStore:
    """Fallback: TF-IDF vectors + cosine similarity (no embedding model or Chroma needed)."""

    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(ngram_range=(1, 2)).fit([d[1] for d in docs])
        self.matrix = self.vec.transform([d[1] for d in docs])

    def search(self, query, k=1):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [(self.docs[i][0], float(sims[i])) for i in order]


_tfidf = None
_chroma = None  # None = not built yet, False = unavailable


def _chroma_store():
    """LangChain + Chroma vector store with Ollama embeddings (nomic-embed-text), built once per process."""
    global _chroma
    if _chroma is None:
        try:
            emb = llm.embedder()
            if emb is None:
                raise RuntimeError("embeddings unavailable")
            from langchain_chroma import Chroma
            from langchain_core.documents import Document
            vs = Chroma(collection_name="symptoms", embedding_function=emb,
                        collection_metadata={"hnsw:space": "cosine"})
            vs.add_documents([Document(page_content=t, metadata={"specialty": s}) for s, t in KB])
            _chroma = vs
        except Exception:
            _chroma = False
    return _chroma


def _chroma_search(query):
    vs = _chroma_store()
    if not vs:
        return None
    doc, dist = vs.similarity_search_with_score(query, k=1)[0]  # cosine distance
    return doc.metadata["specialty"], 1.0 - float(dist)


def retrieve_specialty(symptoms, threshold=0.08):
    """Hybrid retrieval. Returns (specialty or None, similarity, backend)."""
    global _tfidf
    if llm.CONFIG["use_llm"]:
        try:
            r = _chroma_search(symptoms)
        except Exception:
            r = None
        if r and r[1] >= CONFIG["min_sim"]:
            return r[0], r[1], "chroma"
    if _tfidf is None:
        _tfidf = VectorStore(KB)
    spec, score = _tfidf.search(symptoms)[0]
    return (spec, score, "tfidf") if score >= threshold else (None, score, "tfidf")


# ---------------------------------------------------------------- long-term memory
def get_memory(pid):
    c = db.conn()
    out = {r["key"]: r["value"] for r in c.execute("SELECT key,value FROM memory WHERE patient_id=?", (pid,))}
    c.close()
    return out


def set_memory(pid, key, value):
    c = db.conn()
    c.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?)", (pid, key, value))
    c.commit()
    c.close()


def feedback_counts(pid, doctor_id):
    c = db.conn()
    rows = c.execute("SELECT event, COUNT(*) n FROM feedback WHERE patient_id=? AND doctor_id=? GROUP BY event",
                     (pid, doctor_id)).fetchall()
    c.close()
    d = {r["event"]: r["n"] for r in rows}
    return d.get("accepted", 0), d.get("rejected", 0)


# ---------------------------------------------------------------- feedback-based adaptation
def record_feedback(pid, doctor_id, event):
    c = db.conn()
    c.execute("INSERT INTO feedback(patient_id,doctor_id,event) VALUES (?,?,?)", (pid, doctor_id, event))
    c.commit()
    c.close()
    acc, _ = feedback_counts(pid, doctor_id)
    if event == "accepted" and acc >= 2:  # learned preference after repeated acceptance
        set_memory(pid, "preferred_doctor", db.get_doctor(doctor_id)["name"])


def doctor_score(pid, doctor_id, doctor_name):
    """+2 preferred doctor, +1 per past acceptance (max 3), -2 per past rejection."""
    acc, rej = feedback_counts(pid, doctor_id)
    score = min(acc, 3) - 2 * rej
    if get_memory(pid).get("preferred_doctor") == doctor_name:
        score += 2
    return score
