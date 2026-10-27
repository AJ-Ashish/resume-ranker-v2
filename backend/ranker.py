"""
ranker.py
Core scoring engine: sentence embeddings + cosine similarity, plus
keyword-based skill extraction for explainability.

Render-friendly version:
- Uses a smaller sentence-transformer model
- CPU-only inference
- No gradient tracking
- No sklearn runtime import
- Smaller max sequence length
"""

import os
import re
from functools import lru_cache

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

torch.set_num_threads(1)

_model = None


def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu",
        )
        _model.max_seq_length = 256
        _model.eval()

    return _model


COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "html", "css", "rest api", "graphql", "mongodb", "postgresql", "mysql",
    "data analysis", "data visualization", "excel", "power bi", "tableau",
    "communication", "leadership", "project management", "agile", "scrum",
]


def extract_skills(text: str) -> set:
    text_lower = text.lower()
    found = set()

    for skill in COMMON_SKILLS:
        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(skill)
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(pattern, text_lower):
            found.add(skill)

    return found


def _encode(texts: list[str]):
    """Generate normalized embeddings with minimal inference overhead."""
    model = get_model()

    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return embeddings


@lru_cache(maxsize=512)
def _embed_one(text: str):
    """
    Embedding for one text, remembered in memory. Browsing jobs scores one resume
    against every open job, so without this the same resume (and the same job
    descriptions) would be re-embedded on every page load. Results are identical.
    """
    return _encode([text])[0]


def combine_score(semantic: float, job_skills: set, resume_skills: set, must_have: set) -> float:
    """
    Blend meaning similarity with skill coverage when HR has marked must-have skills.

        weighted coverage = (2 x must-haves matched + nice-to-haves matched)
                            / (2 x must-haves total  + nice-to-haves total)
        final score       = semantic x (0.6 + 0.4 x weighted coverage)

    Full coverage keeps the semantic score as it is; no coverage keeps 60% of it,
    so a resume missing the required skills sinks in the ranking. If the job has
    no must-have skills the semantic score is returned unchanged (old behaviour).
    """
    must = set(must_have) & set(job_skills)
    if not must:
        return semantic
    nice = set(job_skills) - must
    total = 2 * len(must) + len(nice)
    got = 2 * len(must & resume_skills) + len(nice & resume_skills)
    coverage = got / total
    # max(..., 0): a negative similarity times a penalty would wrongly move the score up.
    return round(max(semantic, 0) * (0.6 + 0.4 * coverage), 2)


def score_resume(job_description: str, resume_text: str, must_have=None) -> dict:
    """
    Score a single resume against a single job description.

    must_have: optional list of skills HR marked as required for this job.

    Returns:
        {
            "score": float,                 # final score (uses must-haves if given)
            "semantic_score": float,        # meaning similarity only
            "matched_skills": list[str],
            "missing_skills": list[str],
            "missing_must_have": list[str]  # required skills the resume lacks
        }
    """
    # With normalized embeddings, cosine similarity equals dot product.
    similarity = float(_embed_one(job_description) @ _embed_one(resume_text))
    semantic = round(similarity * 100, 2)

    jd_skills = extract_skills(job_description)
    resume_skills = extract_skills(resume_text)

    matched = sorted(jd_skills & resume_skills)
    missing = sorted(jd_skills - resume_skills)
    must = set(must_have or []) & jd_skills

    return {
        "score": combine_score(semantic, jd_skills, resume_skills, must),
        "semantic_score": semantic,
        "matched_skills": matched,
        "missing_skills": missing,
        "missing_must_have": sorted(must - resume_skills),
    }


def rank_resumes(
    job_description: str,
    resumes: list[dict],
    must_have=None,
) -> list[dict]:
    """
    Batch version (used by HR bulk screening).

    resumes = [{"id": ..., "filename": ..., "text": ...}, ...]
    must_have = optional list of required skills for the job

    Returns resumes augmented with score / semantic_score / matched_skills /
    missing_skills / missing_must_have, sorted by score descending.
    """
    if not resumes:
        return []

    jd_embedding = _embed_one(job_description)
    resume_embeddings = _encode([r["text"] for r in resumes])

    # Normalized vectors -> dot product = cosine similarity.
    similarities = resume_embeddings @ jd_embedding

    jd_skills = extract_skills(job_description)
    must = set(must_have or []) & jd_skills

    results = []
    for i, resume in enumerate(resumes):
        resume_skills = extract_skills(resume["text"])
        semantic = round(float(similarities[i]) * 100, 2)
        results.append({
            **resume,
            "score": combine_score(semantic, jd_skills, resume_skills, must),
            "semantic_score": semantic,
            "matched_skills": sorted(jd_skills & resume_skills),
            "missing_skills": sorted(jd_skills - resume_skills),
            "missing_must_have": sorted(must - resume_skills),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
