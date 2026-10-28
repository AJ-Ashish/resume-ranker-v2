"""Lightweight resume ranking engine for low-memory hosting.

Uses token-based cosine similarity plus explicit skill matching. This avoids
loading PyTorch/Sentence Transformers into the web-service process, which can
exceed the RAM limit on small free hosting instances.
"""
from collections import Counter
import math
import re

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


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "our", "that",
    "the", "their", "this", "to", "was", "were", "will", "with", "you",
    "your", "we", "they", "he", "she", "i", "me", "my", "not", "but",
}


def _tokens(text: str) -> list[str]:
    """Normalize text into useful word tokens without external ML libraries."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", (text or "").lower())
    return [w.strip(".-") for w in words if len(w.strip(".-")) > 1 and w not in _STOP_WORDS]


def _embed(text: str) -> Counter:
    return Counter(_tokens(text))


def _cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity over token counts; returns a value from 0 to 1."""
    a, b = _embed(text_a), _embed(text_b)
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(token, 0) for token, count in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if not norm_a or not norm_b:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def extract_skills(text: str) -> set:
    text_lower = (text or "").lower()
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


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
    """Score one resume using lightweight text similarity and skill coverage."""
    semantic = round(_cosine_similarity(job_description, resume_text) * 100, 2)
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


def rank_resumes(job_description: str, resumes: list[dict], must_have=None) -> list[dict]:
    """Rank a batch of resumes without loading heavyweight ML libraries."""
    if not resumes:
        return []
    jd_skills = extract_skills(job_description)
    must = set(must_have or []) & jd_skills
    results = []
    for resume in resumes:
        resume_text = resume.get("text", "") or ""
        resume_skills = extract_skills(resume_text)
        semantic = round(_cosine_similarity(job_description, resume_text) * 100, 2)
        results.append({
            **resume,
            "score": combine_score(semantic, jd_skills, resume_skills, must),
            "semantic_score": semantic,
            "matched_skills": sorted(jd_skills & resume_skills),
            "missing_skills": sorted(jd_skills - resume_skills),
            "missing_must_have": sorted(must - resume_skills),
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results
