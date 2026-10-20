import re
from typing import Iterable

# Lightweight illustrative skill vocabulary. Extend it when using a larger dataset.
SKILLS = {
    "python", "java", "c++", "sql", "html", "css", "javascript", "react", "spring boot",
    "rest api", "rest apis", "pandas", "numpy", "scikit-learn", "machine learning", "statistics",
    "excel", "power bi", "data visualization", "dashboard", "linux", "networking", "cryptography",
    "git", "data structures", "algorithms", "model evaluation", "responsive design", "security monitoring",
}


def extract_skills(text: str, vocabulary: Iterable[str] = SKILLS) -> set[str]:
    normalized = str(text or "").lower()
    found = set()
    for skill in vocabulary:
        # Word boundaries avoid matching e.g. 'java' inside unrelated longer words.
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.add(skill)
    return found


def skill_gap(resume_text: str, required_skills: str) -> dict:
    candidate = extract_skills(resume_text)
    required = extract_skills(required_skills)
    matched = sorted(candidate & required)
    missing = sorted(required - candidate)
    return {"matched": matched, "missing": missing, "required_count": len(required),
            "coverage": round(len(matched) / len(required), 3) if required else 0.0}
