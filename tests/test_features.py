import pandas as pd
from src.data.preprocess import clean_text, validate_jobs
from src.features.match_features import extract_skills, skill_gap
from src.models.recommender import recommend_jobs


def test_clean_text_normalizes_whitespace():
    assert clean_text("  Python!!   SQL ") == "python sql"


def test_skill_gap_detects_missing_skills():
    result = skill_gap("I use Python and SQL", "Python, SQL, React")
    assert "python" in result["matched"]
    assert "sql" in result["matched"]
    assert "react" in result["missing"]
    assert result["coverage"] == 0.667


def test_extract_skills_does_not_match_substrings():
    assert "java" not in extract_skills("javascript developer")


def test_validate_jobs_requires_columns():
    try:
        validate_jobs(pd.DataFrame({"title": ["Developer"]}))
    except ValueError as exc:
        assert "skills" in str(exc)
    else:
        raise AssertionError("Expected missing-column validation error")


def test_recommender_ranks_relevant_job_first():
    jobs = pd.DataFrame([
        {"title": "Frontend Developer", "skills": "HTML CSS React", "description": "Build websites"},
        {"title": "Data Analyst", "skills": "Python SQL pandas", "description": "Analyze data"},
    ])
    ranked = recommend_jobs("Python SQL pandas data analysis", jobs, top_n=2)
    assert ranked.iloc[0]["title"] == "Data Analyst"
    assert ranked.iloc[0]["match_score"] >= ranked.iloc[1]["match_score"]


def test_empty_resume_returns_empty_results():
    jobs = pd.DataFrame([{"title": "A", "skills": "Python", "description": "Data"}])
    assert recommend_jobs("", jobs).empty
