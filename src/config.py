from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT_DIR / "models"
RESUME_DATA_PATH = SAMPLE_DATA_DIR / "sample_resumes.csv"
JOBS_DATA_PATH = SAMPLE_DATA_DIR / "sample_jobs.csv"
