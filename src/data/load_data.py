import pandas as pd
from src.config import RESUME_DATA_PATH, JOBS_DATA_PATH


def load_sample_resumes() -> pd.DataFrame:
    """Load clearly labelled synthetic resume examples."""
    return pd.read_csv(RESUME_DATA_PATH).fillna("")


def load_jobs() -> pd.DataFrame:
    """Load the local job corpus; sample rows are illustrative, not live vacancies."""
    return pd.read_csv(JOBS_DATA_PATH).fillna("")
