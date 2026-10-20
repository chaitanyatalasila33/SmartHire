# SmartHire — Resume-to-Job Matching & Career Guidance Engine

SmartHire is an academic prototype that demonstrates classical machine learning for resume classification, content-based job recommendations, and skill-gap analysis. It uses Python, scikit-learn, TF-IDF, cosine similarity, and Streamlit.

> **Important data note:** The included resume examples and job listings are synthetic demo data. Job records are illustrative and are not live vacancies. Results demonstrate software functionality, not real-world hiring performance. Replace the sample data with appropriately licensed public datasets before making claims about generalization.

## Features

- Resume text extraction from PDF, DOCX, or TXT.
- Supervised resume-category classifier using TF-IDF + Logistic Regression.
- Content-based job ranking using TF-IDF + cosine similarity.
- Keyword-based skill-gap report for the top recommendation.
- Streamlit user interface with match scores and explanatory limitations.
- Reproducible training script, evaluation report, and unit tests.

## Project structure

```text
SmartHire/
├── app/streamlit_app.py
├── data/
│   ├── README.md
│   └── sample/                 # synthetic examples only
├── notebooks/                  # exploratory notebook(s)
├── reports/                    # report and generated evaluation
├── scripts/train.py
├── src/
│   ├── data/                   # loading and preprocessing
│   ├── features/               # TF-IDF and skill matching
│   ├── models/                 # classifier, recommender, clustering helper
│   ├── parsing/                # resume extraction
│   └── evaluate.py
├── tests/
├── requirements.txt
└── README.md
```

## Run locally

Python 3.10 or 3.11 is recommended.

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train and evaluate the classifier

Run from the repository root:

```bash
python scripts/train.py
```

This writes `models/resume_classifier.joblib` and `reports/evaluation.json`. Reported metrics are calculated from the small synthetic dataset and should not be treated as a meaningful real-world benchmark.

### 3. Run the web app

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed in the terminal. Upload a text-based PDF, DOCX, or TXT resume, or paste resume text. The app trains the demo classifier on startup so the demo can run without a previously generated model file.

### 4. Run tests

```bash
python -m pytest -q
```

## Datasets

The application uses `data/sample/sample_resumes.csv` and `data/sample/sample_jobs.csv` for demonstration. To conduct a proper experiment, obtain suitable resume-category and job-posting datasets from public sources such as Kaggle, review their licenses and usage conditions, document the source and date accessed, and map their columns into the schema used by `src/data/load_data.py`. Do not scrape LinkedIn or Naukri in violation of their terms. Do not commit private resumes or large datasets to GitHub.

Expected resume schema: `resume_text`, `category`.

Expected job schema: `title`, `company`, `location`, `skills`, `description`, and optionally `experience`.

## Methodology

1. Normalize resume/job text.
2. Transform text into TF-IDF feature vectors.
3. Train Logistic Regression to predict a resume category.
4. Compute cosine similarity between the resume vector and job-description vectors; return the highest-ranked matches.
5. Compare detected skills against the top job's listed skills to identify possible gaps.

## Limitations

- The demo dataset is tiny and synthetic; classifier metrics will not generalize to real applicants.
- A cosine-similarity score is a text similarity score, not a hiring probability.
- Skill detection uses a limited hand-maintained vocabulary and can miss synonyms, context, and proficiency.
- The project is an educational prototype, not a validated recruitment or employment decision system.
- Scanned PDFs may require OCR, which is not included.
- Listings are not live jobs and should not be used as real vacancies.

## GitHub upload

1. Create a new repository on GitHub, for example `SmartHire`.
2. Keep the repository public only if your submission instructions require a publicly accessible link.
3. From the project root, run:

```bash
git init
git add .
git commit -m "Add SmartHire ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/SmartHire.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the exact remote URL GitHub gives you. Open the repository URL in a private/incognito browser window to verify access before submitting it. Do not commit virtual environments, credentials, personal resumes, or licensed datasets unless redistribution is explicitly permitted.

## Responsible use

This prototype can assist learning and exploration but should not be used as the sole basis for employment decisions. Human review and representative validation are necessary before any real-world use.
