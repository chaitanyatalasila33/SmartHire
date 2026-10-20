import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from src.data.load_data import load_jobs, load_sample_resumes
from src.models.classifier import build_classifier, predict_category
from src.models.recommender import recommend_jobs
from src.features.match_features import skill_gap
from src.parsing.resume_parser import extract_resume_text

st.set_page_config(page_title="SmartHire | Career Guidance", page_icon="🎯", layout="wide")
st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#11294b,#2459a9); color: white; margin-bottom: 1.2rem;}
.hero h1 {color: white; margin-bottom: .2rem;}
.small-note {color: #5b6574; font-size: .9rem;}
</style>
<div class="hero"><h1>🎯 SmartHire</h1><p>Resume-to-job matching and career guidance using classical machine learning</p></div>
""", unsafe_allow_html=True)
st.caption("Academic prototype • Uses TF-IDF, cosine similarity, and a supervised text classifier. Demo job listings are illustrative, not live vacancies.")

@st.cache_data
def get_jobs():
    return load_jobs()

@st.cache_data
def get_training_data():
    return load_sample_resumes()

@st.cache_resource
def get_classifier():
    df = get_training_data()
    model = build_classifier()
    model.fit(df["resume_text"].astype(str), df["category"].astype(str).str.strip())
    return model

with st.sidebar:
    st.header("How it works")
    st.write("1. Upload or paste resume text")
    st.write("2. Classify the likely job domain")
    st.write("3. Rank jobs by text similarity")
    st.write("4. Compare skills and find gaps")
    st.divider()
    st.info("For a real deployment, replace synthetic sample records with properly licensed datasets and evaluate on a representative held-out dataset.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Upload your resume")
    uploaded = st.file_uploader("Choose PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
with right:
    st.subheader("Or paste resume text")
    pasted = st.text_area("Resume text", height=170, placeholder="Example: Python, SQL, pandas, machine learning...")

resume_text = pasted.strip()
if uploaded is not None:
    try:
        uploaded_text = extract_resume_text(uploaded.name, uploaded.getvalue())
        if not resume_text:
            resume_text = uploaded_text
        else:
            st.warning("Both a file and pasted text were provided; using the pasted text.")
    except Exception as exc:
        st.error(str(exc))

col_a, col_b = st.columns([1, 3])
with col_a:
    top_n = st.selectbox("Jobs to show", [3, 5, 8], index=1)
with col_b:
    st.write("")
    run = st.button("Analyze resume", type="primary", use_container_width=True)

if run:
    if not resume_text:
        st.warning("Upload a readable resume or paste some resume text first.")
    else:
        model = get_classifier()
        category, confidence = predict_category(model, resume_text)
        st.divider()
        st.subheader("Your analysis")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted domain", category)
        m2.metric("Classifier confidence", f"{confidence:.0%}")
        jobs = recommend_jobs(resume_text, get_jobs(), top_n=top_n)
        m3.metric("Jobs compared", len(get_jobs()))
        st.caption("Confidence is the classifier's estimated probability on a tiny synthetic demo dataset; it is not a hiring probability.")

        st.subheader("Recommended jobs")
        if jobs.empty:
            st.info("No recommendations could be generated from the supplied text.")
        else:
            for _, job in jobs.iterrows():
                title = str(job.get("title", "Job role"))
                company = str(job.get("company", "Company not listed"))
                location = str(job.get("location", "Location not listed"))
                score = float(job.get("match_score", 0))
                with st.container(border=True):
                    h1, h2 = st.columns([4, 1])
                    h1.markdown(f"**{title}**  \n{company} · {location}")
                    h2.metric("Text match", f"{score:.1f}%")
                    st.write("**Required skills:**", str(job.get("skills", "Not listed")))
                    st.write(str(job.get("description", "")))
                    st.caption("Match score is cosine similarity, not the probability of being hired. Listings are sample records.")

        st.subheader("Skill-gap report")
        all_skills = sorted({s.strip() for skills in get_jobs()["skills"].astype(str) for s in skills.split(",") if s.strip()})
        # Use the top recommendation's requirements as the target role's skill profile.
        target = jobs.iloc[0] if not jobs.empty else None
        required = str(target.get("skills", "")) if target is not None else ", ".join(all_skills)
        gap = skill_gap(resume_text, required)
        s1, s2, s3 = st.columns(3)
        s1.metric("Skill coverage for top match", f"{gap['coverage']:.0%}")
        s2.metric("Matched skills", len(gap["matched"]))
        s3.metric("Skills to build", len(gap["missing"]))
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Skills found**")
            st.write(", ".join(gap["matched"]) if gap["matched"] else "No listed skills detected yet.")
        with g2:
            st.markdown("**Potential skill gaps**")
            st.write(", ".join(gap["missing"]) if gap["missing"] else "No gaps detected from the current keyword list.")
        if gap["missing"]:
            st.markdown("**Suggested next steps**")
            for skill in gap["missing"]:
                st.write(f"- Practise **{skill}** through a small project or course, then add it to your resume only when you have the skill.")
        st.warning("This is an educational prototype. It must not be used as the sole basis for hiring decisions. Skill detection uses a limited keyword list and can miss synonyms or context.")

with st.expander("About the demo data and methodology"):
    st.write("The included resumes and job listings are synthetic examples for reproducible demonstration. The classifier is trained on those examples when the app starts. The recommender uses TF-IDF vectors and cosine similarity. No live job sites are scraped.")
