import streamlit as st
import pandas as pd
import pdfplumber
import re
import string
import time
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

#MainMenu {
visibility: hidden;
}

footer {
visibility: hidden;
}

header {
visibility: hidden;
}

.stApp {
background: linear-gradient(
135deg,
#020617 0%,
#0f172a 50%,
#111827 100%
);
color: white;
font-family: 'Segoe UI', sans-serif;
}

/* =========================================================
HERO SECTION
========================================================= */

.hero-container{
padding-top:40px;
padding-bottom:40px;
text-align:center;
}

.hero-badge{
display:inline-block;
padding:12px 24px;
background: rgba(0,229,255,0.1);
border:1px solid rgba(0,229,255,0.3);
border-radius:40px;
font-size:14px;
font-weight:600;
color:#00e5ff;
margin-bottom:25px;
backdrop-filter: blur(12px);
}

.hero-title{
font-size:70px;
font-weight:800;
line-height:1.1;
background: linear-gradient(
90deg,
#00e5ff,
#8b5cf6,
#ffffff
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:25px;
}

.hero-subtitle{
font-size:22px;
color:#cbd5e1;
max-width:950px;
margin:auto;
line-height:1.8;
margin-bottom:40px;
}

/* =========================================================
GLASS CARD
========================================================= */

.glass-card{
background: rgba(255,255,255,0.05);
border:1px solid rgba(255,255,255,0.08);
padding:28px;
border-radius:24px;
backdrop-filter: blur(14px);
box-shadow:0 0 30px rgba(0,229,255,0.08);
margin-bottom:25px;
}

/* =========================================================
METRIC CARDS
========================================================= */

.metric-card{
background: linear-gradient(
145deg,
rgba(255,255,255,0.06),
rgba(255,255,255,0.03)
);
padding:25px;
border-radius:24px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0 0 25px rgba(0,229,255,0.08);
transition:0.3s;
}

.metric-card:hover{
transform:translateY(-4px);
box-shadow:0 0 40px rgba(139,92,246,0.3);
}

.metric-title{
font-size:16px;
color:#94a3b8;
margin-bottom:12px;
font-weight:600;
}

.metric-value{
font-size:42px;
font-weight:800;
color:#00e5ff;
}

/* =========================================================
BUTTONS
========================================================= */

.stButton > button{
width:100%;
height:60px;
border:none;
border-radius:18px;
font-size:18px;
font-weight:700;
background: linear-gradient(
90deg,
#06b6d4,
#8b5cf6
);
color:white;
box-shadow:0 0 25px rgba(139,92,246,0.45);
transition:0.3s;
}

.stButton > button:hover{
transform:translateY(-3px);
box-shadow:0 0 40px rgba(0,229,255,0.45);
}

/* =========================================================
INPUTS
========================================================= */

textarea,
input{
background-color: rgba(255,255,255,0.06) !important;
color:white !important;
border-radius:16px !important;
border:1px solid rgba(255,255,255,0.08) !important;
}

.stTextArea label,
.stFileUploader label,
.stSlider label{
color:#e2e8f0 !important;
font-size:17px !important;
font-weight:600 !important;
}

/* =========================================================
DATAFRAME
========================================================= */

[data-testid="stDataFrame"]{
border-radius:20px;
overflow:hidden;
border:1px solid rgba(255,255,255,0.08);
}

/* =========================================================
SIDEBAR
========================================================= */

section[data-testid="stSidebar"]{
background: rgba(255,255,255,0.03);
border-right:1px solid rgba(255,255,255,0.08);
}

.sidebar-title{
font-size:34px;
font-weight:800;
color:#00e5ff;
margin-bottom:10px;
}

.sidebar-sub{
color:#cbd5e1;
font-size:15px;
line-height:1.8;
margin-bottom:20px;
}

/* =========================================================
WORKFLOW CARDS
========================================================= */

.workflow-card{
background: rgba(255,255,255,0.05);
padding:22px;
border-radius:22px;
text-align:center;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0 0 20px rgba(0,229,255,0.05);
font-weight:700;
font-size:18px;
}

/* =========================================================
TOP CANDIDATE CARD
========================================================= */

.candidate-card{
background: rgba(255,255,255,0.05);
padding:28px;
border-radius:24px;
border:1px solid rgba(255,255,255,0.08);
box-shadow:0 0 30px rgba(139,92,246,0.12);
margin-bottom:25px;
}

.candidate-name{
font-size:34px;
font-weight:700;
color:#00e5ff;
margin-bottom:18px;
}

.candidate-text{
font-size:18px;
line-height:2;
color:#e2e8f0;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SKILLS DATASET
# =========================================================

skills_dataset = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "data science",
    "react",
    "node.js",
    "html",
    "css",
    "javascript",
    "mongodb",
    "flask",
    "django",
    "git",
    "linux",
    "power bi",
    "tableau",
    "numpy",
    "pandas",
    "opencv",
    "scikit-learn"
]

# =========================================================
# NLP FUNCTIONS
# =========================================================

def preprocess_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\s+', ' ', text)

    return text


def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

    return text


def extract_skills(text):

    detected = []

    for skill in skills_dataset:

        if skill in text:

            detected.append(skill)

    return detected


def extract_contact_info(text):

    email = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone = re.findall(
        r"\+?\d[\d -]{8,12}\d",
        text
    )

    return email, phone


def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return round(similarity * 100, 2)


def experience_score(text):

    experience = re.findall(r"\d+\+?\s*years?", text)

    if experience:

        years = int(re.findall(r"\d+", experience[0])[0])

        if years >= 5:
            return 100

        elif years >= 3:
            return 75

        elif years >= 1:
            return 50

    return 25


def education_score(text):

    text = text.lower()

    if "phd" in text:
        return 100

    elif "mtech" in text or "master" in text:
        return 85

    elif "btech" in text or "bachelor" in text:
        return 70

    return 40


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
    🤖 SmartHire AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-sub">
    AI-powered semantic recruitment platform using NLP,
    TF-IDF vectorization, cosine similarity,
    intelligent candidate ranking and recruiter analytics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🚀 Features")

    st.markdown("""
    ✅ Resume Parsing  
    ✅ Semantic Matching  
    ✅ Candidate Ranking  
    ✅ Skill Gap Analysis  
    ✅ AI Dashboard  
    ✅ Recruiter Analytics  
    ✅ CSV Export  
    """)

# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero-container">

<div class="hero-badge">
🚀 AI-Powered Recruitment Platform
</div>

<div class="hero-title">
Smart Resume Screening <br>
Powered by AI & NLP
</div>

<div class="hero-subtitle">
Automatically analyze resumes, extract skills,
compute semantic similarity scores, rank candidates,
and shortlist top applicants using Natural Language Processing.
</div>

</div>
""", unsafe_allow_html=True)

# =========================================================
# WORKFLOW SECTION
# =========================================================

st.markdown("## ⚙️ AI Recruitment Workflow")

w1, w2, w3, w4, w5 = st.columns(5)

with w1:
    st.markdown("""
    <div class="workflow-card">
    📂<br><br>
    Upload Resume
    </div>
    """, unsafe_allow_html=True)

with w2:
    st.markdown("""
    <div class="workflow-card">
    🧠<br><br>
    NLP Processing
    </div>
    """, unsafe_allow_html=True)

with w3:
    st.markdown("""
    <div class="workflow-card">
    📊<br><br>
    Semantic Matching
    </div>
    """, unsafe_allow_html=True)

with w4:
    st.markdown("""
    <div class="workflow-card">
    🏆<br><br>
    Candidate Ranking
    </div>
    """, unsafe_allow_html=True)

with w5:
    st.markdown("""
    <div class="workflow-card">
    ✅<br><br>
    Shortlisting
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="glass-card">
""", unsafe_allow_html=True)

job_description = st.text_area(
    "📄 Paste Job Description",
    height=220,
    placeholder="Paste job description here..."
)

uploaded_files = st.file_uploader(
    "📂 Upload Candidate Resumes (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

threshold = st.slider(
    "🎯 Minimum Match Score for Shortlisting",
    0,
    100,
    70
)

analyze = st.button("🚀 Analyze Candidates")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MAIN ANALYSIS
# =========================================================

if analyze:

    if uploaded_files and job_description:

        progress = st.progress(0)

        with st.spinner("🤖 AI engine analyzing resumes..."):

            candidate_data = []

            all_skills = []

            for index, file in enumerate(uploaded_files):

                resume_text = extract_text_from_pdf(file)

                cleaned_resume = preprocess_text(resume_text)

                cleaned_job = preprocess_text(job_description)

                similarity = calculate_similarity(
                    cleaned_resume,
                    cleaned_job
                )

                skills = extract_skills(cleaned_resume)

                required_skills = extract_skills(cleaned_job)

                missing_skills = [
                    skill for skill in required_skills
                    if skill not in skills
                ]

                email, phone = extract_contact_info(
                    resume_text
                )

                exp_score = experience_score(cleaned_resume)

                edu_score = education_score(cleaned_resume)

                final_score = round(
                    (
                        similarity * 0.6 +
                        exp_score * 0.25 +
                        edu_score * 0.15
                    ),
                    2
                )

                if final_score >= threshold:

                    status = "Selected"

                elif final_score >= 50:

                    status = "Review"

                else:

                    status = "Rejected"

                candidate_data.append({

                    "Candidate": file.name,

                    "Match Score": similarity,

                    "Experience Score": exp_score,

                    "Education Score": edu_score,

                    "Final Score": final_score,

                    "Detected Skills": ", ".join(skills),

                    "Missing Skills": ", ".join(missing_skills),

                    "Email": ", ".join(email),

                    "Phone": ", ".join(phone),

                    "Status": status
                })

                all_skills.extend(skills)

                progress.progress(
                    int((index + 1) / len(uploaded_files) * 100)
                )

                time.sleep(0.4)

        # =========================================================
        # DATAFRAME
        # =========================================================

        df = pd.DataFrame(candidate_data)

        df = df.sort_values(
            by="Final Score",
            ascending=False
        )

        shortlisted_df = df[
            df["Final Score"] >= threshold
        ]

        # =========================================================
        # DASHBOARD METRICS
        # =========================================================

        st.markdown("## 📊 AI Recruitment Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    Total Resumes
                </div>

                <div class="metric-value">
                    {len(uploaded_files)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    Selected Candidates
                </div>

                <div class="metric-value">
                    {len(shortlisted_df)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    Skills Detected
                </div>

                <div class="metric-value">
                    {len(set(all_skills))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    Top Match
                </div>

                <div class="metric-value">
                    {df['Final Score'].max()}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # =========================================================
        # TOP RECOMMENDED CANDIDATES
        # =========================================================

        st.markdown("## 🏆 Top Recommended Candidates")

        top_candidates = df.head(3)

        for _, row in top_candidates.iterrows():

            st.markdown(f"""
            <div class="candidate-card">

            <div class="candidate-name">
            👤 {row['Candidate']}
            </div>

            <div class="candidate-text">

            <b>Final Score:</b> {row['Final Score']}%<br>

            <b>Match Score:</b> {row['Match Score']}%<br>

            <b>Detected Skills:</b> {row['Detected Skills']}<br>

            <b>Missing Skills:</b>
            <span style="color:#f87171;">
            {row['Missing Skills']}
            </span><br>

            <b>Status:</b> {row['Status']}

            </div>

            </div>
            """, unsafe_allow_html=True)

        # =========================================================
        # CANDIDATE TABLE
        # =========================================================

        st.markdown("## 📋 Candidate Ranking Dashboard")

        st.dataframe(
            df,
            use_container_width=True
        )

        # =========================================================
        # SHORTLISTED CANDIDATES
        # =========================================================

        st.markdown("## ✅ Shortlisted Candidates")

        st.dataframe(
            shortlisted_df,
            use_container_width=True
        )

        # =========================================================
        # CSV EXPORT
        # =========================================================

        csv = shortlisted_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Shortlisted Candidates CSV",
            csv,
            "shortlisted_candidates.csv",
            "text/csv"
        )

        # =========================================================
        # SKILL DISTRIBUTION CHART
        # =========================================================

        if all_skills:

            st.markdown("## 📈 Skill Distribution Analysis")

            skill_counts = pd.Series(
                all_skills
            ).value_counts().reset_index()

            skill_counts.columns = [
                "Skill",
                "Count"
            ]

            fig = px.bar(
                skill_counts,
                x="Skill",
                y="Count",
                title="Detected Skills Across Resumes"
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # =========================================================
        # SCORE COMPARISON CHART
        # =========================================================

        st.markdown("## 📊 Candidate Score Comparison")

        fig2 = go.Figure()

        fig2.add_trace(
            go.Bar(
                x=df["Candidate"],
                y=df["Final Score"],
                name="Final Score"
            )
        )

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =========================================================
        # SKILL GAP ANALYSIS
        # =========================================================

        st.markdown("## 🧠 Skill Gap Analysis")

        for _, row in df.iterrows():

            st.markdown(f"""
            <div class="glass-card">

            <h3 style="color:#00e5ff;">
            👤 {row['Candidate']}
            </h3>

            <p style="font-size:18px; line-height:2;">

            <b>Detected Skills:</b>
            {row['Detected Skills']}<br>

            <b>Missing Skills:</b>

            <span style="color:#f87171;">
            {row['Missing Skills']}
            </span>

            </p>

            </div>
            """, unsafe_allow_html=True)

        # =========================================================
        # AI INSIGHTS
        # =========================================================

        st.markdown("## 🧠 AI Recruitment Insights")

        top_candidate = df.iloc[0]["Candidate"]

        top_score = df.iloc[0]["Final Score"]

        st.markdown(f"""
        <div class="glass-card">

        <h2 style="color:#00e5ff;">
        🔥 Best Candidate Recommendation
        </h2>

        <p style="font-size:20px; line-height:2;">

        <b>{top_candidate}</b>
        achieved the highest ranking with a final
        AI match score of <b>{top_score}%</b>.

        <br><br>

        The system recommends prioritizing this candidate
        for interview shortlisting.

        </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.warning(
            "⚠ Please upload resumes and enter job description."
        )