import streamlit as st
import pandas as pd
import pdfplumber
import nltk
import re
import string
import time
import plotly.express as px

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.authentication import login, logout


# ---------- PAGE CONFIG ----------

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)


# ---------- LOAD CSS ----------

with open("assets/styles.css") as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# ---------- NLTK ----------

nltk.download("stopwords")

stop_words = stopwords.words("english")


# ---------- SESSION ----------

if "logged_in" not in st.session_state:

    st.session_state.logged_in = False


# ---------- LOGIN CHECK ----------

if not st.session_state.logged_in:

    login()

    st.stop()


# ---------- LOGOUT ----------

logout()


# ---------- LOAD SKILLS DATASET ----------

skills_df = pd.read_csv("skills_dataset.csv")

skills_list = skills_df["Skill"].dropna().tolist()


# ---------- SIDEBAR ----------

st.sidebar.title("🤖 AI Recruitment Panel")

st.sidebar.markdown("---")

menu = st.sidebar.radio(

    "Navigation",

    [

        "Dashboard",

        "Analytics",

        "About Project"

    ]
)


st.sidebar.markdown("---")

st.sidebar.info(

    """
    AI Resume Screening System
    
    Built using:
    
    • Python  
    • NLP  
    • TF-IDF  
    • Cosine Similarity  
    • Streamlit
    """
)


# ---------- FUNCTIONS ----------

def preprocess_text(text):

    text = text.lower()

    text = text.translate(

        str.maketrans(

            '',

            '',

            string.punctuation

        )
    )

    words = text.split()

    words = [

        word

        for word in words

        if word not in stop_words
    ]

    return " ".join(words)


def extract_text_from_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text

    return text


def extract_skills(text):

    found_skills = []

    text = text.lower()

    for skill in skills_list:

        if skill.lower() in text:

            found_skills.append(skill)

    return list(set(found_skills))


def extract_contact_details(text):

    email = re.findall(

        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",

        text
    )

    phone = re.findall(

        r"\+?\d[\d -]{8,12}\d",

        text
    )

    return email, phone


def calculate_similarity(

    resume_text,

    job_description

):

    documents = [

        resume_text,

        job_description

    ]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(

        tfidf_matrix[0:1],

        tfidf_matrix[1:2]

    )[0][0]

    return round(similarity * 100, 2)


def calculate_skill_score(

    resume_skills,

    jd_skills

):

    if len(jd_skills) == 0:

        return 0

    matched = len(

        set(resume_skills)

        &

        set(jd_skills)
    )

    return round(

        (matched / len(jd_skills)) * 100,

        2
    )


def experience_score(text):

    exp = re.findall(

        r"\d+\+?\s*years?",

        text.lower()
    )

    if exp:

        years = int(

            re.findall(

                r"\d+",

                exp[0]
            )[0]
        )

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

        return 80

    elif "btech" in text or "bachelor" in text:

        return 60

    return 30


def find_missing_skills(

    resume_skills,

    jd_skills

):

    missing = list(

        set(jd_skills)

        -

        set(resume_skills)
    )

    return missing


# ---------- MAIN DASHBOARD ----------

if menu == "Dashboard":

    st.markdown(

        """
        <div class="login-container">
        
            <h1 class="login-title">
                🤖 AI Resume Screening System
            </h1>
        
            <p class="login-subtitle">
                Semantic AI-Powered Candidate Evaluation Platform
            </p>
        
        </div>
        """,

        unsafe_allow_html=True
    )


    st.markdown("## 📄 Paste Job Description")

    job_description = st.text_area(

        "",

        height=220,

        placeholder="""
        Example:
        
        We are looking for an AI Engineer with skills in:
        
        Python
        Machine Learning
        Deep Learning
        NLP
        SQL
        Docker
        TensorFlow
        """
    )


    st.markdown("## 📂 Upload Candidate Resumes")

    uploaded_files = st.file_uploader(

        "",

        type=["pdf"],

        accept_multiple_files=True
    )


    threshold = st.slider(

        "🎯 Minimum Match Score",

        0,

        100,

        60
    )


    analyze = st.button(

        "🚀 Analyze Candidates"
    )


    if analyze:

        if uploaded_files and job_description:

            with st.spinner(

                "AI Engine Processing Resumes..."
            ):

                time.sleep(2)

            candidate_data = []

            all_skills = []

            processed_jd = preprocess_text(

                job_description
            )

            jd_skills = extract_skills(

                processed_jd
            )

            for file in uploaded_files:

                raw_text = extract_text_from_pdf(file)

                clean_text = preprocess_text(

                    raw_text
                )

                skills = extract_skills(

                    clean_text
                )

                email, phone = extract_contact_details(

                    raw_text
                )

                similarity_score = calculate_similarity(

                    clean_text,

                    processed_jd
                )

                skill_score = calculate_skill_score(

                    skills,

                    jd_skills
                )

                exp_score = experience_score(

                    clean_text
                )

                edu_score = education_score(

                    clean_text
                )

                final_score = round(

                    (

                        0.5 * similarity_score +

                        0.2 * skill_score +

                        0.2 * exp_score +

                        0.1 * edu_score

                    ),

                    2
                )

                missing_skills = find_missing_skills(

                    skills,

                    jd_skills
                )

                candidate_data.append({

                    "Candidate": file.name,

                    "Match Score": similarity_score,

                    "Skill Score": skill_score,

                    "Experience Score": exp_score,

                    "Education Score": edu_score,

                    "Final Score": final_score,

                    "Email": ", ".join(email),

                    "Phone": ", ".join(phone),

                    "Skills": ", ".join(skills),

                    "Missing Skills": ", ".join(missing_skills)
                })

                all_skills.extend(skills)

            df = pd.DataFrame(candidate_data)

            df = df.sort_values(

                by="Final Score",

                ascending=False
            )

            selected_candidates = df[

                df["Final Score"] >= threshold

            ]


            # ---------- METRICS ----------

            st.markdown("## 📊 Recruiter Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(

                "📄 Total Resumes",

                len(uploaded_files)
            )

            col2.metric(

                "🎯 Top Score",

                round(df["Final Score"].max(), 2)
            )

            col3.metric(

                "✅ Selected",

                len(selected_candidates)
            )

            col4.metric(

                "🧠 Skills Detected",

                len(set(all_skills))
            )


            # ---------- TOP CANDIDATES ----------

            st.markdown("## 🏆 Top Recommended Candidates")

            st.dataframe(

                df[

                    [

                        "Candidate",

                        "Final Score",

                        "Match Score",

                        "Skills"

                    ]

                ].head(5),

                use_container_width=True
            )


            # ---------- COMPLETE TABLE ----------

            st.markdown("## 📋 Complete Candidate Analysis")

            st.dataframe(

                df,

                use_container_width=True
            )


            # ---------- SELECTED CANDIDATES ----------

            st.markdown("## ✅ Selected Candidates")

            st.dataframe(

                selected_candidates,

                use_container_width=True
            )


            # ---------- DOWNLOAD CSV ----------

            csv = selected_candidates.to_csv(

                index=False

            ).encode("utf-8")

            st.download_button(

                "📥 Download Selected Candidates CSV",

                csv,

                "selected_candidates.csv",

                "text/csv"
            )


            # ---------- SKILL ANALYTICS ----------

            st.markdown("## 📈 Skill Distribution")

            if all_skills:

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

                    title="Most Common Skills",

                    template="plotly_dark"
                )

                st.plotly_chart(

                    fig,

                    use_container_width=True
                )

            # ---------- SKILL GAP ANALYTICS ----------

            st.markdown("## 🚨 Skill Gap Analytics")

            all_missing_skills = []

            for item in candidate_data:
                missing = item["Missing Skills"]
                if missing:
                    split_skills = [
                        skill.strip()
                        for skill in missing.split(",")
                        if skill.strip()
                    ]
                    all_missing_skills.extend(split_skills)

            if all_missing_skills:
                missing_df = pd.Series(
                    all_missing_skills
                ).value_counts().reset_index()
                missing_df.columns = [
                    "Missing Skill",
                    "Count"
                ]
                fig_gap = px.bar(
                    missing_df,
                    x="Missing Skill",
                    y="Count",
                    title="Most Missing Skills Among Candidates",
                    template="plotly_dark"
                )
                st.plotly_chart(
                    fig_gap,
                    use_container_width=True
                )
            else:
                st.success(
                    "No major skill gaps detected."
                )
                
            # ---------- RECRUITER INSIGHTS ----------    
            st.markdown("## 🧠 AI Recruitment Insights")

            if len(all_missing_skills) > 0:
                top_gap = pd.Series(
                    all_missing_skills
                ).value_counts().idxmax()
                st.warning(
                    f"""
                    Most candidates are missing:
        
                    🔥 {top_gap}
        
                    Recruiters should focus on hiring candidates
                    with stronger expertise in this skill.
                    """
                )
            else:
                st.success(
                    """
                    Candidates match most required skills.
                    """
                )

            # ---------- TOP SKILLS TABLE ----------

            st.markdown("## 📋 Most In-Demand Skills")

            if all_skills:
                top_skills_df = pd.Series(
                    all_skills
                ).value_counts().reset_index()
                top_skills_df.columns = [
                    "Skill",
                    "Frequency"
                ]
                st.dataframe(
                    top_skills_df,
                    use_container_width=True
                )


            # ---------- CANDIDATE GAP TABLE ----------

            st.markdown("## ⚠️ Candidate Skill Gap Report")

            gap_report = []

            for item in candidate_data:

                gap_report.append({

                    "Candidate": item["Candidate"],

                    "Missing Skills": item["Missing Skills"],

                    "Final Score": item["Final Score"]

                })

            gap_df = pd.DataFrame(gap_report)

            st.dataframe(

                gap_df,

                use_container_width=True
            )
            # ---------- SCORE ANALYTICS ----------

            st.markdown("## 📊 Candidate Score Visualization")

            fig2 = px.bar(

                df,

                x="Candidate",

                y="Final Score",

                color="Final Score",

                title="Candidate Ranking",

                template="plotly_dark"
            )

            st.plotly_chart(

                fig2,

                use_container_width=True
            )


            # ---------- PIE CHART ----------

            st.markdown("## 🥧 Candidate Selection Ratio")

            selection_df = pd.DataFrame({

                "Category": [

                    "Selected",

                    "Rejected"
                ],

                "Count": [

                    len(selected_candidates),

                    len(df) - len(selected_candidates)
                ]
            })

            fig3 = px.pie(

                selection_df,

                names="Category",

                values="Count",

                title="Selection Distribution",

                template="plotly_dark"
            )

            st.plotly_chart(

                fig3,

                use_container_width=True
            )

        else:

            st.warning(

                "Please upload resumes and enter job description."
            )


# ---------- ANALYTICS PAGE ----------

elif menu == "Analytics":

    st.markdown(

        """
        <div class="login-container">
        
            <h1 class="login-title">
                📈 Recruitment Analytics
            </h1>
        
            <p class="login-subtitle">
                AI-driven hiring insights dashboard
            </p>
        
        </div>
        """,

        unsafe_allow_html=True
    )

    st.info(

        """
        This section displays:
        
        • Candidate trends  
        • Recruitment insights  
        • Skill demand analysis  
        • Selection statistics  
        """
    )

    st.image(

        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",

        use_container_width=True
    )


# ---------- ABOUT PAGE ----------

elif menu == "About Project":

    st.markdown(

        """
        <div class="login-container">
        
            <h1 class="login-title">
                🤖 About This Project
            </h1>
        
            <p class="login-subtitle">
                AI-Powered Semantic Resume Screening System
            </p>
        
        </div>
        """,

        unsafe_allow_html=True
    )

    st.markdown(

        """
        ## 🚀 Project Features
        
        - Resume PDF Parsing  
        - TF-IDF Vectorization  
        - Cosine Similarity Matching  
        - Skill Extraction  
        - Skill Gap Analysis  
        - Candidate Ranking  
        - Recruiter Dashboard  
        - CSV Export System  
        - Interactive Analytics  
        
        
        ## 🧠 Technologies Used
        
        - Python  
        - Streamlit  
        - NLP  
        - Scikit-learn  
        - TF-IDF  
        - Cosine Similarity  
        - Pandas  
        - Plotly  
        
        
        ## 🔮 Future Scope
        
        - BERT Semantic Matching  
        - AI Interview Generation  
        - Voice-Based Candidate Screening  
        - Cloud Database Integration  
        - React Frontend  
        - FastAPI Backend  
        """
    )