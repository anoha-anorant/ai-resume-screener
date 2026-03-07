import streamlit as st
import nltk
import string
import pdfplumber
import re
import pandas as pd
import matplotlib.pyplot as plt
import time

from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = stopwords.words("english")

skills_list = [
"python","java","c++","machine learning","deep learning",
"sql","pandas","numpy","data analysis","nlp","tensorflow",
"pytorch","html","css","javascript","node.js","react",
"flask","django","docker","kubernetes","git"
]


# ---------- UI STYLE ----------

st.markdown("""
<style>

.stApp {
background: radial-gradient(circle at top,#0f172a,#020617);
color:white;
font-family: 'Segoe UI', sans-serif;
}

.main-card{
background: rgba(255,255,255,0.05);
padding:30px;
border-radius:20px;
box-shadow:0 0 40px rgba(0,255,255,0.15);
backdrop-filter: blur(12px);
margin-top:40px;
}

.glow-title{
text-align:center;
font-size:40px;
font-weight:700;
color:#00e5ff;
text-shadow:0 0 15px #00e5ff;
}

button[kind="primary"]{
background: linear-gradient(90deg,#6366f1,#8b5cf6);
border:none;
border-radius:12px;
height:50px;
font-size:18px;
box-shadow:0 0 20px rgba(99,102,241,0.6);
}

</style>
""", unsafe_allow_html=True)


# ---------- FUNCTIONS ----------

def preprocess(text):

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    words = [w for w in words if w not in stop_words]

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

    for skill in skills_list:

        if skill in text:

            found_skills.append(skill)

    return found_skills


def extract_contact(text):

    email = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone = re.findall(
        r"\+?\d[\d -]{8,12}\d",
        text
    )

    return email, phone


def experience_score(text):

    exp = re.findall(r"\d+\+?\s*years?", text)

    if exp:

        years = int(re.findall(r"\d+", exp[0])[0])

        if years >= 5:
            return 1
        elif years >= 3:
            return 0.7
        elif years >= 1:
            return 0.4

    return 0.2


def education_score(text):

    text = text.lower()

    if "phd" in text:
        return 1

    elif "mtech" in text or "master" in text:
        return 0.8

    elif "btech" in text or "bachelor" in text:
        return 0.6

    else:
        return 0.3


def skill_match_score(resume_text, job_desc):

    resume_skills = extract_skills(resume_text)

    jd_skills = extract_skills(job_desc)

    if len(jd_skills) == 0:
        return 0

    match = len(set(resume_skills) & set(jd_skills))

    return match / len(jd_skills)


# ---------- APP UI ----------

st.markdown('<p class="glow-title">🤖 AI Resume Screening Engine</p>', unsafe_allow_html=True)

st.markdown('<div class="main-card">', unsafe_allow_html=True)


job_desc = st.text_area("📄 Paste Job Description")


uploaded_files = st.file_uploader(
"📂 Upload Resumes (PDF)",
type=["pdf"],
accept_multiple_files=True
)


if st.button("🚀 Analyze Candidates"):


    if uploaded_files and job_desc:


        with st.spinner("AI engine analyzing resumes..."):

            time.sleep(2)


        candidate_data = []

        all_skills = []


        for file in uploaded_files:


            text = extract_text_from_pdf(file)


            skills = extract_skills(text.lower())


            email, phone = extract_contact(text)


            skill_score = skill_match_score(text.lower(), job_desc.lower())


            exp_score = experience_score(text.lower())


            edu_score = education_score(text.lower())


            final_score = (

                0.5 * skill_score +

                0.3 * exp_score +

                0.2 * edu_score

            )


            candidate_data.append({

                "Candidate": file.name,

                "Skill Score": round(skill_score*100,2),

                "Experience Score": round(exp_score*100,2),

                "Education Score": round(edu_score*100,2),

                "Final Score": round(final_score*100,2),

                "Email": ", ".join(email),

                "Phone": ", ".join(phone)

            })


            all_skills.extend(skills)


        df = pd.DataFrame(candidate_data)


        df = df.sort_values(by="Final Score", ascending=False)


        col1,col2,col3 = st.columns(3)

        col1.metric("Total Resumes", len(uploaded_files))

        col2.metric("Detected Skills", len(all_skills))

        col3.metric("Top Score", df["Final Score"].max())


        st.subheader("🏆 AI Recommended Candidates")

        st.write(df.head(3)[["Candidate","Final Score"]])


        st.subheader("📊 Recruiter Dashboard")

        st.dataframe(df)


        if all_skills:

            st.subheader("📈 Skill Distribution")


            skill_counts = pd.Series(all_skills).value_counts()


            fig, ax = plt.subplots()


            skill_counts.plot(kind="bar", ax=ax)


            st.pyplot(fig)


        st.subheader("📥 Download Ranking Report")


        csv = df.to_csv(index=False).encode("utf-8")


        st.download_button(

            "Download CSV",

            csv,

            "candidate_ranking.csv",

            "text/csv"

        )


    else:

        st.warning("Please upload resumes and enter job description.")


st.markdown('</div>', unsafe_allow_html=True)