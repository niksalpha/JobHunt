import streamlit as st
import pandas as pd
import json
import time
from io import BytesIO
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pypdf import PdfReader
from fpdf import FPDF
from jobspy import scrape_jobs  # Real-time network scraping engine


# --- CUSTOM RESUME TEXT EXTRACTOR ---
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


# --- LLM PROFILE GENERATOR ---
def convert_resume_to_profile(raw_text, api_key):
    client = genai.Client(api_key=api_key)
    system_instruction = "Extract unstructured data from a candidate's resume and format it into clean JSON matching the baseline architecture."

    prompt = f"""
    Analyze the raw resume text below and structure it exactly into the specified JSON architecture.
    RAW RESUME TEXT:\n{raw_text}

    TARGET JSON ARCHITECTURE:
    {{
      "contact_info": {{"name": "Full Name", "email": "email@domain.com"}},
      "technical_skills": {{"languages": ["Python", "SQL"], "frameworks_and_tools": ["Django"]}},
      "master_experience": [
        {{"company": "Company A", "role": "Engineer", "bullet_points": ["Achieved X using Y"]}}
      ]
    }}
    Return ONLY valid raw JSON matching this format. No markdown blocks.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    return json.loads(response.text)


# --- ATS TAILORED RESUME & SCORE SCHEMA ---
class TailoredResumeResponse(BaseModel):
    ats_score: int = Field(...,
                           description="Estimated ATS Match score from 0 to 100 based on keyword and skill coverage.")
    score_breakdown: str = Field(...,
                                 description="Short explanation of why this score was given and target keywords matched.")
    tailored_resume_text: str = Field(...,
                                      description="The complete tailored resume formatted cleanly with clear section headings.")


def generate_tailored_resume_and_ats_score(candidate_profile, job_title, job_company, job_description, ats_keywords,
                                           api_key):
    """
    Generates a tailored resume alongside an ATS optimization score using Gemini structured outputs.
    """
    client = genai.Client(api_key=api_key)

    prompt = f"""
    You are an expert ATS Resume Optimization Specialist.

    CANDIDATE MASTER PROFILE:
    {json.dumps(candidate_profile, indent=2)}

    TARGET JOB DETAILS:
    Title: {job_title}
    Company: {job_company}
    Description: {job_description}
    Target ATS Keywords to Emphasize: {", ".join(ats_keywords)}

    INSTRUCTIONS:
    1. Rewrite the candidate's resume tailored specifically to this target job description.
    2. Naturally incorporate the target ATS keywords into experience bullet points without keyword stuffing.
    3. Evaluate the tailored resume against the job description and assign an ATS Match Score (0-100%).
    4. Provide a brief 2-sentence score breakdown explaining keyword match coverage.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TailoredResumeResponse,
            temperature=0.2
        )
    )
    return json.loads(response.text)


# --- PDF GENERATOR ENGINE ---
def create_pdf_from_text(text_content):
    """
    Converts resume text into a clean PDF byte stream.
    Handles blank lines and spacing safely to prevent FPDFException.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=10)

    # Replace non-latin characters that might crash standard PDF encoding
    clean_text = text_content.encode("latin-1", "replace").decode("latin-1")

    for line in clean_text.split("\n"):
        line = line.strip()

        # Handle blank lines safely by adding vertical spacing instead of multi_cell
        if not line:
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.set_font("Arial", 'B', 14)
            pdf.multi_cell(0, 8, line.replace("# ", ""))
            pdf.set_font("Arial", size=10)
        elif line.startswith("## "):
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 6, line.replace("## ", ""))
            pdf.set_font("Arial", size=10)
        elif line.startswith("### "):
            pdf.set_font("Arial", 'B', 11)
            pdf.multi_cell(0, 6, line.replace("### ", ""))
            pdf.set_font("Arial", size=10)
        else:
            pdf.multi_cell(0, 5, line)

    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output


# --- PYDANTIC SCHEMAS ---
class JobAssessment(BaseModel):
    match_score: int = Field(..., description="Fit score from 1 to 10.")
    reasoning: str = Field(..., description="A short 2-sentence match reasoning.")
    target_ats_keywords: list[str] = Field(..., description="5-8 technical keywords.")


# --- STREAMLIT UI DESIGN ---
st.set_page_config(page_title="Autonomous Job Triage Engine", layout="wide")
st.title("🎯 Autonomous Job Triage Hub")
st.write(
    "Upload your resume, set your targets, and let the agent hunt something, score, evaluate, and generate tailored ATS resumes instantly. Don't worry, we store no personal information")

# --- SIDEBAR: USER AUTHENTICATION & INPUTS ---
with st.sidebar:
    st.header("🔑 Authentication & Control")
    user_api_key = st.text_input("Gemini API Key", type="password",
                                 help="Input your free Gemini API Key to power the evaluations.")
    st.sidebar.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")

    st.header("📋 Search Parameters")
    target_roles = st.text_input("Target Roles (Comma separated)", value="")
    target_location = st.text_input("Target Location", value="Hyderabad, Telangana")
    is_remote = st.checkbox("Strictly Remote Positions Only", value=False)
    min_score = st.slider("Minimum Acceptable Match Score", min_value=1, max_value=10, value=7)

    st.header("🌐 Platform Target Selection")
    choose_linkedin = st.checkbox("LinkedIn", value=True)
    choose_indeed = st.checkbox("Indeed", value=True)

    st.header("📄 Candidate Profile Source")
    uploaded_resume = st.file_uploader("Upload Master Resume (PDF)", type=["pdf"])

# Store active session profiles and results
if "dynamic_profile" not in st.session_state:
    st.session_state.dynamic_profile = None

st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(45deg, #ff4b4b, #ff8c42);
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
    transition: transform 0.2s;
}
div.stButton > button:first-child:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# --- Button ---
if st.button("🚀 Deploy Job Hunt - Hit me!!"):
    st.toast("Launching the Job Hunt Engine....", icon="🚀")
    if not user_api_key:
        st.error("Please provide a valid Gemini API Key to run the triage matrix.")
        st.stop()
    if not uploaded_resume:
        st.error("Please upload a PDF resume to initialize target mapping.")
        st.stop()

    # Phase 1: Dynamic Profile Creation from Uploaded PDF
    with st.status("🧠 Lemme analyze and parse incoming resume, hang on!...") as status:
        raw_resume_text = extract_text_from_pdf(uploaded_resume)
        dynamic_profile = convert_resume_to_profile(raw_resume_text, user_api_key)

        role_list = [r.strip() for r in target_roles.split(",") if r.strip()]

        dynamic_profile["target_preferences"] = {
            "roles": role_list,
            "location": target_location,
            "min_fit_score": min_score
        }
        st.session_state.dynamic_profile = dynamic_profile
        status.update(label="✅ Live Resume Profile Compiled In-Memory!", state="complete")

    # Phase 2: Real-time Live Network Scraper Pull via JobSpy
    platforms = []
    if choose_linkedin: platforms.append("linkedin")
    if choose_indeed: platforms.append("indeed")

    if not platforms:
        st.error("Please select at least one search platform in the sidebar options.")
        st.stop()

    with st.status("🔎 Deploying Network Scrapers across live target platforms...") as status:
        try:
            primary_query = role_list[0] if role_list else "Software Engineer"

            jobs_df = scrape_jobs(
                site_name=platforms,
                search_term=primary_query,
                location=target_location,
                results_wanted=10,
                hours_old=72,
                is_remote=is_remote,
                linkedin_fetch_description=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )

            total_found = len(jobs_df)
            status.update(label=f"✅ Discovered {total_found} live active postings online!", state="complete")
        except Exception as e:
            status.update(label="⚠️ Scraping network run failed or timed out.", state="error")
            st.error(f"Network error details: {e}")
            st.stop()

    if total_found == 0:
        st.info(
            "The live scraper returned 0 jobs for those specific keyword parameters. Try adjusting your location or role filters.")
        st.stop()

    # Phase 3: Live LLM Match Evaluation
    st.subheader("📊 Live Match Evaluation Stream")
    client = genai.Client(api_key=user_api_key)
    high_fit_matches = []

    for idx, row in jobs_df.iterrows():
        job_description = row.get('description', '')
        if not job_description or pd.isna(job_description):
            continue

        job_title = row.get('title', 'Unknown Title')
        job_company = row.get('company', 'Unknown Company')

        with st.spinner(f"Assessing position: {job_title} at {job_company}..."):
            prompt = f"PROFILE:\n{json.dumps(dynamic_profile)}\n\nJOB:\nTitle: {job_title}\nDescription: {job_description}"
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="Unbiasedly score the job against the profile (1-10) and return JSON.",
                        response_mime_type="application/json",
                        response_schema=JobAssessment,
                        temperature=0.2
                    )
                )
                assessment = json.loads(response.text)

                if assessment['match_score'] >= min_score:
                    row_dict = row.to_dict()
                    row_dict['match_score'] = assessment['match_score']
                    row_dict['reasoning'] = assessment['reasoning']
                    row_dict['ats_keywords'] = assessment.get('target_ats_keywords', [])
                    high_fit_matches.append(row_dict)
                    st.success(f"🎯 Match Found ({assessment['match_score']}/10): {job_title} at {job_company}")
                else:
                    st.warning(f"📉 Low Fit ({assessment['match_score']}/10): {job_title} at {job_company}")

                time.sleep(4)
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    st.error("🚨 Gemini Free Tier quota hit! Gracefully displaying current matches...")
                    break
                st.error(f"⚠️ Error processing row: {e}")

    st.session_state.high_fit_matches = high_fit_matches

# --- PHASE 4: DISPLAY DIGITAL DASHBOARD RESULTS WITH ATS SCORE & PDF EXPORT ---
if "high_fit_matches" in st.session_state:
    st.markdown("---")
    st.subheader("🎯 Custom Matched Opportunities Matrix")

    matches = st.session_state.high_fit_matches
    if not matches:
        st.info("No scraped listings crossed your minimum match baseline criteria during this scan.")
    else:
        for i, job in enumerate(matches):
            with st.container():
                col1, col2 = st.columns([4, 1])

                title = job.get('title', 'Position')
                company = job.get('company', 'Company')
                location = job.get('location', 'Not Specified')
                site = job.get('site', 'Web Search')
                description = job.get('description', '')
                ats_keywords = job.get('ats_keywords', [])

                with col1:
                    st.markdown(f"### {title} — *{company}*")
                    st.caption(f"📍 Location: {location} | 🌐 Source Platform: {site}")
                    st.markdown(f"**AI Evaluation Insight:** {job['reasoning']}")
                    if ats_keywords:
                        st.markdown(f"**Key Target ATS Keywords:** `{', '.join(ats_keywords)}`")

                with col2:
                    st.metric(label="Match Quality", value=f"{job['match_score']}/10")
                    target_url = job.get('job_url') if pd.notna(job.get('job_url')) else "https://google.com"
                    st.markdown(f"[🔗 Apply to Position]({target_url})")

                # --- CUSTOM RESUME & ATS SCORE EXPANDER ---
                with st.expander(f"📝 Analyze ATS Score & Build PDF Resume for {title}"):
                    if st.button(f"Generate ATS Score & Resume ##{i}"):
                        if not user_api_key:
                            st.error("API Key required.")
                        else:
                            with st.spinner("Calculating ATS Score & generating tailored resume..."):
                                res_data = generate_tailored_resume_and_ats_score(
                                    candidate_profile=st.session_state.dynamic_profile,
                                    job_title=title,
                                    job_company=company,
                                    job_description=description,
                                    ats_keywords=ats_keywords,
                                    api_key=user_api_key
                                )
                                st.session_state[f"ats_res_{i}"] = res_data

                    # Display ATS Score and Download PDF
                    if f"ats_res_{i}" in st.session_state:
                        res_data = st.session_state[f"ats_res_{i}"]

                        st.markdown("---")
                        metric_col, text_col = st.columns([1, 3])

                        with metric_col:
                            st.metric(label="📈 ATS Match Score", value=f"{res_data['ats_score']}%")

                        with text_col:
                            st.markdown(f"**ATS Optimization Breakdown:** {res_data['score_breakdown']}")

                        resume_text = res_data['tailored_resume_text']
                        st.markdown("#### Preview Tailored Resume")
                        st.text_area("Resume Content", value=resume_text, height=250)

                        # Generate PDF Stream for Download
                        pdf_file = create_pdf_from_text(resume_text)
                        clean_filename = f"Tailored_Resume_{company.replace(' ', '_')}_{title.replace(' ', '_')}.pdf"

                        st.download_button(
                            label="📥 Download Tailored Resume (.PDF)",
                            data=pdf_file,
                            file_name=clean_filename,
                            mime="application/pdf"
                        )

                st.markdown("---")
