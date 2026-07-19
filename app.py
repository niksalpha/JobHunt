import streamlit as st
import pandas as pd
import yaml
import json
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pypdf import PdfReader
from python_jobspy import scrape_jobs  # 🕵️‍♂️ Real-time network scraping engine

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

# --- PYDANTIC SCHEMAS ---
class JobAssessment(BaseModel):
    match_score: int = Field(..., description="Fit score from 1 to 10.")
    reasoning: str = Field(..., description="A short 2-sentence match reasoning.")
    target_ats_keywords: list[str] = Field(..., description="5-8 technical keywords.")

# --- STREAMLIT UI DESIGN ---
st.set_page_config(page_title="Autonomous Job Triage Engine", layout="wide")
st.title("🎯 Autonomous Job Triage Hub")
st.write("Upload your resume, set your targets, and let the agent hunt, score, and evaluate live matching positions instantly.")

# --- SIDEBAR: USER AUTHENTICATION & INPUTS ---
with st.sidebar:
    st.header("🔑 Authentication & Control")
    user_api_key = st.text_input("Gemini API Key", type="password", help="Input your free Gemini API Key to power the evaluations.")
    st.sidebar.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")
    
    st.header("📋 Search Parameters")
    target_roles = st.text_input("Target Roles (Comma separated)", value="Python Developer, Software Engineer")
    target_location = st.text_input("Target Location", value="San Francisco, CA")
    is_remote = st.checkbox("Strictly Remote Positions Only", value=False)
    min_score = st.slider("Minimum Acceptable Match Score", min_value=1, max_value=10, value=7)
    
    st.header("🌐 Platform Target Selection")
    choose_linkedin = st.checkbox("LinkedIn", value=True)
    choose_indeed = st.checkbox("Indeed", value=True)
    choose_ziprecruiter = st.checkbox("ZipRecruiter (Cloudflare bypass enabled)", value=False)

    st.header("📄 Candidate Profile Source")
    uploaded_resume = st.file_uploader("Upload Master Resume (PDF)", type=["pdf"])

# --- PROCESSING LOOP EXECUTION ---
if st.button("🚀 Deploy Job Hunt Triage Loop"):
    if not user_api_key:
        st.error("Please provide a valid Gemini API Key to run the triage matrix.")
        st.stop()
    if not uploaded_resume:
        st.error("Please upload a PDF resume to initialize target mapping.")
        st.stop()
        
    # Phase 1: Dynamic Profile Creation from Uploaded PDF
    with st.status("🧠 Analyzing and parsing incoming resume architecture...") as status:
        raw_resume_text = extract_text_from_pdf(uploaded_resume)
        dynamic_profile = convert_resume_to_profile(raw_resume_text, user_api_key)
        
        # Parse the comma-separated target roles into a clean Python list
        role_list = [r.strip() for r in target_roles.split(",") if r.strip()]
        
        # Inject dynamic configurations directly from the user UI layout inputs
        dynamic_profile["target_preferences"] = {
            "roles": role_list,
            "location": target_location,
            "min_fit_score": min_score
        }
        status.update(label="✅ Live Resume Profile Compiled In-Memory!", state="complete")
    
    # Phase 2: Real-time Live Network Scraper Pull via JobSpy
    platforms = []
    if choose_linkedin: platforms.append("linkedin")
    if choose_indeed: platforms.append("indeed")
    if choose_ziprecruiter: platforms.append("zip_recruiter")
    
    if not platforms:
        st.error("Please select at least one search platform in the sidebar options.")
        st.stop()

    with st.status("🔎 Deploying Network Scrapers across live target platforms...") as status:
        try:
            # We target the primary role provided in the text inputs for the scraper query
            primary_query = role_list[0] if role_list else "Software Engineer"
            
            # Execute JobSpy synchronously using the user parameters from the browser interface
            jobs_df = scrape_jobs(
                site_name=platforms,
                search_term=primary_query,
                location=target_location,
                results_wanted=10, # Kept to 10 for rapid web responsive feedback loop
                hours_old=72,
                is_remote=is_remote,
                linkedin_fetch_description=True,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )
            
            total_found = len(jobs_df)
            status.update(label=f"✅ Discovered {total_found} live active postings online!", state="complete")
        except Exception as e:
            status.update(label="⚠️ Scraping network run failed or timed out.", state="error")
            st.error(f"Network error details: {e}")
            st.stop()

    if total_found == 0:
        st.info("The live scraper returned 0 jobs for those specific keyword parameters. Try adjusting your location or role filters.")
        st.stop()

    # Phase 3: Live LLM Match Evaluation
    st.subheader("📊 Live Match Evaluation Stream")
    client = genai.Client(api_key=user_api_key)
    high_fit_matches = []

    for idx, row in jobs_df.iterrows():
        # Handle cases where description cells might drop empty string returns
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
                
                # Check metrics against user minimum layout parameter
                if assessment['match_score'] >= min_score:
                    row_dict = row.to_dict()
                    row_dict['match_score'] = assessment['match_score']
                    row_dict['reasoning'] = assessment['reasoning']
                    high_fit_matches.append(row_dict)
                    st.success(f"🎯 Match Found ({assessment['match_score']}/10): {job_title} at {job_company}")
                else:
                    st.warning(f"📉 Low Fit ({assessment['match_score']}/10): {job_title} at {job_company}")
                
                # Free Tier Rate Limit Breaker Safe Pacing
                time.sleep(4)
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    st.error("🚨 Gemini Free Tier quota hit! Gracefully displaying current matches...")
                    break
                st.error(f"⚠️ Error processing row: {e}")

    # --- PHASE 4: DISPLAY DIGITAL DASHBOARD RESULTS ---
    st.markdown("---")
    st.subheader("🎯 Custom Matched Opportunities Matrix")
    
    if not high_fit_matches:
        st.info("No scraped listings crossed your minimum match baseline criteria during this scan.")
    else:
        for job in high_fit_matches:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {job.get('title', 'Position')} — *{job.get('company', 'Company')}*")
                    st.caption(f"📍 Location: {job.get('location', 'Not Specified')} | 🌐 Source Platform: {job.get('site', 'Web Search')}")
                    st.markdown(f"**AI Evaluation Insight:** {job['reasoning']}")
                with col2:
                    st.metric(label="Match Quality", value=f"{job['match_score']}/10")
                    # Fallback cleanly if direct application target links are missing
                    target_url = job.get('job_url') if pd.notna(job.get('job_url')) else "https://google.com"
                    st.markdown(f"[🔗 Apply to Position]({target_url})")
                st.markdown("---")
