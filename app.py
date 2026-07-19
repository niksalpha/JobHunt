import streamlit as st
import pandas as pd
import yaml
import json
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from pypdf import PdfReader


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
st.write(
    "Upload your resume, set your targets, and let the agent hunt, score, and isolate matching positions instantly.")

# --- SIDEBAR: USER AUTHENTICATION & INPUTS ---
with st.sidebar:
    st.header("🔑 Authentication & Control")
    user_api_key = st.text_input("Gemini API Key", type="password",
                                 help="Input your free Gemini API Key to power the evaluations.")

    st.header("📋 Search Parameters")
    target_roles = st.text_input("Target Roles (Comma separated)", value="Python Developer, Software Engineer")
    target_location = st.text_input("Target Location", value="San Francisco, CA")
    is_remote = st.checkbox("Strictly Remote Positions Only", value=False)
    min_score = st.slider("Minimum Acceptable Match Score", min_value=1, max_value=10, value=7)

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

    # Phase 1: Dynamic Profile Creation
    with st.status("🧠 Analyzing and parsing incoming resume architecture...") as status:
        raw_resume_text = extract_text_from_pdf(uploaded_resume)
        dynamic_profile = convert_resume_to_profile(raw_resume_text, user_api_key)
        # Inject dynamic settings from UI
        dynamic_profile["target_preferences"] = {
            "roles": [r.strip() for r in target_roles.split(",")],
            "location": target_location,
            "min_fit_score": min_score
        }
        status.update(label="✅ Resume Profile Compiled In-Memory!", state="complete")

    # Phase 2: Mock Scraper Pull (Tied to Jobspy Logic)
    with st.status("🔎 Deploying Network Scrapers across LinkedIn & Indeed...") as status:
        # Real integration uses jobspy.scrape_jobs(site_name=['linkedin', 'indeed'], search_term=target_roles...)
        # Using mock dataframe to simulate incoming scrape rows for presentation flow
        mock_scraped_data = [
            {"title": "Python Backend Engineer", "company": "Alpha Tech", "location": target_location,
             "job_url": "https://linkedin.com",
             "description": "Looking for a Software Engineer with high proficiency in Python, building data automations and managing web scrapers."},
            {"title": "Frontend Developer", "company": "Beta Systems", "location": target_location,
             "job_url": "https://indeed.com",
             "description": "Building interactive UI layers using React, HTML CSS layout styling, and standard javascript frameworks."}
        ]
        df_scraped = pd.DataFrame(mock_scraped_data)
        status.update(label=f"✅ Discovered {len(df_scraped)} raw active postings!", state="complete")

    # Phase 3: Live LLM Match Evaluation
    st.subheader("📊 Live Match Evaluation Stream")
    client = genai.Client(api_key=user_api_key)
    high_fit_matches = []

    for idx, row in df_scraped.iterrows():
        with st.spinner(f"Assessing position: {row['title']} at {row['company']}..."):
            prompt = f"PROFILE:\n{json.dumps(dynamic_profile)}\n\nJOB:\nTitle: {row['title']}\nDescription: {row['description']}"
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

                # Check performance score against user filter slider
                if assessment['match_score'] >= min_score:
                    row_dict = row.to_dict()
                    row_dict['match_score'] = assessment['match_score']
                    row_dict['reasoning'] = assessment['reasoning']
                    high_fit_matches.append(row_dict)
                    st.success(f"🎯 Match Found ({assessment['match_score']}/10): {row['title']} at {row['company']}")
                else:
                    st.warning(f"📉 Low Fit ({assessment['match_score']}/10): {row['title']} at {row['company']}")

                # Free Tier Pace Throttle
                time.sleep(5)
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    st.error("🚨 Gemini Free Tier quota hit! Gracefully displaying accumulated matches...")
                    break
                st.error(f"⚠️ Error processing row: {e}")

    # --- PHASE 4: DISPLAY DIGITAL DASHBOARD RESULTS ---
    st.markdown("---")
    st.subheader("🎯 Custom Matched Opportunities Matrix")

    if not high_fit_matches:
        st.info("No jobs crossed your minimum score baseline criteria during this scan window.")
    else:
        for job in high_fit_matches:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### {job['title']} — *{job['company']}*")
                    st.caption(f"📍 Location: {job['location']}")
                    st.markdown(f"**AI Evaluation Insight:** {job['reasoning']}")
                with col2:
                    st.metric(label="Match Quality", value=f"{job['match_score']}/10")
                    st.markdown(f"[🔗 Apply to Position]({job['job_url']})")
                st.markdown("---")