import json
import re
import time
from io import BytesIO
from typing import Any, Dict, List
from fpdf import FPDF
from google import genai
from google.genai import types
from jobspy import scrape_jobs
import pandas as pd
from pydantic import BaseModel, Field
from pypdf import PdfReader
import streamlit as st
import streamlit.components.v1 as components


# --- CUSTOM RESUME TEXT EXTRACTOR ---
def extract_text_from_pdf(uploaded_file: Any) -> str:
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"
    return full_text


# --- LLM PROFILE GENERATOR ---
def convert_resume_to_profile(raw_text: str, api_key: str) -> Dict[str, Any]:
    client = genai.Client(api_key=api_key)
    system_instruction = (
        "Extract unstructured data from a candidate's resume and format it into"
        " clean JSON matching the baseline architecture."
    )

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
    gen_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(str(gen_response.text)) if gen_response.text else {}


# --- ATS TAILORED RESUME & SCORE SCHEMA ---
class TailoredResumeResponse(BaseModel):
    ats_score: int = Field(
        ...,
        description="Estimated ATS Match score from 0 to 100 based on keyword and skill coverage.",
    )
    score_breakdown: str = Field(
        ...,
        description="Short explanation of why this score was given and target keywords matched.",
    )
    tailored_resume_text: str = Field(
        ...,
        description=(
            "The complete tailored resume formatted cleanly in Markdown"
            " (Use # for Name, ## for Sections like EXPERIENCE, SKILLS, ### for"
            " Job Titles/Company, and * for bullet points)."
        ),
    )


def generate_tailored_resume_and_ats_score(
    candidate_profile: Dict[str, Any],
    job_title: str,
    job_company: str,
    job_description: str,
    ats_keywords: List[str],
    api_key: str,
) -> Dict[str, Any]:
    """Generates a tailored resume alongside an ATS optimization score using Gemini structured outputs."""
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
    2. Format the resume using standard Markdown:
       - `# CANDIDATE NAME` at top
       - Contact info line immediately under name
       - `## SUMMARY`
       - `## EXPERIENCE`
       - `### Job Title | Company | Location | Dates` for role headers
       - Bullet points using `*` for achievements
       - `## SKILLS` (categorized cleanly)
    3. Naturally incorporate the target ATS keywords into experience bullet points without keyword stuffing.
    4. Evaluate the tailored resume against the job description and assign an ATS Match Score (0-100%).
    5. Provide a brief 2-sentence score breakdown explaining keyword match coverage.
    """

    gen_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TailoredResumeResponse,
            temperature=0.2,
        ),
    )
    return json.loads(str(gen_response.text)) if gen_response.text else {}


# --- SANITIZE UNICODE FOR FPDF ---
def sanitize_text_for_pdf(text: str) -> str:
    """Replaces non-latin1 characters with safe ASCII equivalents."""
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2026": "...",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


# --- MODERN HIGH-QUALITY ATS PDF ENGINE ---
class ATS_Resume_PDF(FPDF):
    def header(self) -> None:
        pass

    def footer(self) -> None:
        pass

    def draw_bold_markdown_line(self, markdown_line: str, default_font_size: int = 10) -> None:
        """Parses line for **bold text** and renders inline styled segments."""
        parts = re.split(r"(\*\*.*?\*\*)", markdown_line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                clean_part = part[2:-2]
                self.set_font("Helvetica", "B", default_font_size)
                self.write(5, clean_part)
            else:
                self.set_font("Helvetica", "", default_font_size)
                self.write(5, part)
        self.ln(5)


def create_pdf_from_text(text_content: str) -> BytesIO:
    """Converts Markdown formatted resume text into a highly styled, ATS-compliant PDF document."""
    pdf = ATS_Resume_PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=15, right=15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    clean_text = sanitize_text_for_pdf(text_content)

    lines = clean_text.split("\n")
    i = 0

    while i < len(lines):
        line_item = lines[i].strip()

        if not line_item:
            i += 1
            continue

        # Main Candidate Name Header (#)
        if line_item.startswith("# "):
            name_text = line_item.replace("# ", "").strip().upper()
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(24, 43, 73)  # Professional Navy Blue Accent
            pdf.cell(usable_w, 9, name_text, ln=True, align="C")
            pdf.ln(1)

        # Section Headers (## EXPERIENCE, ## SKILLS, ## SUMMARY)
        elif line_item.startswith("## "):
            section_title = line_item.replace("## ", "").strip().upper()
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(24, 43, 73)
            pdf.cell(usable_w, 6, section_title, ln=True)

            # Section Underline Divider
            pdf.set_draw_color(200, 205, 215)
            pdf.set_line_width(0.4)
            current_y = pdf.get_y()
            pdf.line(pdf.l_margin, current_y, pdf.w - pdf.r_margin, current_y)
            pdf.ln(3)
            pdf.set_text_color(40, 40, 40)  # Reset body text color

        # Sub-Headers / Job Titles / Companies (###)
        elif line_item.startswith("### "):
            subtitle = line_item.replace("### ", "").strip()
            pdf.ln(1.5)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(usable_w, 5, subtitle)
            pdf.ln(1)

        # Bullet Points (* or -)
        elif line_item.startswith("* ") or line_item.startswith("- "):
            bullet_text = line_item[2:].strip()
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)

            # Indented bullet point formatting
            pdf.set_x(pdf.l_margin)
            pdf.cell(5, 5, "-", ln=False)  # Clean bullet point hyphen
            pdf.set_x(pdf.l_margin + 5)
            pdf.multi_cell(usable_w - 5, 5, bullet_text)
            pdf.ln(0.5)

        # Standard body text & Contact Details
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(pdf.l_margin)

            if "**" in line_item:
                pdf.draw_bold_markdown_line(line_item, default_font_size=10)
            else:
                # Check if centered contact info line (under main title)
                if i < 3 and "@" in line_item:
                    pdf.set_font("Helvetica", "", 9.5)
                    pdf.set_text_color(100, 100, 100)
                    pdf.cell(usable_w, 5, line_item, ln=True, align="C")
                    pdf.ln(2)
                else:
                    pdf.multi_cell(usable_w, 5, line_item)
                    pdf.ln(1)

        i += 1

    # Safe version-agnostic PDF byte export
    raw_output = pdf.output()
    if isinstance(raw_output, (bytes, bytearray)):
        pdf_bytes = bytes(raw_output)
    elif isinstance(raw_output, str):
        pdf_bytes = raw_output.encode("latin-1", errors="replace")
    else:
        pdf_bytes = bytes(raw_output)

    pdf_output = BytesIO(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output


# --- PYDANTIC SCHEMAS ---
class JobAssessment(BaseModel):
    match_score: int = Field(..., description="Fit score from 1 to 10.")
    reasoning: str = Field(..., description="A short 2-sentence match reasoning.")
    target_ats_keywords: List[str] = Field(default=[], description="5-8 technical keywords.")


# --- STREAMLIT UI DESIGN ---
st.set_page_config(page_title="Autonomous Job Triage Engine", layout="wide")
st.title("🎯 Autonomous Job Triage Hub")
st.write(
    str(
        "Upload your resume, set your targets, and let the agent hunt, "
        "score, evaluate, and generate tailored ATS resumes instantly. Don't "
        "worry, we store no personal information."
    )
)

# --- SIDEBAR: USER AUTHENTICATION & INPUTS ---
with st.sidebar:
    st.header("🔑 Authentication & Control")
    user_api_key = st.text_input(
        label="Gemini API Key",
        type="password",
        help="Input your free Gemini API Key to power the evaluations.",
    )
    st.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/)")

    st.header("📋 Search Parameters")
    target_roles = st.text_input(label="Target Roles (Comma separated)", value="")
    target_location = st.text_input(label="Target Location", value="Hyderabad, Telangana")

    results_wanted_input = st.text_input(
        label="Max Job Results to Fetch",
        value="5",
        help=(
            "Specify how many job postings to fetch per platform (1-50). Lower "
            "numbers help stay within Gemini free-tier rate limits."
        ),
    )

    num_results = 5
    if results_wanted_input.strip():
        if not results_wanted_input.strip().isdigit():
            st.warning("⚠️ Please enter a positive integer for Max Job Results (defaulting to 5).")
        else:
            parsed_val = int(results_wanted_input.strip())
            if parsed_val <= 0:
                st.warning("⚠️ Value must be greater than 0 (defaulting to 5).")
            else:
                num_results = parsed_val

    is_remote = st.checkbox(label="Strictly Remote Positions Only", value=False)
    min_score = st.slider(label="Minimum Acceptable Match Score", min_value=1, max_value=10, value=7)

    st.header("🌐 Platform Target Selection")
    choose_linkedin = st.checkbox(label="LinkedIn", value=True)
    choose_indeed = st.checkbox(label="Indeed", value=True)

    st.header("📄 Candidate Profile Source")
    uploaded_resume = st.file_uploader(label="Upload Master Resume (PDF)", type=["pdf"])

if "dynamic_profile" not in st.session_state:
    st.session_state.dynamic_profile = None

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

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
    status_container = st.status(label="🧠 Lemme analyze and parse incoming resume, hang on!...")
    with status_container:
        raw_resume_text = extract_text_from_pdf(uploaded_resume)
        dynamic_profile = convert_resume_to_profile(raw_resume_text, user_api_key)

        role_list = [r.strip() for r in target_roles.split(",") if r.strip()]

        dynamic_profile["target_preferences"] = {
            "roles": role_list,
            "location": target_location,
            "min_fit_score": min_score,
        }
        st.session_state.dynamic_profile = dynamic_profile
        status_container.update(
            label="✅ Live Resume Profile Compiled In-Memory!", state="complete"
        )

    # Phase 2: Real-time Live Network Scraper Pull via JobSpy
    platforms = []
    if choose_linkedin:
        platforms.append("linkedin")
    if choose_indeed:
        platforms.append("indeed")

    if not platforms:
        st.error("Please select at least one search platform in the sidebar options.")
        st.stop()

    scrape_status = st.status(label="🔎 Deploying Network Scrapers across live target platforms...")
    total_found = 0
    jobs_df = pd.DataFrame()
    with scrape_status:
        try:
            primary_query = role_list[0] if role_list else "Software Engineer"

            jobs_df = scrape_jobs(
                site_name=platforms,
                search_term=primary_query,
                location=target_location,
                results_wanted=num_results,
                hours_old=72,
                is_remote=is_remote,
                linkedin_fetch_description=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                        " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0"
                        " Safari/537.36"
                    )
                },
            )

            total_found = len(jobs_df)
            scrape_status.update(
                label=f"✅ Discovered {total_found} live active postings online!",
                state="complete",
            )
        except Exception as e:
            scrape_status.update(label="⚠️ Scraping network run failed or timed out.", state="error")
            st.error(f"Network error details: {e}")
            st.stop()

    if total_found == 0 or jobs_df.empty:
        st.info(
            "The live scraper returned 0 jobs for those specific keyword "
            "parameters. Try adjusting your location or role filters."
        )
        st.stop()

    # Phase 3: Live LLM Match Evaluation
    st.subheader("📊 Live Match Evaluation Stream")
    eval_client = genai.Client(api_key=user_api_key)
    high_fit_matches = []

    for idx, row in jobs_df.iterrows():
        job_description = row.get("description", "")
        if not job_description or pd.isna(job_description):
            continue

        job_title = str(row.get("title", "Unknown Title"))
        job_company = str(row.get("company", "Unknown Company"))

        spinner_message = f"Assessing position: {job_title} at {job_company}..."
        with st.spinner(text=spinner_message):
            eval_prompt = (
                f"PROFILE:\n{json.dumps(dynamic_profile)}\n\nJOB:\nTitle:"
                f" {job_title}\nDescription: {job_description}"
            )
            try:
                eval_response = eval_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=eval_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "Unbiasedly score the job against the profile (1-10) and return JSON."
                        ),
                        response_mime_type="application/json",
                        response_schema=JobAssessment,
                        temperature=0.2,
                    ),
                )
                assessment = json.loads(str(eval_response.text)) if eval_response.text else {}

                match_score = assessment.get("match_score", 0)
                if match_score >= min_score:
                    row_dict = row.to_dict()
                    row_dict["match_score"] = match_score
                    row_dict["reasoning"] = assessment.get("reasoning", "")
                    row_dict["ats_keywords"] = assessment.get("target_ats_keywords", [])
                    high_fit_matches.append(row_dict)
                    st.success(f"🎯 Match Found ({match_score}/10): {job_title} at {job_company}")
                else:
                    st.warning(f"📉 Low Fit ({match_score}/10): {job_title} at {job_company}")

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
        for idx, job in enumerate(matches):
            with st.container():
                col1, col2 = st.columns([4, 1])

                title = str(job.get("title", "Position"))
                company = str(job.get("company", "Company"))
                location = str(job.get("location", "Not Specified"))
                site = str(job.get("site", "Web Search"))
                description = str(job.get("description", ""))
                ats_keywords = job.get("ats_keywords", [])

                with col1:
                    st.markdown(f"### {title} — *{company}*")
                    st.caption(f"📍 Location: {location} | 🌐 Source Platform: {site}")
                    st.markdown(f"**AI Evaluation Insight:** {job['reasoning']}")
                    if ats_keywords:
                        st.markdown(f"**Key Target ATS Keywords:** `{', '.join(ats_keywords)}`")

                with col2:
                    st.metric(label="Match Quality", value=f"{job['match_score']}/10")
                    raw_url = job.get("job_url")
                    target_url = str(raw_url) if pd.notna(raw_url) else "https://google.com"
                    st.markdown(f"[🔗 Apply to Position]({target_url})")

                btn_key = f"btn_gen_{idx}_{hash(title)}"
                res_key = f"ats_res_{idx}_{hash(title)}"

                # --- CUSTOM RESUME & ATS SCORE EXPANDER ---
                # Sets expanded to True once results are ready so user doesn't have to manually click
                is_expanded = res_key in st.session_state
                with st.expander(
                    f"📝 ATS Score & Tailored Resume for {title}",
                    expanded=is_expanded,
                ):
                    if st.button("Generate ATS Score & Tailored Resume", key=btn_key):
                        if not user_api_key:
                            st.error("API Key required.")
                        else:
                            with st.spinner(text="Calculating ATS Score & generating tailored resume..."):
                                try:
                                    res_data = generate_tailored_resume_and_ats_score(
                                        candidate_profile=st.session_state.dynamic_profile,
                                        job_title=title,
                                        job_company=company,
                                        job_description=description,
                                        ats_keywords=ats_keywords,
                                        api_key=user_api_key,
                                    )
                                    st.session_state[res_key] = res_data

                                    # Notify user via Streamlit Toast & Web Audio chime
                                    st.toast(
                                        f"🎉 Tailored Resume ready for {title}!",
                                        icon="✅",
                                    )
                                    components.html(
                                        """
                                        <script>
                                            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                                            const osc = audioCtx.createOscillator();
                                            const gain = audioCtx.createGain();
                                            osc.type = 'sine';
                                            osc.frequency.setValueAtTime(587.33, audioCtx.currentTime); // D5
                                            osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.15); // A5
                                            gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
                                            gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.5);
                                            osc.connect(gain);
                                            gain.connect(audioCtx.destination);
                                            osc.start();
                                            osc.stop(audioCtx.currentTime + 0.5);
                                        </script>
                                        """,
                                        height=0,
                                    )
                                    st.rerun()

                                except Exception as e:
                                    if (
                                        "429" in str(e)
                                        or "RESOURCE_EXHAUSTED" in str(e)
                                        or "Quota" in str(e)
                                    ):
                                        st.error(
                                            "🚨 Gemini API Rate Limit / Quota reached! Please wait "
                                            "for sometime and click the button again or use a "
                                            "different Key if available."
                                        )
                                    else:
                                        st.error(f"⚠️ API Error: {e}")

                    # Display ATS Score, Markdown Copy Option, and Download PDF automatically once loaded
                    if res_key in st.session_state:
                        res_data = st.session_state[res_key]

                        st.markdown("---")
                        metric_col, text_col = st.columns([1, 3])

                        with metric_col:
                            st.metric(label="📈 ATS Match Score", value=f"{res_data['ats_score']}%")

                        with text_col:
                            st.markdown(f"**ATS Optimization Breakdown:** {res_data['score_breakdown']}")

                        resume_text = res_data["tailored_resume_text"]
                        st.markdown("#### Preview Tailored Resume")
                        st.text_area(
                            label="Resume Preview",
                            value=resume_text,
                            height=250,
                            key=f"txt_{res_key}",
                        )

                        # --- COPY MARKDOWN & DOWNLOAD OPTIONS ---
                        st.markdown("#### 📤 Export Options")
                        dl_col, copy_col = st.columns([1, 1])

                        with dl_col:
                            # Generate PDF Stream for Download safely
                            pdf_file = create_pdf_from_text(resume_text)
                            clean_filename = (
                                f"Tailored_Resume_{company.replace(' ', '_')}_{title.replace(' ', '_')}.pdf"
                            )

                            st.download_button(
                                label="📥 Download Tailored Resume (.PDF)",
                                data=pdf_file,
                                file_name=clean_filename,
                                mime="application/pdf",
                                key=f"dl_{res_key}",
                            )

                        with copy_col:
                            # Custom HTML/JS Copy Button
                            escaped_markdown = json.dumps(resume_text)
                            copy_button_html = f"""
                            <button id="copyBtn_{idx}" style="
                                background-color: #2e7d32;
                                color: white;
                                padding: 8px 16px;
                                border: none;
                                border-radius: 6px;
                                font-weight: bold;
                                cursor: pointer;
                                width: 100%;
                                height: 42px;
                            ">📋 Copy Resume (Markdown)</button>

                            <script>
                            document.getElementById("copyBtn_{idx}").addEventListener("click", function() {{
                                const textToCopy = {escaped_markdown};
                                navigator.clipboard.writeText(textToCopy).then(function() {{
                                    const btn = document.getElementById("copyBtn_{idx}");
                                    btn.innerText = "✅ Copied to Clipboard!";
                                    btn.style.backgroundColor = "#1b5e20";
                                    setTimeout(() => {{
                                        btn.innerText = "📋 Copy Resume (Markdown)";
                                        btn.style.backgroundColor = "#2e7d32";
                                    }}, 2500);
                                }}).catch(function(err) {{
                                    console.error("Failed to copy: ", err);
                                }});
                            }});
                            </script>
                            """
                            components.html(copy_button_html, height=50)

                st.markdown("---")
