import streamlit as st
import requests
from utils.data_loader import load_all_data

st.subheader("AI Insight Lab")
st.caption("Generate business insights for specific communication patterns.")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

data, errors = load_all_data()

if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

theme_summary = data["theme_summary"]
theme_timeline = data["theme_timeline"]
sender_theme_distribution = data["sender_theme_distribution"]
email_theme_dataset = data["email_theme_dataset"]

# --------------------------------
# USER CONTROLS
# --------------------------------

st.markdown("### Select Analysis Scope")

col1, col2 = st.columns(2)

with col1:
    available_themes = sorted(email_theme_dataset["theme"].dropna().unique())
    selected_theme = st.selectbox(
        "Theme to analyze",
        options=available_themes
    )

with col2:
    available_months = sorted(theme_timeline["month"].dropna().unique())
    selected_month = st.selectbox(
        "Month focus",
        options=["All Months"] + available_months
    )

include_sender_analysis = st.checkbox(
    "Include sender concentration analysis",
    value=True
)

st.divider()

# --------------------------------
# BUILD FILTERED DATA
# --------------------------------

filtered_df = email_theme_dataset.copy()

filtered_df = filtered_df[
    filtered_df["theme"] == selected_theme
]

if selected_month != "All Months":
    month_theme_df = theme_timeline[
        (theme_timeline["theme"] == selected_theme) &
        (theme_timeline["month"] == selected_month)
    ]
    month_email_count = int(month_theme_df["email_count"].sum())
else:
    month_email_count = None

email_count = len(filtered_df)
unique_senders = filtered_df["sender"].nunique()
cluster_count = filtered_df["cluster"].nunique()

top_senders = (
    filtered_df.groupby("sender")
    .size()
    .sort_values(ascending=False)
    .head(5)
    .to_dict()
)

# --------------------------------
# BUILD PROMPT
# --------------------------------

def build_prompt():

    return f"""
You are a business communication analyst helping leadership understand internal email patterns.

Analyze the following communication slice.

Theme being analysed:
{selected_theme}

Email volume for this theme:
{email_count}

Unique senders involved:
{unique_senders}

Clusters involved:
{cluster_count}

Top senders:
{top_senders}

Month focus:
{selected_month}

If month focus is "All Months", treat the analysis as overall theme behaviour.

Write the output using this structure:

## Key Insight
Describe the most important communication pattern visible.

## Why This Matters
Explain what this pattern suggests about internal coordination or communication behaviour.

## Recommended Actions
Provide 3 practical leadership actions based only on the evidence.

Important:
If one sender appears frequently, interpret that as concentration of communication ownership, not as proof of poor performance or individual failure.

Rules:
- Do NOT invent numbers
- Avoid vague speculation
- Keep language business-oriented
- Focus on coordination load, communication concentration, and workflow behaviour
- Do NOT diagnose individuals or imply that a named sender is personally at fault
- Frame observations at the process, workflow, or communication-pattern level
- Recommended actions must focus on business processes, communication design, or workload distribution
- If a sender dominates the data, describe it as message concentration or communication dependency, not as a personal problem
- Distinguish clearly between observed facts and possible interpretations.
- Avoid assuming inefficiency unless the data directly supports it.
- Use cautious language such as "suggests", "may indicate", or "could imply".
""".strip()


# --------------------------------
# OLLAMA CALL
# --------------------------------

@st.cache_data(show_spinner=False)
def generate_summary(prompt):

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()
    result = response.json()

    return result.get("response", "").strip()


# --------------------------------
# RUN ANALYSIS
# --------------------------------

if st.button("Generate Insight"):

    prompt = build_prompt()

    with st.spinner("Analyzing communication pattern..."):

        try:
            summary = generate_summary(prompt)

            st.markdown("### AI Analysis Result")
            st.markdown(summary)

        except requests.exceptions.ConnectionError:
            st.error("Ollama is not running.")

        except Exception as e:
            st.error(f"AI generation failed: {e}")

# --------------------------------
# DEBUG VIEW
# --------------------------------

with st.expander("View analysis metrics"):
    st.json({
        "theme": selected_theme,
        "email_count": email_count,
        "unique_senders": unique_senders,
        "clusters": cluster_count,
        "top_senders": top_senders
    })