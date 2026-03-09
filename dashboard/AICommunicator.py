import streamlit as st
import requests
from utils.data_loader import load_all_data

st.set_page_config(
    page_title="AI Communication Insights Engine",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

st.title("AI Communication Insights Engine")
st.caption("AI-powered analysis of internal communication patterns")

data, errors = load_all_data()

if errors:
    st.error("Some data files could not be loaded correctly.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

theme_summary = data["theme_summary"]
theme_timeline = data["theme_timeline"]
cluster_explorer = data["cluster_explorer"]
sender_theme_distribution = data["sender_theme_distribution"]
email_theme_dataset = data["email_theme_dataset"]

st.caption("New here? Open the guide below for a quick walkthrough of what each section shows.")

with st.expander("How to use this dashboard", expanded=False):
    st.write("""
This dashboard looks at internal email communication and helps you spot patterns.

You can use it to understand:\n
\n• what topics appear most often\n
\n• when communication increases\n
\n• who sends most of the messages\n
\n• whether some messages are repeated\n
\n• what business insights the AI can generate\n
""")

    st.markdown("""
**What each section shows**

**Theme Trends**  
Shows the main topics in the email data and how they change over time.

**Theme Explorer**  
Lets you read example emails for each theme.

**Cluster Intelligence**  
Shows the raw groups found by the machine learning model.

**Sender Analysis**  
Shows who sends the most messages and which themes they contribute to.

**Message Redundancy**  
Finds very similar or repeated emails.

**AI Insights**  
Generates short business-style explanations of the patterns in the data.
""")

    st.markdown("""
**Technology used**

• Text embeddings to understand email meaning  
• Clustering to group similar emails into themes  
• Similarity scoring to detect repeated messages  
• Local LLM to generate plain-English summaries
""")
       

# -----------------------------
# KPI METRICS
# -----------------------------
total_emails = int(theme_summary["email_count"].sum())
total_themes = int(theme_summary["theme"].nunique())

largest_theme_row = theme_summary.sort_values("email_count", ascending=False).iloc[0]
largest_theme = str(largest_theme_row["theme"])
largest_theme_count = int(largest_theme_row["email_count"])

month_totals = theme_timeline.groupby("month", as_index=False)["email_count"].sum()
most_active_month_row = month_totals.sort_values("email_count", ascending=False).iloc[0]
most_active_month = str(most_active_month_row["month"])
most_active_month_count = int(most_active_month_row["email_count"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Emails Analysed", f"{total_emails:,}")
col2.metric("Themes Discovered", total_themes)
col3.metric("Largest Theme", f"{largest_theme} ({largest_theme_count:,})")
col4.metric("Most Active Month", f"{most_active_month} ({most_active_month_count:,})")

st.divider()



# -----------------------------
# EXECUTIVE OVERVIEW TEXT
# -----------------------------
st.subheader("Executive Overview")
st.write(
    """
This dashboard uses AI and unsupervised machine learning to discover communication themes,
analyze trends, inspect message clusters, and generate business-focused summaries.

Use the AI summary below for a quick leadership view, then explore the sidebar pages
for deeper evidence on themes, clusters, sender behaviour, and communication patterns.
"""
)

# -----------------------------
# AI SUMMARY HELPERS
# -----------------------------
def build_insight_context():
    top_themes = (
        theme_summary.sort_values("email_count", ascending=False)
        .head(5)[["theme", "email_count"]]
        .to_dict(orient="records")
    )

    top_senders = (
        sender_theme_distribution.groupby("sender", as_index=False)["email_count"]
        .sum()
        .sort_values("email_count", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )

    theme_sender_diversity = (
        email_theme_dataset.groupby("theme")["sender"]
        .nunique()
        .reset_index(name="unique_senders")
        .sort_values("unique_senders", ascending=False)
    )

    most_diverse_theme = (
        theme_sender_diversity.iloc[0].to_dict()
        if not theme_sender_diversity.empty
        else {"theme": "N/A", "unique_senders": 0}
    )

    largest_cluster_data = (
        cluster_explorer.sort_values("email_count", ascending=False)
        .head(1)[["cluster", "theme", "email_count"]]
        .to_dict(orient="records")
    )

    largest_cluster = largest_cluster_data[0] if largest_cluster_data else {
        "cluster": "N/A",
        "theme": "N/A",
        "email_count": 0
    }

    return {
        "total_emails": total_emails,
        "total_themes": total_themes,
        "largest_theme": {
            "theme": largest_theme,
            "email_count": largest_theme_count
        },
        "most_active_month": {
            "month": most_active_month,
            "email_count": most_active_month_count
        },
        "top_themes": top_themes,
        "top_senders": top_senders,
        "most_diverse_theme": most_diverse_theme,
        "largest_cluster": largest_cluster
    }


def build_ollama_prompt(context: dict) -> str:
    return f"""
You are an AI business analyst helping leadership understand internal communication patterns.

Use only the facts provided below. Do not invent numbers, causes, departments, or events.
Do not mention technical machine learning terms such as embeddings, clustering, UMAP, or HDBSCAN.

Write the output in this exact structure:

## Executive Summary
- 3 concise bullets describing the most important communication patterns.

## Why It Matters
Provide 2 short paragraphs explaining the likely business relevance of these patterns.

## Recommended Actions
1. A practical action for leadership.
2. A second practical action for leadership.
3. A third practical action for leadership.

Style requirements:
- business-oriented
- concise
- analytical
- action-focused
- grounded in the provided facts
- where uncertain, use phrases like "suggests" or "may indicate"

Facts:
- Total emails analysed: {context['total_emails']}
- Total themes discovered: {context['total_themes']}
- Largest theme: {context['largest_theme']['theme']} ({context['largest_theme']['email_count']} emails)
- Most active month: {context['most_active_month']['month']} ({context['most_active_month']['email_count']} emails)
- Theme with widest sender participation: {context['most_diverse_theme']['theme']} ({context['most_diverse_theme']['unique_senders']} unique senders)
- Largest cluster: Cluster {context['largest_cluster']['cluster']} mapped to {context['largest_cluster']['theme']} ({context['largest_cluster']['email_count']} emails)

Top themes:
{context['top_themes']}

Top senders overall:
{context['top_senders']}

Focus on:
- communication load
- coordination overhead
- concentration of message ownership
- broad participation vs concentrated communication
- practical leadership action
""".strip()


@st.cache_data(show_spinner=False)
def generate_ollama_insight(prompt: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()


# -----------------------------
# AI EXECUTIVE SUMMARY
# -----------------------------
st.markdown("### AI Executive Summary")
st.info("Generate a business-focused summary of the main communication patterns and recommended next steps.")

insight_context = build_insight_context()

summary_col, side_col = st.columns([2, 1])

with summary_col:
    if st.button("Generate Executive Summary", key="overview_ai_summary_btn"):
        try:
            prompt = build_ollama_prompt(insight_context)
            with st.spinner("Generating executive AI summary..."):
                ai_summary = generate_ollama_insight(prompt, OLLAMA_MODEL)

            if ai_summary:
                st.markdown(ai_summary)
            else:
                st.warning("The model returned an empty response.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to Ollama. Make sure Ollama is running locally on port 11434.")
        except requests.exceptions.Timeout:
            st.error("The Ollama request timed out. Try again or use a smaller/faster local model.")
        except Exception as e:
            st.error(f"Error generating AI summary: {e}")

with side_col:
    st.markdown("#### Quick Signals")

    top_sender_row = (
        sender_theme_distribution.groupby("sender", as_index=False)["email_count"]
        .sum()
        .sort_values("email_count", ascending=False)
        .iloc[0]
    )

    st.metric("Top Sender", str(top_sender_row["sender"]))
    st.metric("Top Sender Volume", f"{int(top_sender_row['email_count']):,}")

    most_diverse_theme_row = (
        email_theme_dataset.groupby("theme")["sender"]
        .nunique()
        .reset_index(name="unique_senders")
        .sort_values("unique_senders", ascending=False)
        .iloc[0]
    )

    st.metric("Broadest Participation Theme", str(most_diverse_theme_row["theme"]))
    st.metric("Unique Senders in Theme", f"{int(most_diverse_theme_row['unique_senders']):,}")

with st.expander("View AI summary input metrics"):
    st.json(insight_context)

st.divider()

# -----------------------------
# QUICK VISUAL SNAPSHOT
# -----------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Top Themes Snapshot")
    top_themes_df = (
        theme_summary.sort_values("email_count", ascending=False)
        .head(8)
        .copy()
    )
    st.bar_chart(
        top_themes_df.set_index("theme")["email_count"],
        use_container_width=True
    )

with right_col:
    st.markdown("### Monthly Communication Volume")
    monthly_volume_df = (
        theme_timeline.groupby("month", as_index=False)["email_count"]
        .sum()
        .sort_values("month")
    )
    st.line_chart(
        monthly_volume_df.set_index("month")["email_count"],
        use_container_width=True
    )

st.divider()

st.caption(
    "Use the sidebar to explore detailed theme trends, representative emails, cluster structure, sender patterns, and the dedicated AI insights page."
)