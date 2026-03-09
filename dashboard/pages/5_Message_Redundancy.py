import streamlit as st
import requests
from utils.data_loader import load_all_data

st.subheader("Message Redundancy Detection")
st.caption("Identify repeated or highly similar communications within a selected theme.")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# -----------------------------
# HELPERS
# -----------------------------
def trim_text(text, limit=350):
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "..."

# -----------------------------
# LOAD DATA
# -----------------------------
data, errors = load_all_data()

if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

email_theme_dataset = data["email_theme_dataset"]

if email_theme_dataset.empty:
    st.warning("email_theme_dataset is empty.")
    st.stop()

if "theme" not in email_theme_dataset.columns or "clean_body" not in email_theme_dataset.columns:
    st.error("Required columns 'theme' and/or 'clean_body' are missing.")
    st.write("Available columns:", list(email_theme_dataset.columns))
    st.stop()

# -----------------------------
# IMPORT HEAVY LIBRARIES SAFELY
# -----------------------------
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    st.error(f"Could not import scikit-learn: {e}")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error(f"Could not import sentence-transformers: {e}")
    st.stop()

# -----------------------------
# USER CONTROLS
# -----------------------------
available_themes = sorted(email_theme_dataset["theme"].dropna().unique().tolist())

if not available_themes:
    st.warning("No themes available in the dataset.")
    st.stop()

selected_theme = st.selectbox(
    "Select theme",
    options=available_themes
)

similarity_threshold = st.slider(
    "Similarity threshold",
    min_value=0.70,
    max_value=0.95,
    value=0.85,
    step=0.01
)

sample_size = st.slider(
    "Emails to analyze",
    min_value=20,
    max_value=300,
    value=100,
    step=10
)

st.divider()

# -----------------------------
# FILTER DATA
# -----------------------------
theme_df = email_theme_dataset[email_theme_dataset["theme"] == selected_theme].copy()
theme_df = theme_df.dropna(subset=["clean_body"])
theme_df["clean_body"] = theme_df["clean_body"].astype(str).str.strip()

# Remove empty and very short messages to reduce false positives
theme_df = theme_df[
    (theme_df["clean_body"] != "") &
    (theme_df["clean_body"].str.len() > 30)
].head(sample_size)

st.write(f"Selected theme emails available for analysis: **{len(theme_df)}**")

if len(theme_df) < 10:
    st.warning("Not enough emails in this theme to run redundancy detection.")
    st.stop()

texts = theme_df["clean_body"].tolist()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading embedding model..."):
    model = load_model()

# -----------------------------
# EMBEDDINGS + SIMILARITY
# -----------------------------
with st.spinner("Generating embeddings and checking similarity..."):
    embeddings = model.encode(texts, show_progress_bar=False)
    similarity_matrix = cosine_similarity(embeddings)

redundant_pairs = []
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        score = similarity_matrix[i][j]
        if score >= similarity_threshold:
            redundant_pairs.append((i, j, score))

redundant_pairs = sorted(redundant_pairs, key=lambda x: x[2], reverse=True)

# -----------------------------
# OUTPUT
# -----------------------------
st.markdown("### Redundancy Summary")
col1, col2 = st.columns(2)
col1.metric("Emails Analysed", len(texts))
col2.metric("Detected Similar Message Pairs", len(redundant_pairs))

st.markdown("### Similar Message Pairs")

if not redundant_pairs:
    st.success("No high-similarity messages detected at the selected threshold. This suggests the messages in this theme are mostly unique rather than repeated.")
else:
    for idx, (i, j, score) in enumerate(redundant_pairs[:10], start=1):
        st.markdown(f"**Pair {idx} — Similarity Score: {score:.2f}**")

        c1, c2 = st.columns(2)
        with c1:
            st.write("Message A")
            st.info(trim_text(texts[i], limit=350))

        with c2:
            st.write("Message B")
            st.info(trim_text(texts[j], limit=350))

        st.divider()

st.divider()

# -----------------------------
# AI EXPLANATION
# -----------------------------
st.markdown("### AI Explanation of Redundancy")
st.caption("Generate a business-oriented explanation of what these repeated messages may indicate.")

def build_redundancy_prompt(selected_theme, emails_analyzed, redundant_pairs_count, example_pairs):
    return f"""
You are a business analyst helping explain internal communication redundancy patterns.

Use only the facts provided below. Do not invent causes or departments.
Do not mention technical ML terms.

Write the output in this exact structure:

## Key Finding
Summarize the redundancy pattern in plain language.

## Why It Matters
Explain what this may indicate about communication behaviour or workflow design.

## Suggested Actions
Provide 3 short, practical actions.

Rules:
- Be cautious and evidence-based.
- If messages look templated or automated, say so.
- If the redundancy appears low, say that the communication seems mostly unique.
- Do not blame individuals.
- Focus on process, communication design, and message repetition.
- Distinguish between automated/template-style repetition and human communication redundancy.
- If the examples look like alerts, notifications, or system-generated content, say that clearly.
- Do not imply process inefficiency unless the examples support that conclusion.

Facts:
- Theme analysed: {selected_theme}
- Emails analysed: {emails_analyzed}
- Redundant message pairs detected: {redundant_pairs_count}
- Example similar pairs:
{example_pairs}
""".strip()


@st.cache_data(show_spinner=False)
def generate_redundancy_ai_summary(prompt: str, model_name: str) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()


example_pairs = []
for i, j, score in redundant_pairs[:3]:
    example_pairs.append({
        "similarity_score": round(float(score), 2),
        "message_a": trim_text(texts[i], limit=200),
        "message_b": trim_text(texts[j], limit=200)
    })

if st.button("Generate Redundancy AI Explanation", key="redundancy_ai_btn"):
    try:
        prompt = build_redundancy_prompt(
            selected_theme=selected_theme,
            emails_analyzed=len(texts),
            redundant_pairs_count=len(redundant_pairs),
            example_pairs=example_pairs
        )

        with st.spinner("Generating redundancy explanation..."):
            redundancy_summary = generate_redundancy_ai_summary(prompt, OLLAMA_MODEL)

        if redundancy_summary:
            st.markdown(redundancy_summary)
        else:
            st.warning("The model returned an empty response.")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Ollama. Make sure Ollama is running locally on port 11434.")
    except requests.exceptions.Timeout:
        st.error("The Ollama request timed out. Try again or use a smaller/faster local model.")
    except Exception as e:
        st.error(f"Error generating redundancy explanation: {e}")