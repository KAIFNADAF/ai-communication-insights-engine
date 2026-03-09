import streamlit as st
from utils.data_loader import load_all_data

def trim_text(text, limit=400):
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "..."

st.subheader("Theme Explorer")
st.caption("Inspect AI-discovered themes through representative email examples.")

data, errors = load_all_data()
if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

theme_examples = data["theme_examples"]
email_theme_dataset = data["email_theme_dataset"]

available_example_themes = sorted(theme_examples["theme"].dropna().unique().tolist())

selected_theme = st.selectbox(
    "Select a theme to inspect",
    options=available_example_themes,
    key="theme_explorer_select"
)

theme_example_rows = theme_examples[theme_examples["theme"] == selected_theme].copy()
theme_email_rows = email_theme_dataset[email_theme_dataset["theme"] == selected_theme].copy()

theme_count = len(theme_email_rows)
unique_senders = theme_email_rows["sender"].nunique() if not theme_email_rows.empty else 0
cluster_count = theme_email_rows["cluster"].nunique() if not theme_email_rows.empty else 0

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Emails in Theme", f"{theme_count:,}")
metric_col2.metric("Unique Senders", f"{unique_senders:,}")
metric_col3.metric("Clusters Represented", f"{cluster_count:,}")

st.markdown(f"### Example Communication for: {selected_theme}")

if not theme_example_rows.empty:
    for i, row in enumerate(theme_example_rows.head(3).itertuples(index=False), start=1):
        example_text = trim_text(row.clean_body, limit=400)

        st.markdown(f"**Example {i}**")
        st.info(example_text)
else:
    st.warning("No representative examples available for this theme.")

with st.expander("View matching emails from the master dataset"):
    preview_cols = ["sender", "cluster", "clean_body"]
    available_preview_cols = [col for col in preview_cols if col in theme_email_rows.columns]

    preview_df = theme_email_rows[available_preview_cols].copy()

    if "clean_body" in preview_df.columns:
        preview_df["clean_body"] = preview_df["clean_body"].astype(str).str.slice(0, 300)

    st.dataframe(preview_df.head(20), width="stretch")