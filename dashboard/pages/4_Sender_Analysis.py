import streamlit as st
from utils.data_loader import load_all_data

st.subheader("Sender Analysis")
st.caption("Understand who drives different communication themes across the organization.")

data, errors = load_all_data()
if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

sender_theme_distribution = data["sender_theme_distribution"]

st.markdown("### Top Communicators Overall")

sender_totals = (
    sender_theme_distribution
    .groupby("sender", as_index=False)["email_count"]
    .sum()
    .sort_values("email_count", ascending=False)
)

top_n_senders = st.slider(
    "Number of top senders to display",
    min_value=5,
    max_value=30,
    value=15,
    key="top_sender_slider"
)

top_senders_df = sender_totals.head(top_n_senders)

st.bar_chart(
    top_senders_df.set_index("sender")["email_count"],
    width="stretch"
)

with st.expander("View sender totals table"):
    st.dataframe(top_senders_df, width="stretch")

st.markdown("### Top Senders by Theme")

available_themes_sender = sorted(
    sender_theme_distribution["theme"].dropna().unique().tolist()
)

selected_sender_theme = st.selectbox(
    "Select theme",
    options=available_themes_sender,
    key="sender_theme_select"
)

theme_sender_df = (
    sender_theme_distribution[
        sender_theme_distribution["theme"] == selected_sender_theme
    ]
    .groupby("sender", as_index=False)["email_count"]
    .sum()
    .sort_values("email_count", ascending=False)
)

top_theme_senders = theme_sender_df.head(15)

st.bar_chart(
    top_theme_senders.set_index("sender")["email_count"],
    width="stretch"
)

with st.expander("View sender-theme table"):
    st.dataframe(top_theme_senders, width="stretch")