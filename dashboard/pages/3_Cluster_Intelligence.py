import streamlit as st
from utils.data_loader import load_all_data

def trim_text(text, limit=400):
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "..."

st.subheader("Cluster Intelligence")
st.caption("Inspect the raw communication clusters discovered by the unsupervised ML pipeline.")

data, errors = load_all_data()
if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

cluster_explorer = data["cluster_explorer"]
email_theme_dataset = data["email_theme_dataset"]

cluster_view_df = cluster_explorer.copy()

min_cluster_size = int(cluster_view_df["email_count"].min()) if not cluster_view_df.empty else 0
max_cluster_size = int(cluster_view_df["email_count"].max()) if not cluster_view_df.empty else 1

if max_cluster_size <= min_cluster_size:
    max_cluster_size = min_cluster_size + 1

selected_min_size = st.slider(
    "Minimum emails in cluster",
    min_value=min_cluster_size,
    max_value=max_cluster_size,
    value=max(min_cluster_size, 5),
    key="cluster_min_size"
)

filtered_clusters = cluster_view_df[cluster_view_df["email_count"] >= selected_min_size].copy()
filtered_clusters = filtered_clusters.sort_values("email_count", ascending=False)

cluster_options = filtered_clusters["cluster"].dropna().unique().tolist()

if cluster_options:
    selected_cluster = st.selectbox(
        "Select a cluster to inspect",
        options=cluster_options,
        key="cluster_explorer_select"
    )

    cluster_row = filtered_clusters[filtered_clusters["cluster"] == selected_cluster].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Cluster ID", int(cluster_row["cluster"]))
    c2.metric("Assigned Theme", str(cluster_row["theme"]))
    c3.metric("Emails in Cluster", f"{int(cluster_row['email_count']):,}")

    st.markdown("### Representative Cluster Example")
    example_email = trim_text(cluster_row["example_email"], limit=400)
    st.info(example_email)

    related_emails = email_theme_dataset[email_theme_dataset["cluster"] == selected_cluster].copy()

    with st.expander("View emails mapped to this cluster"):
        preview_cols = ["sender", "theme", "clean_body"]
        available_cols = [col for col in preview_cols if col in related_emails.columns]

        preview_df = related_emails[available_cols].copy()
        if "clean_body" in preview_df.columns:
            preview_df["clean_body"] = preview_df["clean_body"].astype(str).str.slice(0, 300)

        st.dataframe(preview_df.head(20), width="stretch")

    with st.expander("View filtered cluster table"):
        st.dataframe(filtered_clusters, width="stretch")
else:
    st.warning("No clusters available for the selected minimum email count.")