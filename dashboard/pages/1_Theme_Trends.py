import streamlit as st
from utils.data_loader import load_all_data

st.subheader("Theme & Trend Analysis")
st.caption("Themes were discovered automatically using semantic clustering and AI-assisted labeling.")

data, errors = load_all_data()
if errors:
    st.error("Data loading failed.")
    for err in errors:
        st.write(f"- {err}")
    st.stop()

theme_summary = data["theme_summary"]
theme_timeline = data["theme_timeline"]

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### Top Communication Themes")

    max_themes = max(5, min(20, len(theme_summary)))
    default_themes = min(10, len(theme_summary)) if len(theme_summary) > 0 else 5

    top_n = st.slider(
        "Select number of themes to display",
        min_value=5,
        max_value=max_themes,
        value=default_themes,
        key="top_n_themes"
    )

    theme_chart_df = (
        theme_summary.sort_values("email_count", ascending=False)
        .head(top_n)
        .copy()
    )

    st.bar_chart(
        theme_chart_df.set_index("theme")["email_count"],
        width="stretch"
    )

    with st.expander("View theme summary table"):
        st.dataframe(theme_chart_df, width="stretch")

with right_col:
    st.markdown("### Communication Trends Over Time")

    available_themes = sorted(theme_timeline["theme"].dropna().unique().tolist())
    default_theme_count = min(5, len(available_themes))

    selected_themes = st.multiselect(
        "Select themes to compare over time",
        options=available_themes,
        default=available_themes[:default_theme_count],
        key="selected_themes_timeline"
    )

    timeline_df = theme_timeline.copy()

    if selected_themes:
        timeline_df = timeline_df[timeline_df["theme"].isin(selected_themes)]

    timeline_pivot = (
        timeline_df.pivot_table(
            index="month",
            columns="theme",
            values="email_count",
            aggfunc="sum",
            fill_value=0
        ).sort_index()
    )

    if not timeline_pivot.empty:
        st.line_chart(timeline_pivot, width="stretch")
    else:
        st.warning("No timeline data available for the selected themes.")

    with st.expander("View timeline data table"):
        st.dataframe(timeline_df.sort_values(["month", "theme"]), width="stretch")

st.divider()

st.markdown("### Communication Anomaly Detection")
st.caption("Highlights months where communication volume was unusually high compared with the overall monthly pattern.")

monthly_volume = (
    theme_timeline.groupby("month", as_index=False)["email_count"]
    .sum()
    .sort_values("month")
)

if len(monthly_volume) >= 3:
    mean_volume = monthly_volume["email_count"].mean()
    std_volume = monthly_volume["email_count"].std()

    anomaly_threshold = mean_volume + (2 * std_volume)
    anomalies = monthly_volume[monthly_volume["email_count"] > anomaly_threshold].copy()

    if not anomalies.empty:
        st.warning("Unusual communication spikes detected.")

        anomaly_display = anomalies.copy()
        anomaly_display["vs_average"] = (
            anomaly_display["email_count"] / mean_volume
        ).round(2)

        st.dataframe(
            anomaly_display.rename(columns={
                "month": "Month",
                "email_count": "Email Volume",
                "vs_average": "Times Above Average"
            }),
            width="stretch"
        )

        top_anomaly = anomaly_display.sort_values("email_count", ascending=False).iloc[0]
        st.info(
            f"Highest spike detected in {top_anomaly['month']}, "
            f"with {int(top_anomaly['email_count']):,} emails. "
            f"This was {top_anomaly['vs_average']:.2f}× above the monthly average."
        )
    else:
        st.success("No major communication spikes detected.")
else:
    st.info("Not enough monthly data available to detect anomalies.")