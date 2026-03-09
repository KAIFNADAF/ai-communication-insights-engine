import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

REQUIRED_FILES = {
    "theme_summary": "theme_summary.csv",
    "theme_timeline": "theme_timeline.csv",
    "cluster_explorer": "cluster_explorer.csv",
    "theme_examples": "theme_examples.csv",
    "sender_theme_distribution": "sender_theme_distribution.csv",
    "email_theme_dataset": "email_theme_dataset.csv",
}

EXPECTED_COLUMNS = {
    "theme_summary": {"theme", "email_count"},
    "theme_timeline": {"month", "theme", "email_count"},
    "cluster_explorer": {"cluster", "theme", "email_count", "example_email"},
    "theme_examples": {"theme", "clean_body"},
    "sender_theme_distribution": {"sender", "theme", "email_count"},
    "email_theme_dataset": {"sender", "clean_body", "cluster", "theme"},
}

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

def validate_dataframe(df: pd.DataFrame, required_cols: set, file_label: str) -> list[str]:
    missing = required_cols - set(df.columns)
    if missing:
        return [f"{file_label}: missing columns -> {', '.join(sorted(missing))}"]
    return []

@st.cache_data
def load_all_data():
    data = {}
    errors = []

    for key, filename in REQUIRED_FILES.items():
        file_path = DATA_DIR / filename

        if not file_path.exists():
            errors.append(f"Missing file: {file_path}")
            continue

        try:
            df = load_csv(file_path)
            data[key] = df
            errors.extend(validate_dataframe(df, EXPECTED_COLUMNS[key], filename))
        except Exception as e:
            errors.append(f"Error loading {filename}: {e}")

    return data, errors