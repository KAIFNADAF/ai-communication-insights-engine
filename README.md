# AI Communication Insights Engine

The **AI Communication Insights Engine** is an analytics system designed to analyze large volumes of internal corporate communications and extract meaningful insights.

The system uses **natural language processing, embeddings, clustering, and generative AI** to identify communication themes, detect redundancy, and generate executive-level summaries.

This project demonstrates how AI can transform unstructured internal communication data into actionable insights for leadership teams.

---

# Project Motivation

Large organizations generate thousands of internal messages daily through email, Slack, and other communication platforms.

Understanding communication trends across teams can help organizations:

- identify emerging topics
- detect redundant messaging
- measure communication efficiency
- surface important operational themes

However, manually analyzing large volumes of messages is impractical.

This project explores how **AI-driven analysis pipelines** can automatically extract insights from communication data.

---

# Dataset

The project uses the **Enron Email Dataset** as a proxy for corporate internal communication data.

The dataset contains thousands of emails exchanged within the Enron organization.

For this project:

- emails are cleaned and filtered
- irrelevant messages are removed
- meaningful communication content is extracted
- the dataset is used for unsupervised analysis

The dataset is treated as an **analysis corpus**, not a training dataset.

---

# Core Pipeline

The analysis pipeline consists of the following stages:

### 1. Data Loading

Email data is loaded and parsed from the dataset.

Key fields extracted include:

- sender
- recipient
- timestamp
- subject
- message body

---

### 2. Data Cleaning

Noise is removed from the dataset, including:

- automated system messages
- email signatures
- reply chains
- empty messages

This ensures that downstream analysis focuses only on meaningful communication content.

---

### 3. Semantic Embeddings

Each message is converted into vector embeddings using a sentence transformer model.

Embedding models allow semantic similarity to be measured between communications.

Example model:
sentence-transformers/all-MiniLM-L6-v2


---

### 4. Dimensionality Reduction

High-dimensional embeddings are reduced using **UMAP** to enable clustering.

This step preserves semantic structure while making clustering computationally feasible.

---

### 5. Clustering

Messages are grouped into clusters using **HDBSCAN**.

HDBSCAN identifies groups of semantically similar communications without requiring a predefined number of clusters.

Clusters represent recurring communication themes across the organization.

---

### 6. AI Insight Generation

Generative AI is used to interpret each cluster and produce human-readable insights such as:

- topic summaries
- recurring issues
- communication trends
- potential operational signals

These summaries transform raw clusters into **executive-level insights**.

---

# Dashboard Interface

The results are presented through a **Streamlit analytics dashboard**.

The dashboard allows users to:

- explore discovered communication themes
- view cluster summaries
- inspect representative messages
- identify redundant communications
- understand emerging trends

---

# Technologies Used

- Python
- Pandas
- Sentence Transformers
- UMAP
- HDBSCAN
- FAISS
- Streamlit
- Local LLMs (Ollama)
- Generative AI for insight summaries

---

# Architecture
Email Dataset
│
▼
Data Cleaning & Filtering
│
▼
Text Embeddings
│
▼
Dimensionality Reduction (UMAP)
│
▼
Clustering (HDBSCAN)
│
▼
Theme Detection
│
▼
Generative AI Insight Generation
│
▼
Streamlit Dashboard


---

# Example Insights Generated

Examples of insights the system can produce:

- recurring operational topics across teams
- communication overload signals
- redundant internal messaging
- emerging business discussions

These insights help leadership teams better understand internal communication patterns.

---

# Future Improvements

Possible extensions include:
- real-time Slack/email ingestion
- trend detection over time
- anomaly detection in communication patterns
- knowledge graph extraction
- integration with enterprise communication platforms

# Author
Kaif Nadaf
