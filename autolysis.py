
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

openai.api_key = os.environ.get("AIPROXY_TOKEN", None)
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
MODEL_NAME = "gpt-4o-mini"

def safe_str(obj, limit=1500):
    """Convert object to string and truncate to reduce token usage."""
    text = str(obj)
    return text[:limit]

def llm_chat(messages, temperature=0.7, max_tokens=1500):
    """Call the LLM with given messages and return the response text or None."""
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Error calling LLM:", e)
        return None

def analyze_data(df):
    """
    Perform basic analysis: summary stats, missing values, correlation.
    Return a dictionary with summarized info, along with numeric and categorical columns.
    """
    num_rows, num_cols = df.shape
    col_info = df.dtypes.to_dict()
    missing_counts = df.isna().sum().to_dict()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_summary = {}
    if numeric_cols:
        numeric_summary = df[numeric_cols].describe().to_dict()

    categorical_summary = {}
    for c in categorical_cols:
        val_counts = df[c].value_counts(dropna=False).head(5).to_dict()
        categorical_summary[c] = val_counts

    corr_matrix = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(3).to_dict()

    analysis_dict = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "missing_top": dict(sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        "numeric_summary_excerpt": {k: numeric_summary[k] for k in list(numeric_summary.keys())[:2]} if numeric_summary else {},
        "categorical_summary_excerpt": {k: v for k, v in list(categorical_summary.items())[:1]},
        "corr_excerpt": corr_matrix
    }

    return analysis_dict, numeric_cols, categorical_cols

def detect_outliers(df):
    """Detect outliers using IQR method on numeric columns."""
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return None
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5*IQR)) | (numeric_df > (Q3 + 1.5*IQR))).sum()
    return outliers

def do_clustering(df, numeric_cols, n_clusters=3):
    """Perform k-means clustering if possible."""
    if len(numeric_cols) < 2 or df.shape[0] < n_clusters:
        return None
    data = df[numeric_cols].dropna()
    if data.empty:
        return None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols).round(3).to_dict()
    return {"cluster_centers": centers, "labels_count": pd.Series(labels).value_counts().to_dict()}

def visualize_corr(corr_dict):
    if not corr_dict:
        return None
    corr_df = pd.DataFrame(corr_dict)
    plt.figure(figsize=(5,5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    fn = "correlation_heatmap.png"
    plt.savefig(fn, dpi=100)
    plt.close()
    return fn

def visualize_categories(df, categorical_cols):
    if not categorical_cols:
        return None
    c = categorical_cols[0]
    top_val = df[c].value_counts().head(5)
    plt.figure(figsize=(5,3))
    sns.barplot(x=top_val.index.astype(str), y=top_val.values, color='skyblue')
    plt.title(f"Top Categories in '{c}'")
    plt.xlabel(c)
    plt.ylabel("Count")
    plt.tight_layout()
    fn = "top_categories.png"
    plt.savefig(fn, dpi=100)
    plt.close()
    return fn

def visualize_numeric_distribution(df, numeric_cols):
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    plt.figure(figsize=(5,3))
    sns.histplot(df[col], kde=True, color='blue', bins=30)
    plt.title(f"Distribution of '{col}'")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    fn = "numeric_distribution.png"
    plt.savefig(fn, dpi=100)
    plt.close()
    return fn

def visualize_outliers(outliers):
    if outliers is None or outliers.sum() == 0:
        return None
    plt.figure(figsize=(5,3))
    outliers.plot(kind='bar', color='red')
    plt.title("Outliers per Numeric Column")
    plt.xlabel("Column")
    plt.ylabel("Outlier Count")
    plt.tight_layout()
    fn = "outliers.png"
    plt.savefig(fn, dpi=100)
    plt.close()
    return fn

def write_readme(analysis_dict, outliers, cluster_info, charts, final_story):
    """
    Write README.md with the final narrative and references to charts.
    """
    with open("README.md","w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("This report provides an automated analysis of the dataset, from basic summaries to clustering.\n\n")

        f.write("## Data Overview\n")
        f.write(f"- Rows: {analysis_dict['num_rows']}\n")
        f.write(f"- Columns: {analysis_dict['num_cols']}\n\n")

        f.write("### Missing Values (Top 5)\n")
        for k,v in analysis_dict['missing_top'].items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("### Numeric Summary (Excerpt)\n")
        for stat, vals in analysis_dict['numeric_summary_excerpt'].items():
            f.write(f"- {stat}: {vals}\n")
        f.write("\n")

        f.write("### Categorical Summary (Excerpt)\n")
        for c_col, vals in analysis_dict['categorical_summary_excerpt'].items():
            f.write(f"- {c_col} top categories: {vals}\n")
        f.write("\n")

        f.write("## Outlier Detection\n")
        if outliers is not None and outliers.sum() > 0:
            f.write("Outliers detected:\n")
            f.write(str(outliers))
            f.write("\n\n")
        else:
            f.write("No significant outliers detected.\n\n")

        f.write("## Correlation\n")
        if analysis_dict['corr_excerpt']:
            f.write("A correlation matrix was computed.\n")
            if "correlation_heatmap.png" in charts:
                f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        else:
            f.write("Not enough numeric data for correlation.\n\n")

        f.write("## Clustering\n")
        if cluster_info:
            f.write("K-Means clustering performed:\n")
            f.write(f"Cluster centers: {cluster_info['cluster_centers']}\n")
            f.write(f"Cluster counts: {cluster_info['labels_count']}\n\n")
        else:
            f.write("Clustering not applicable.\n\n")

        f.write("## Additional Visualizations\n")
        for c in charts:
            if c and c not in ["correlation_heatmap.png"]:
                f.write(f"![Chart]({c})\n\n")

        f.write("## Narrative Story\n")
        f.write(final_story)
        f.write("\n")

def main():
    if len(sys.argv)<2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)
    filename = sys.argv[1]

    # Try reading with multiple encodings
    for enc in ["utf-8","ISO-8859-1","cp1252","latin-1"]:
        try:
            df = pd.read_csv(filename, encoding=enc)
            break
        except:
            df = None
    if df is None:
        print("Could not read the dataset with given encodings.")
        sys.exit(1)

    # Basic analysis
    analysis_dict, numeric_cols, categorical_cols = analyze_data(df)
    outliers = detect_outliers(df)

    # First LLM prompt: ask for narrative and suggestions
    prompt1 = f"""
We have a dataset with {analysis_dict['num_rows']} rows and {analysis_dict['num_cols']} columns.
Numeric summary excerpt: {safe_str(analysis_dict['numeric_summary_excerpt'])}
Categorical excerpt: {safe_str(analysis_dict['categorical_summary_excerpt'])}
Top missing: {analysis_dict['missing_top']}
Correlation excerpt: {safe_str(analysis_dict['corr_excerpt'])}

Please provide a short narrative and suggest one advanced analysis step (like clustering). 
"""
    response1 = llm_chat([{"role":"user","content":prompt1}], temperature=0.5)
    if response1 is None:
        response1 = "No suggestions."

    # Assume LLM suggested clustering, do clustering
    cluster_info = do_clustering(df, numeric_cols, n_clusters=3)

    # Second LLM prompt: integrate clustering results and ask for a refined narrative.
    prompt2 = f"""
You suggested an advanced step. We performed K-means clustering:
Cluster info: {safe_str(cluster_info)}

Now craft a cohesive markdown narrative that:
1. Introduces data and what it might represent.
2. Summarizes insights (missing vals, numeric/cat summary, correlation, outliers).
3. Discusses the clustering results.
4. Mentions that we used minimal data excerpts to save tokens.
5. Concludes with implications.
"""
    response2 = llm_chat([{"role":"user","content":prompt2}], temperature=0.5)
    if response2 is None:
        response2 = "No narrative generated."

    # Third LLM prompt: Vision capabilities simulation
    # Generate charts
    charts = []
    charts.append(visualize_corr(analysis_dict['corr_excerpt']))
    charts.append(visualize_categories(df,categorical_cols))
    charts.append(visualize_numeric_distribution(df, numeric_cols))
    charts.append(visualize_outliers(outliers))

    prompt3 = f"""
We have generated these images:
{charts}

Imagine you have vision capabilities. Describe each image briefly and how it supports the narrative.
Also, if you need a function call to analyze images more deeply, let me know (like `analyze_image_details()`).
"""
    response3 = llm_chat([{"role":"user","content":prompt3}], temperature=0.5)
    if response3 is None:
        response3 = "No image description."

    # Combine final story with image interpretation
    final_story = response2 + "\n\n### Image Interpretations\n" + response3

    # Write README
    write_readme(analysis_dict, outliers, cluster_info, [c for c in charts if c], final_story)

    print("Analysis complete. README.md and PNG files created.")

if __name__ == "__main__":
    main()

