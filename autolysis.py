
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

# Silence some warnings that can occur due to data variability
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

openai.api_key = os.environ.get("AIPROXY_TOKEN", None)
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
MODEL_NAME = "gpt-4o-mini"

def safe_str(obj, limit=2000):
    """Convert object to string safely and truncate to avoid large token usage."""
    text = str(obj)
    return text[:limit]

def llm_chat(prompt, temperature=0.7, max_tokens=2000):
    """
    A helper function to call the LLM (gpt-4o-mini) via AI Proxy.
    Returns the LLM's response as a string or None if there's an error.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that carefully follows user instructions."},
        {"role": "user", "content": prompt}
    ]

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

def basic_analysis(df):
    """
    Perform a basic analysis of the dataframe including summary statistics,
    missing values, correlation, and categorical frequency.
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

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": {col: str(dtype) for col, dtype in col_info.items()},
        "missing_values_top": dict(sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        "numeric_summary_excerpt": {k: numeric_summary[k] for k in list(numeric_summary.keys())[:3]} if numeric_summary else {},
        "categorical_summary_excerpt": {k: v for k, v in list(categorical_summary.items())[:2]},
        "corr_matrix_excerpt": corr_matrix
    }, numeric_cols, categorical_cols

def detect_outliers(df):
    """
    Detect outliers using IQR method on numeric columns.
    Returns a series with outlier counts per column.
    """
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.Series(dtype=int)

    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    return outliers

def attempt_clustering(df, numeric_cols, n_clusters=3):
    """
    Attempt K-Means clustering on numeric columns (if enough columns are available).
    Returns cluster labels and a summary of cluster means.
    """
    if len(numeric_cols) < 2 or df.shape[0] < n_clusters:
        return None, None

    numeric_data = df[numeric_cols].dropna()
    if numeric_data.empty:
        return None, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(numeric_data)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)

    return labels, cluster_centers.round(3).to_dict()

def visualize_corr_matrix(corr_dict):
    """
    Visualize the correlation matrix from a dictionary representation.
    Saves as correlation_heatmap.png
    """
    if not corr_dict:
        return None
    # Convert dict back to DataFrame
    corr_df = pd.DataFrame(corr_dict)
    plt.figure(figsize=(6,6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    img_name = "correlation_heatmap.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def visualize_top_category(df, categorical_cols):
    """
    Visualize top categories of the first categorical column if available.
    Saves as top_categories.png
    """
    if not categorical_cols:
        return None
    first_cat = categorical_cols[0]
    top_values = df[first_cat].value_counts().head(5)
    plt.figure(figsize=(6,4))
    sns.barplot(x=top_values.index.astype(str), y=top_values.values, color="skyblue")
    plt.title(f"Top Categories in '{first_cat}'")
    plt.xlabel(first_cat)
    plt.ylabel("Count")
    plt.tight_layout()
    img_name = "top_categories.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def visualize_numeric_distribution(df, numeric_cols):
    """
    Visualize distribution of the first numeric column if available.
    Saves as numeric_distribution.png
    """
    if not numeric_cols:
        return None
    first_num = numeric_cols[0]
    plt.figure(figsize=(6,4))
    sns.histplot(df[first_num], kde=True, color='blue', bins=30)
    plt.title(f"Distribution of '{first_num}'")
    plt.xlabel(first_num)
    plt.ylabel("Frequency")
    plt.tight_layout()
    img_name = "numeric_distribution.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def visualize_outliers(outliers):
    """
    Visualize outliers count per numeric column if any outliers detected.
    Saves as outliers.png
    """
    if outliers is None or outliers.sum() == 0:
        return None
    plt.figure(figsize=(6,4))
    outliers.plot(kind='bar', color='red')
    plt.title("Outliers per Numeric Column")
    plt.xlabel("Column")
    plt.ylabel("Number of Outliers")
    plt.tight_layout()
    img_name = "outliers.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def write_readme(analysis_dict, outliers, cluster_centers, chart_paths, final_story):
    """
    Write README.md with a structured narrative, referencing the charts and
    including the final story from the LLM.
    """
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("This report provides an automated analysis of the provided dataset. We examined its structure, performed statistical analyses, detected outliers, explored correlations, and even attempted clustering.\n\n")

        f.write("## Data Overview\n")
        f.write(f"- Rows: {analysis_dict['num_rows']}\n")
        f.write(f"- Columns: {analysis_dict['num_cols']}\n")
        f.write("### Missing Values (Top 5)\n")
        for col, val in analysis_dict['missing_values_top'].items():
            f.write(f"- {col}: {val} missing values\n")

        f.write("\n### Numeric Summary (Excerpt)\n")
        for stat_key, stat_vals in analysis_dict['numeric_summary_excerpt'].items():
            f.write(f"- **{stat_key}**: {stat_vals}\n")

        f.write("\n### Categorical Summary (Excerpt)\n")
        for cat_col, cat_vals in analysis_dict['categorical_summary_excerpt'].items():
            f.write(f"- **{cat_col}** top categories: {cat_vals}\n")

        f.write("\n## Outlier Detection\n")
        f.write("We detected outliers in numeric columns using the IQR method:\n")
        f.write(f"{outliers.to_string()}\n\n" if outliers is not None else "No numeric columns or no outliers.\n\n")

        f.write("## Correlation Analysis\n")
        if analysis_dict['corr_matrix_excerpt']:
            f.write("A correlation matrix was computed to understand relationships between numeric variables.\n")
            if "correlation_heatmap.png" in chart_paths:
                f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        else:
            f.write("Not enough numeric columns for correlation analysis.\n\n")

        f.write("## Clustering (Experimental)\n")
        if cluster_centers:
            f.write("We attempted K-Means clustering on numeric columns. Cluster centers:\n")
            f.write(f"{cluster_centers}\n\n")
        else:
            f.write("Clustering was not performed or not applicable.\n\n")

        f.write("## Additional Visualizations\n")
        for chart in chart_paths:
            if chart and chart != "correlation_heatmap.png":
                f.write(f"![Chart]({chart})\n\n")

        f.write("## Narrative Story\n")
        f.write(final_story)
        f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    # Load dataset with a fallback encoding
    encodings_to_try = ["utf-8", "ISO-8859-1", "cp1252", "latin-1"]
    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filename, encoding=enc)
            break
        except:
            pass
    if df is None:
        print(f"Error reading {filename} with tried encodings.")
        sys.exit(1)

    # Basic analysis
    analysis_dict, numeric_cols, categorical_cols = basic_analysis(df)
    outliers = detect_outliers(df)

    # First LLM call: Ask for narrative and additional steps
    prompt1 = f"""
We have a dataset with {analysis_dict['num_rows']} rows and {analysis_dict['num_cols']} columns.
We computed some summary stats and partial info:
Numeric summary excerpt: {safe_str(analysis_dict['numeric_summary_excerpt'])}
Categorical summary excerpt: {safe_str(analysis_dict['categorical_summary_excerpt'])}
Top missing values: {analysis_dict['missing_values_top']}
Correlation matrix excerpt: {safe_str(analysis_dict['corr_matrix_excerpt'])}

Please provide a short narrative of what this data might represent and suggest one or two additional analytical steps (like clustering or regression or outlier analysis) that could yield interesting insights.
"""
    llm_response = llm_chat(prompt1, temperature=0.5)
    if llm_response is None:
        llm_response = "No suggestions from LLM."

    # Suppose LLM suggested clustering or something else, attempt clustering if numeric data available
    labels, cluster_centers = attempt_clustering(df, numeric_cols, n_clusters=3)

    # Prepare a second prompt after attempting additional analysis
    prompt2 = f"""
We followed your suggestion and performed a clustering analysis on numeric columns (if possible).
Cluster centers: {safe_str(cluster_centers)}

We also have outliers detected: {safe_str(outliers.to_dict() if outliers is not None else 'No outliers')}

Now, please craft a cohesive, story-like Markdown narrative that:
1. Introduces the data and what it might represent.
2. Summarizes the key insights uncovered from the analysis (mention missing values, numeric/cat summaries, correlation, outliers, clustering).
3. Discusses the visualizations (like the correlation heatmap, top category barplot, numeric distributions, and outlier plot).
4. Concludes with implications or next steps.
"""
    final_story = llm_chat(prompt2, temperature=0.5)
    if final_story is None:
        final_story = "No story generated from LLM."

    # Generate visualizations
    chart_paths = []
    corr_chart = visualize_corr_matrix(analysis_dict['corr_matrix_excerpt'])
    if corr_chart:
        chart_paths.append(corr_chart)
    cat_chart = visualize_top_category(df, categorical_cols)
    if cat_chart:
        chart_paths.append(cat_chart)
    dist_chart = visualize_numeric_distribution(df, numeric_cols)
    if dist_chart:
        chart_paths.append(dist_chart)
    outliers_chart = visualize_outliers(outliers)
    if outliers_chart:
        chart_paths.append(outliers_chart)

    # Write README with the final narrative
    write_readme(analysis_dict, outliers, cluster_centers, chart_paths, final_story)

    print("Analysis complete. README.md and PNG files created.")

if __name__ == "__main__":
    main()
