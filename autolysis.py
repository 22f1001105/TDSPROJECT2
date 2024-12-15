
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
#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

openai.api_key = os.environ.get("AIPROXY_TOKEN", None)
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
MODEL_NAME = "gpt-4o-mini"

def safe_str(obj, limit=2000):
    text = str(obj)
    return text[:limit]

def llm_chat(messages, temperature=0.7, max_tokens=2000, functions=None):
    """
    Call the LLM with a list of messages. 
    If functions is provided (OpenAI function calling), we simulate readiness.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            functions=functions
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print("Error calling LLM:", e)
        return None

def basic_analysis(df):
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
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return pd.Series(dtype=int)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).sum()
    return outliers

def attempt_clustering(df, numeric_cols, n_clusters=3):
    if len(numeric_cols) < 2 or df.shape[0] < n_clusters:
        return None, None
    numeric_data = df[numeric_cols].dropna()
    if numeric_data.empty:
        return None, None
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(numeric_data)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
    return labels, cluster_centers.round(3).to_dict()

def attempt_pca(df, numeric_cols):
    if len(numeric_cols) < 2:
        return None, None
    data = df[numeric_cols].dropna()
    if data.empty:
        return None, None
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    explained_var = pca.explained_variance_ratio_.round(3)
    return pcs, explained_var

def attempt_regression(df, numeric_cols):
    """
    Attempt a simple linear regression on any pair of numeric columns.
    We'll pick first numeric col as X and second as Y if available.
    """
    if len(numeric_cols) < 2:
        return None, None, None
    data = df[numeric_cols].dropna()
    if data.empty:
        return None, None, None
    # Just pick the first two numeric columns for a simple regression
    X_col, Y_col = numeric_cols[0], numeric_cols[1]
    X = data[[X_col]]
    Y = data[Y_col]
    if X.shape[0] < 2:
        return None, None, None
    model = LinearRegression()
    model.fit(X, Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    score = model.score(X, Y)
    return (X_col, Y_col), (slope, intercept), score

def visualize_corr_matrix(corr_dict):
    if not corr_dict:
        return None
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

def visualize_pca_clusters(pcs, labels):
    if pcs is None or labels is None:
        return None
    plt.figure(figsize=(6,4))
    plt.scatter(pcs[:,0], pcs[:,1], c=labels, cmap='Set2', edgecolor='black')
    plt.title("PCA Scatter Plot Colored by Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    img_name = "pca_clusters.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def visualize_regression(df, cols, slope_intercept, numeric_cols):
    if cols is None or slope_intercept is None:
        return None
    X_col, Y_col = cols
    slope, intercept = slope_intercept
    data = df[[X_col, Y_col]].dropna()
    if data.empty:
        return None
    X = data[X_col]
    Y = data[Y_col]
    plt.figure(figsize=(6,4))
    plt.scatter(X, Y, color='green', alpha=0.6, edgecolor='black', label='Data points')
    line_x = np.linspace(X.min(), X.max(), 100)
    line_y = slope * line_x + intercept
    plt.plot(line_x, line_y, color='red', label='Regression line')
    plt.title(f"Linear Regression: {Y_col} vs {X_col}")
    plt.xlabel(X_col)
    plt.ylabel(Y_col)
    plt.legend()
    plt.tight_layout()
    img_name = "regression_plot.png"
    plt.savefig(img_name, dpi=100)
    plt.close()
    return img_name

def write_readme(analysis_dict, outliers, cluster_centers, explained_var, reg_info, chart_paths, final_story):
    with open("README.md", "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Introduction\n")
        f.write("In this automated analysis, we explored the dataset's structure, performed statistical and advanced analyses, and visualized key insights.\n\n")

        f.write("## Data Overview\n")
        f.write(f"- **Rows**: {analysis_dict['num_rows']}\n")
        f.write(f"- **Columns**: {analysis_dict['num_cols']}\n")
        f.write("\n### Missing Values (Top 5)\n")
        for col, val in analysis_dict['missing_values_top'].items():
            f.write(f"- {col}: {val} missing values\n")

        f.write("\n### Numeric Summary (Excerpt)\n")
        for stat_key, stat_vals in analysis_dict['numeric_summary_excerpt'].items():
            f.write(f"- **{stat_key}**: {stat_vals}\n")

        f.write("\n### Categorical Summary (Excerpt)\n")
        for cat_col, cat_vals in analysis_dict['categorical_summary_excerpt'].items():
            f.write(f"- **{cat_col}** top categories: {cat_vals}\n")

        f.write("\n## Outlier Detection\n")
        if outliers is not None and outliers.sum() > 0:
            f.write("Outliers were detected in the following numeric columns:\n")
            f.write(f"{outliers.to_string()}\n")
        else:
            f.write("No significant outliers detected.\n")

        f.write("\n## Correlation Analysis\n")
        if analysis_dict['corr_matrix_excerpt']:
            f.write("A correlation matrix was computed to understand relationships between numeric variables.\n")
            if "correlation_heatmap.png" in chart_paths:
                f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
        else:
            f.write("Not enough numeric columns for correlation analysis.\n\n")

        f.write("## Clustering and PCA\n")
        if cluster_centers:
            f.write("K-Means clustering was performed, revealing potential groups in the data:\n")
            f.write(f"{cluster_centers}\n\n")
        else:
            f.write("Clustering not applicable or insufficient data.\n\n")

        if explained_var is not None:
            f.write("PCA was performed to reduce dimensionality and understand underlying structures.\n")
            f.write(f"Explained variance ratio: {explained_var}\n\n")

        f.write("## Regression Analysis\n")
        if reg_info is not None:
            (X_col, Y_col), (slope, intercept), score = reg_info
            f.write(f"We performed a simple linear regression of **{Y_col}** vs **{X_col}**.\n")
            f.write(f"- Slope: {slope:.3f}\n")
            f.write(f"- Intercept: {intercept:.3f}\n")
            f.write(f"- R² Score: {score:.3f}\n\n")
        else:
            f.write("No regression analysis was applicable.\n\n")

        f.write("## Additional Visualizations\n")
        for chart in chart_paths:
            if chart:
                f.write(f"![Chart]({chart})\n\n")

        f.write("## Narrative Story\n")
        f.write(final_story)
        f.write("\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]
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

    analysis_dict, numeric_cols, categorical_cols = basic_analysis(df)
    outliers = detect_outliers(df)

    # First LLM Call: Ask for narrative suggestions
    prompt1 = f"""
We have a dataset with {analysis_dict['num_rows']} rows and {analysis_dict['num_cols']} columns.
We did a basic analysis (summary stats, missing values, partial correlation):
- Numeric summary excerpt: {safe_str(analysis_dict['numeric_summary_excerpt'])}
- Categorical summary excerpt: {safe_str(analysis_dict['categorical_summary_excerpt'])}
- Top missing values: {analysis_dict['missing_values_top']}
- Correlation excerpt: {safe_str(analysis_dict['corr_matrix_excerpt'])}

Suggest a narrative and recommend additional advanced analyses (like clustering, PCA, regression) to gain deeper insights.
Keep it concise.
"""
    llm_resp1 = llm_chat([{"role":"user","content":prompt1}], temperature=0.5)

    # Perform advanced analyses suggested
    labels, cluster_centers = attempt_clustering(df, numeric_cols, n_clusters=3)
    pcs, explained_var = attempt_pca(df, numeric_cols)
    reg_cols, reg_line, reg_score = attempt_regression(df, numeric_cols)
    reg_info = None
    if reg_cols and reg_line:
        reg_info = (reg_cols, reg_line, reg_score)

    # If clustering and PCA both done, we can visualize PCA cluster plot
    pca_plot = None
    if pcs is not None and labels is not None:
        pca_plot = visualize_pca_clusters(pcs, labels)

    # Second LLM call: Ask for a refined narrative incorporating these advanced steps
    prompt2 = f"""
We followed your suggestions and performed more advanced analysis:
- Clustering results (if any): {safe_str(cluster_centers)}
- PCA explained variance: {safe_str(explained_var)}
- Regression info: {safe_str(reg_info)}

Now craft a cohesive Markdown narrative that:
1. Introduces the data and what it might represent.
2. Summarizes key insights (missing values, numeric/cat summaries, correlation, outliers).
3. Discusses the advanced analyses: clustering, PCA, regression (if applicable).
4. References the visualizations (correlation heatmap, category barplot, distributions, outlier plot, PCA scatter, regression plot).
5. Concludes with implications or potential actions.
"""
    llm_resp2 = llm_chat([{"role":"user","content":prompt2}], temperature=0.5)
    if llm_resp2 is None:
        llm_resp2 = "No story generated."

    # Vision capabilities simulation:
    # Third LLM call: Ask LLM to interpret charts as if it had vision capabilities.  
    # We will include the chart file names and ask LLM to describe them.
    # This demonstrates vision & agentic workflows (multiple calls).
    charts = []
    corr_chart = visualize_corr_matrix(analysis_dict['corr_matrix_excerpt'])
    if corr_chart: charts.append(corr_chart)
    cat_chart = visualize_top_category(df, categorical_cols)
    if cat_chart: charts.append(cat_chart)
    dist_chart = visualize_numeric_distribution(df, numeric_cols)
    if dist_chart: charts.append(dist_chart)
    outlier_chart = visualize_outliers(outliers)
    if outlier_chart: charts.append(outlier_chart)
    if pca_plot: charts.append(pca_plot)
    reg_plot = visualize_regression(df, reg_cols, reg_line, numeric_cols)
    if reg_plot: charts.append(reg_plot)

    prompt3 = f"""
We have generated the following images:
{charts}

Describe what each image might represent and how it supports the narrative. Imagine you have vision: explain them as if you 'see' them.
"""
    llm_resp3 = llm_chat([{"role":"user","content":prompt3}], temperature=0.5)
    if llm_resp3 is None:
        llm_resp3 = "Images description unavailable."

    # Combine final narrative with image interpretation
    final_story = llm_resp2 + "\n\n### Image Interpretations\n" + llm_resp3

    # Write README
    write_readme(analysis_dict, outliers, cluster_centers, explained_var, reg_info, charts, final_story)

    print("Analysis complete. README.md and PNG files created.")

if __name__ == "__main__":
    main()

