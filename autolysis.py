#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import openai
import matplotlib.pyplot as plt
import seaborn as sns

# Set up OpenAI with the AI Proxy
openai.api_key = os.environ.get("AIPROXY_TOKEN")

print("AIPROXY_TOKEN from env:", os.environ.get("AIPROXY_TOKEN"))

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"


openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
MODEL_NAME = "gpt-4o-mini"

def safe_str(obj, limit=2000):
    # Limit string size sent to LLM to avoid large token usage
    text = str(obj)
    return text[:limit]

def llm_chat(prompt, temperature=0.7, max_tokens=2000):
    """
    A helper function to call the LLM (gpt-4o-mini) via AI Proxy.
    Returns the LLM's response as a string or None if there's an error.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
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

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    filename = sys.argv[1]

    # Load dataset
    try:
        df = pd.read_csv(filename, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        sys.exit(1)

    # Basic analysis
    num_rows, num_cols = df.shape
    col_info = df.dtypes.to_dict()
    missing_counts = df.isna().sum().to_dict()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    numeric_summary = {}
    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe().to_dict()

    categorical_summary = {}
    for c in categorical_cols:
        val_counts = df[c].value_counts(dropna=False).head(5).to_dict()
        categorical_summary[c] = val_counts

    corr_matrix = None
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(3)

    # Summarize for LLM
    summary_for_llm = {
        "filename": filename,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": {col: str(dtype) for col, dtype in col_info.items()},
        "missing_values_top": dict(sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        "numeric_summary_excerpt": {k: v for k, v in list(numeric_summary.items())[:3]},
        "categorical_summary_excerpt": {k: v for k, v in list(categorical_summary.items())[:3]}
    }

    # First LLM call: ask for a narrative and suggestions
    prompt = f"""
I have a dataset: {summary_for_llm}.
Given this summary, suggest a short narrative of what the data might represent, and propose one or two analytical steps that might yield insightful findings.
"""
    llm_response = llm_chat(prompt, temperature=0.5)

    # Create charts
    chart_paths = []

    # If correlation matrix exists, create a heatmap
    if corr_matrix is not None:
        plt.figure(figsize=(6,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        corr_path = "correlation_heatmap.png"
        plt.savefig(corr_path, dpi=100)
        plt.close()
        chart_paths.append(corr_path)

    # If we have a categorical column, show top categories
    if len(categorical_cols) > 0:
        first_cat = categorical_cols[0]
        top_values = df[first_cat].value_counts().head(5)
        plt.figure(figsize=(6,4))
        sns.barplot(x=top_values.index.astype(str), y=top_values.values)
        plt.title(f"Top categories in '{first_cat}'")
        plt.xlabel(first_cat)
        plt.ylabel("Count")
        plt.tight_layout()
        cat_chart_path = "top_categories.png"
        plt.savefig(cat_chart_path, dpi=100)
        plt.close()
        chart_paths.append(cat_chart_path)

    # If we have a numeric column, show a histogram
    if len(numeric_cols) > 0:
        first_num = numeric_cols[0]
        plt.figure(figsize=(6,4))
        sns.histplot(df[first_num], kde=True)
        plt.title(f"Distribution of '{first_num}'")
        plt.xlabel(first_num)
        plt.ylabel("Frequency")
        plt.tight_layout()
        hist_path = "numeric_distribution.png"
        plt.savefig(hist_path, dpi=100)
        plt.close()
        chart_paths.append(hist_path)

    # Final narrative prompt for README.md
    final_prompt = f"""
We started with a dataset named {filename} which has {num_rows} rows and {num_cols} columns.
We performed basic analyses: checked missing values, computed summary statistics for numeric columns, explored top categories for categorical columns, and computed a correlation matrix for numeric data.

We created charts:
{chart_paths}

Key observations:
- Missing values: {missing_counts}
- Numeric columns ({numeric_cols}) summary: {safe_str(numeric_summary)}
- Categorical columns ({categorical_cols}) top categories: {safe_str(categorical_summary)}
- Correlation matrix: {safe_str(corr_matrix)}

You suggested a narrative and additional steps earlier. Now, please craft a cohesive, story-like Markdown narrative that:
1. Introduces the data and what it might represent.
2. Summarizes the key insights uncovered from the analysis.
3. Discusses the visualizations (refer to images by name, e.g. ![Correlation Heatmap](correlation_heatmap.png)).
4. Concludes with implications or next steps.

Produce a concise but insightful narrative.
"""
    readme_content = llm_chat(final_prompt, temperature=0.5)

    if readme_content is None:
        readme_content = "# Analysis Results\n\nCould not generate narrative due to LLM error."

    with open("README.md", "w") as f:
        f.write(readme_content)

    print("Analysis complete. README.md and PNG files created.")

if __name__ == "__main__":
    main()
