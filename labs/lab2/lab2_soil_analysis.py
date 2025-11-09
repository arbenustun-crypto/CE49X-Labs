# CE 49X - Lab 2: Soil Test Data Analysis
# Student Name: Arben Üstün
# Student ID: 2024706018
# Date: October 16, 2025

import pandas as pd
import numpy as np
from pathlib import Path

# Filenames are centralized in constants so it's easy to change later if needed.
RAW_NAME = "soil_test.csv"
CLEAN_NAME = "soil_test_cleaned.csv"
STATS_NAME = "soil_test_stats.csv"


def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Load the soil test dataset from a CSV file with error handling.

    Parameters
    ----------
    file_path : str
        Path to the CSV file (expected columns: sample_id, soil_ph, nitrogen,
        phosphorus, moisture).

    Returns
    -------
    pd.DataFrame | None
        Returns a DataFrame if read succeeds; returns None if the file is missing
        or unreadable. We choose None (instead of raising) so main() can decide
        what to do next without crashing.
    """
    try:
        # read_csv will parse the comma-separated values into a DataFrame
        df = pd.read_csv(file_path)
        print(f"✅ Loaded: {file_path}")
        return df
    except FileNotFoundError:
        # Most common issue: path is wrong or file not in working directory
        print(f"[ERROR] File not found: {file_path}")
    except pd.errors.EmptyDataError:
        # File exists but has no content or cannot be parsed properly
        print(f"[ERROR] File is empty or unreadable: {file_path}")
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset to make it analysis-ready.

    Steps
    -----
    1) Convert target columns to numeric (coerce invalid strings to NaN).
    2) Fill missing values in ['soil_ph','nitrogen','phosphorus','moisture']
       with the column mean (simple baseline imputation).
    3) Remove soil_ph outliers beyond ±3σ (3 standard deviations from the mean).
       - We do this AFTER filling NaNs so mean/std are defined.

    Why the order matters
    ---------------------
    - If we try to compute mean/std before filling NaNs, we might get misleading
      results (or too few valid rows). Filling first stabilizes the stats.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where NaNs are imputed and extreme soil_ph outliers
        are excluded. Index is reset after filtering for clean downstream use.
    """
    df_clean = df.copy()

    # Columns we expect to be numeric for this lab. If any are missing in the CSV,
    # the "if col in df_clean.columns" guard prevents KeyErrors and keeps the run stable.
    cols = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']

    # --- Missing value handling (imputation with mean) ---
    for col in cols:
        if col in df_clean.columns:
            # Force numeric type; non-numeric strings become NaN (errors='coerce').
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

            # Compute the mean of the column ignoring NaNs.
            mean_val = df_clean[col].mean(skipna=True)

            # Fill NaNs in-place with the mean. This ensures no missing values remain.
            df_clean[col].fillna(mean_val, inplace=True)

    # --- Outlier removal in soil_ph using the ±3σ rule ---
    # Rationale:
    #   For distributions that are approximately normal, ~99.7% of data should lie
    #   within ±3 standard deviations. Values outside this range are rare and may
    #   reflect measurement or recording errors (or truly extreme samples).
    if 'soil_ph' in df_clean.columns:
        ph = df_clean['soil_ph']

        # Population std (ddof=0) is fine for this small exercise and consistent
        # with what we print later. Sample std (ddof=1) is also defendable.
        mu, sigma = ph.mean(), ph.std(ddof=0)

        # If sigma is zero, all values are identical; in that case, there are no
        # outliers to remove and filtering would be a no-op.
        if sigma and sigma > 0:
            lower, upper = mu - 3 * sigma, mu + 3 * sigma

            # Keep only rows whose soil_ph falls within [lower, upper].
            mask = (ph >= lower) & (ph <= upper)
            df_clean = df_clean[mask].reset_index(drop=True)

    return df_clean


def compute_statistics(df: pd.DataFrame, column: str) -> dict:
    """
    Compute descriptive statistics for a specific numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.
    column : str
        The numeric column to summarize (e.g., 'soil_ph').

    Returns
    -------
    dict
        Dictionary with keys: min, max, mean, median, std.

    Notes
    -----
    - We coerce the column to numeric to be resilient against stray text entries.
    - We use population std (ddof=0) for consistency with earlier choices.
    """
    if column not in df.columns:
        # Explicit error helps the caller understand what went wrong.
        raise KeyError(f"Column '{column}' not found.")

    # Convert to numeric; drop NaNs so stats are defined on valid values only.
    s = pd.to_numeric(df[column], errors="coerce").dropna()

    if s.empty:
        # If the column has no numeric data after coercion, we can't compute stats.
        raise ValueError(f"Column '{column}' has no numeric data.")

    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
    }


def compute_all_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for ALL numeric columns at once.

    Returns
    -------
    pd.DataFrame
        A tidy table with rows per numeric column and columns:
        [column, min, max, mean, median, std]

    Why this is useful
    ------------------
    - Gives a quick overview of the dataset health.
    - Ready to export to CSV for submission or reporting.
    """
    # Select only numeric dtypes; non-numeric columns (e.g., sample_id if string)
    # are automatically ignored.
    num = df.select_dtypes(include=[np.number])

    if num.empty:
        # If the dataset had no numeric columns, we return an empty DataFrame to
        # avoid writing a misleading stats file.
        return pd.DataFrame()

    stats = {
        "column": [],
        "min": [],
        "max": [],
        "mean": [],
        "median": [],
        "std": []
    }

    # Iterate each numeric column and compute basic stats.
    for col in num.columns:
        s = num[col].dropna()
        if s.empty:
            # Skip fully-NaN numeric columns (defensive programming).
            continue

        stats["column"].append(col)
        stats["min"].append(float(s.min()))
        stats["max"].append(float(s.max()))
        stats["mean"].append(float(s.mean()))
        stats["median"].append(float(s.median()))
        stats["std"].append(float(s.std(ddof=0)))

    return pd.DataFrame(stats)


def main():
    """
    Orchestrates the end-to-end workflow:
    1) Load raw CSV
    2) Clean data (NaNs -> mean, remove soil_ph outliers)
    3) Save cleaned CSV
    4) Compute and save stats for all numeric columns
    5) Print soil_ph stats to console for quick check
    """
    # Resolve the working directory to an absolute path (useful for prints).
    wd = Path(".").resolve()

    # Build full paths from constants; easy to change names in one place.
    raw_path = wd / RAW_NAME
    clean_path = wd / CLEAN_NAME
    stats_path = wd / STATS_NAME

    # -------------------------------
    # (1) LOAD
    # -------------------------------
    df = load_data(str(raw_path))
    if df is None:
        # Early return: no data to process.
        return

    # -------------------------------
    # (2) CLEAN
    # -------------------------------
    df_clean = clean_data(df)

    # -------------------------------
    # (3) SAVE CLEANED CSV
    # -------------------------------
    # index=False avoids adding a 'Unnamed: 0' column; cleaner for grading.
    df_clean.to_csv(clean_path, index=False)
    print(f"✅ Saved cleaned CSV -> {clean_path}")

    # -------------------------------
    # (4) STATS
    # -------------------------------
    # Print soil_ph stats to satisfy the "at least one numeric column" requirement.
    try:
        soil_stats = compute_statistics(df_clean, "soil_ph")
        print("\nDescriptive statistics for 'soil_ph' (cleaned):")
        print(f"  Minimum: {soil_stats['min']:.3f}")
        print(f"  Maximum: {soil_stats['max']:.3f}")
        print(f"  Mean:    {soil_stats['mean']:.3f}")
        print(f"  Median:  {soil_stats['median']:.3f}")
        print(f"  Std Dev: {soil_stats['std']:.3f}")
    except Exception as e:
        # Non-fatal: still proceed to write all-numeric stats below.
        print(f"[WARN] Could not compute soil_ph stats: {e}")

    # Compute stats for ALL numeric columns and save them as a CSV for submission.
    all_stats = compute_all_numeric_stats(df_clean)
    if not all_stats.empty:
        all_stats.to_csv(stats_path, index=False)
        print(f"✅ Saved stats CSV -> {stats_path}")
    else:
        print("[WARN] No numeric columns found for stats CSV.")

    # Optional: if you want to always keep a consistent raw copy next to outputs,
    # uncomment the next line. Not required by the lab, but sometimes convenient.
    # df.to_csv(raw_path, index=False)


if __name__ == "__main__":
    # The standard Python entrypoint. Keeps the module import-safe.
    main()


# =============================================================================
# REFLECTION QUESTIONS (ANSWERED)
# =============================================================================
# 1. What was the most challenging part of this lab?
# Answer: Getting the cleaning step right was trickier than expected. I had to be careful with NaNs and outliers so I didn’t throw away useful data or skew the
# results. Tuning the ±3σ rule and then checking how it moved the soil_ph mean and std took a few iterations.

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: It turns scattered field measurements into something we can use for decisions—like classifying sites, choosing foundation types, and flagging risks.
# Quick, reliable stats help us communicate and justify design choices to clients and stakeholders.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: Automatic detection of numeric columns and simple plots (histograms and boxplots) would make problems jump out visually. A command-line switch to toggle
# outlier handling and a short exportable report (PDF/HTML) would also streamline deliverables.

# 4. How did error handling improve the robustness of your code?
# Answer: Instead of crashing, the script now fails gracefully and tells me what
# went wrong—missing files, empty columns, or non-numeric data. That made testing
# smoother and the overall workflow more reliable.