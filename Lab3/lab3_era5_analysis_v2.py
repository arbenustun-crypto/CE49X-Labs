#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab 3: ERA5 Weather Data Analysis (v2 — explainable)
Author: Arben Üstün (2024706018)
Date: 2025-10-23

Purpose
-------
Analyze ERA5 10 m wind for Berlin and Munich from hourly CSV files and produce:
1) Summary tables (monthly / seasonal / diurnal means, city stats, top-5 extremes)
2) Three figures:
   - fig_monthly_wind_speed.png
   - fig_seasonal_wind_speed.png
   - fig_diurnal_wind_speed_utc.png

Scientific notes (to answer typical questions)
----------------------------------------------
• Wind components: u10m (east–west), v10m (north–south), in m/s.
• Wind speed:  sqrt(u^2 + v^2).
• Wind direction (meteorological “from” direction, degrees):
    0° = from North, 90° = from East, increases clockwise.
  Computed via arctan2(-u, -v) then converted to degrees and wrapped to [0, 360).
• Seasons: DJF (Dec–Jan–Feb), MAM (Mar–Apr–May), JJA (Jun–Jul–Aug), SON (Sep–Oct–Nov).
• Time base: timestamps are treated as UTC to avoid DST ambiguities.

Usage
-----
  python lab3_era5_analysis_v2.py \
      --berlin_csv ./berlin_era5_wind_20241231_20241231.csv \
      --munich_csv ./munich_era5_wind_20241231_20241231.csv \
      --out_dir   ./labs/lab3

Inputs
------
CSV with (case-insensitive) columns:
    timestamp,u10m,v10m[,lat,lon]
Extra columns are ignored. If you omit “.csv” in a path, the script will try to append it.

Outputs
-------
- city_wind_speed_stats.csv
- monthly_wind_speed.csv
- seasonal_wind_speed.csv
- diurnal_wind_speed_utc.csv
- top5_hourly_extremes.csv  (for cross-checking with notable weather events)
- PNG figures listed above

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------- Configuration ---------------------------------

REQUIRED_COLS = ("timestamp", "u10m", "v10m")

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG for more detail
    format="%(levelname)s | %(message)s"
)


# ----------------------------- Utilities -------------------------------------

def _ensure_csv_path(p: Path) -> Path:
    """
    If the given path doesn't exist and has no suffix, also try with '.csv'.
    This lets the CLI accept --berlin_csv ./berlin_era5_wind_... (without .csv).
    """
    if p.exists():
        return p
    if p.suffix == "":
        trial = p.with_suffix(".csv")
        if trial.exists():
            return trial
    return p


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with stripped whitespace."""
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _validate_columns(df: pd.DataFrame, city: str) -> None:
    """Ensure all REQUIRED_COLS exist; raise a clear error if not."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{city}: Missing required columns {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force u10m/v10m to numeric.
    • Non-numeric values become NaN.
    • Later: rows with BOTH components missing are dropped; single missing -> 0.0.
    """
    out = df.copy()
    out["u10m"] = pd.to_numeric(out["u10m"], errors="coerce")
    out["v10m"] = pd.to_numeric(out["v10m"], errors="coerce")
    return out


# ----------------------------- Data Loading ----------------------------------

def load_city_csv(path: Path, city_name: str) -> pd.DataFrame:
    """
    Load and prepare one city's ERA5 CSV.

    Steps:
    1) Resolve path (optionally append .csv).
    2) Read CSV and standardize column names.
    3) Validate required columns.
    4) Parse timestamps as UTC.
    5) Coerce u/v to numeric, handle missing values.
    6) Compute wind_speed and meteorological wind_dir_deg.
    """
    p = _ensure_csv_path(Path(path))
    if not p.exists():
        raise FileNotFoundError(f"{city_name}: File not found at '{path}'. "
                                f"Tried '{p}'. Please check the path.")

    logging.info(f"Loading {city_name} from: {p}")
    df = pd.read_csv(p)

    # Standardize columns and check
    df = _normalize_columns(df)
    _validate_columns(df, city_name)

    # Parse timestamps in UTC; fail early on unparseable rows
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        # show a few problematic indices so it’s actionable
        bad_rows = df.index[df["timestamp"].isna()].tolist()[:5]
        raise ValueError(
            f"{city_name}: Unparsable timestamps at rows: {bad_rows} "
            f"(showing up to 5). Ensure ISO8601 or YYYY-MM-DD HH:MM format."
        )

    # Numeric conversion
    df = _coerce_numeric(df)

    # If both u and v are missing, we can't form a vector ⇒ drop those completely.
    both_missing = df["u10m"].isna() & df["v10m"].isna()
    dropped = int(both_missing.sum())
    if dropped:
        logging.warning(f"{city_name}: Dropping {dropped} rows with both u10m and v10m missing.")
    df = df.loc[~both_missing].copy()

    # If only one component is missing, treat the missing one as 0.0
    df["u10m"] = df["u10m"].fillna(0.0)
    df["v10m"] = df["v10m"].fillna(0.0)

    # --- Vector to speed & (meteorological) direction ---
    # Speed magnitude:
    df["wind_speed"] = np.sqrt(df["u10m"] ** 2 + df["v10m"] ** 2)

    # Meteorological “from” direction (degrees, clockwise from North):
    # arctan2 uses (y, x). For met direction, we invert components: arctan2(-u, -v).
    theta_rad = np.arctan2(-df["u10m"], -df["v10m"])
    theta_deg = np.degrees(theta_rad)
    df["wind_dir_deg"] = (theta_deg + 360.0) % 360.0  # wrap to [0, 360)

    df["city"] = city_name

    # Keep rows sorted in time for neatness of plots and groupbys
    df = df.sort_values("timestamp").reset_index(drop=True)
    logging.info(f"{city_name}: Loaded {len(df)} rows after cleaning.")
    return df


# ----------------------------- Aggregations ----------------------------------

def month_to_season(m: int) -> str:
    """Map month index (1–12) to a climatological season label."""
    if m in (12, 1, 2):
        return "DJF (Winter)"
    if m in (3, 4, 5):
        return "MAM (Spring)"
    if m in (6, 7, 8):
        return "JJA (Summer)"
    return "SON (Autumn)"


def compute_aggregates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute:
      • monthly mean wind_speed per city
      • seasonal mean wind_speed per city
      • diurnal (by UTC hour) mean wind_speed per city
    """
    out = df.copy()
    out["year"] = out["timestamp"].dt.year
    out["month"] = out["timestamp"].dt.month
    out["month_name"] = out["timestamp"].dt.strftime("%b")
    out["hour_utc"] = out["timestamp"].dt.hour
    out["season"] = out["month"].map(month_to_season)

    monthly = (
        out.groupby(["city", "year", "month", "month_name"], as_index=False)["wind_speed"]
           .mean()
           .sort_values(["city", "year", "month"])
    )

    seasonal = (
        out.groupby(["city", "year", "season"], as_index=False)["wind_speed"]
           .mean()
           .sort_values(["city", "year", "season"])
    )

    diurnal = (
        out.groupby(["city", "hour_utc"], as_index=False)["wind_speed"]
           .mean()
           .sort_values(["city", "hour_utc"])
    )

    # Round for clean presentation
    for d in (monthly, seasonal, diurnal):
        d["wind_speed"] = d["wind_speed"].round(3)

    return monthly, seasonal, diurnal


def city_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-city descriptive statistics on wind_speed (m/s).
    Includes count to reflect effective sample size after cleaning.
    """
    stats = (
        df.groupby("city")["wind_speed"]
          .agg(mean="mean",
               std="std",
               min="min",
               p50="median",
               p95=lambda s: s.quantile(0.95),
               max="max",
               n="count")
          .reset_index()
          .round(3)
    )
    return stats


def top_extremes(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Extract top-N highest hourly wind speeds overall.
    Useful for cross-checking with news reports of storms or high-wind days.
    """
    cols = ["timestamp", "city", "u10m", "v10m", "wind_speed", "wind_dir_deg"]
    out = df.sort_values("wind_speed", ascending=False).head(top_n)[cols].copy()
    out["date"] = out["timestamp"].dt.date
    return out.reset_index(drop=True)


# ----------------------------- Plotting --------------------------------------

def plot_monthly(monthly: pd.DataFrame, out_dir: Path) -> None:
    """
    Line plot of monthly mean wind_speed per city across (year, month).
    X-axis strings are YYYY-MM for readability across years.
    """
    plt.figure()
    for city in sorted(monthly["city"].unique()):
        sub = monthly[monthly["city"] == city]
        x = sub.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1)
        y = sub["wind_speed"].values
        plt.plot(x, y, marker="o", label=city)
    plt.title("Monthly Mean 10 m Wind Speed (UTC)")
    plt.xlabel("Month (YYYY-MM)")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_monthly_wind_speed.png", dpi=150)
    plt.close()


def plot_seasonal(seasonal: pd.DataFrame, out_dir: Path) -> None:
    """
    Grouped bars of seasonal mean wind_speed averaged over years (per city).
    Bar order is fixed to DJF, MAM, JJA, SON for climatological readability.
    """
    order = ["DJF (Winter)", "MAM (Spring)", "JJA (Summer)", "SON (Autumn)"]
    seasonal_avg = seasonal.groupby(["city", "season"], as_index=False)["wind_speed"].mean()
    x = np.arange(len(order))
    width = 0.35

    # Get values in a consistent order; if a season is missing, use NaN (plot will leave a gap).
    def vals_for(city: str):
        return [
            seasonal_avg.query("city == @city and season == @s")["wind_speed"].mean()
            if not seasonal_avg.query("city == @city and season == @s").empty else np.nan
            for s in order
        ]

    ber_vals = vals_for("Berlin")
    mun_vals = vals_for("Munich")

    plt.figure()
    plt.bar(x - width/2, ber_vals, width, label="Berlin")
    plt.bar(x + width/2, mun_vals, width, label="Munich")
    plt.xticks(x, order)
    plt.title("Seasonal Mean 10 m Wind Speed (UTC)")
    plt.xlabel("Season")
    plt.ylabel("Wind speed (m/s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_seasonal_wind_speed.png", dpi=150)
    plt.close()


def plot_diurnal(diurnal: pd.DataFrame, out_dir: Path) -> None:
    """
    Line plot of mean diurnal cycle of wind_speed (by UTC hour) for each city.
    """
    plt.figure()
    for city in sorted(diurnal["city"].unique()):
        sub = diurnal[diurnal["city"] == city]
        plt.plot(sub["hour_utc"], sub["wind_speed"], marker="o", label=city)
    plt.title("Diurnal Cycle of 10 m Wind Speed (UTC)")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Wind speed (m/s)")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_diurnal_wind_speed_utc.png", dpi=150)
    plt.close()


# ----------------------------- Main Program ----------------------------------

def main() -> None:
    """
    CLI entrypoint:
      1) Load Berlin & Munich CSVs.
      2) Compute aggregates (monthly/seasonal/diurnal).
      3) Save tables and plots to --out_dir.
      4) Print a short summary with a tip to cross-check extremes.
    """
    parser = argparse.ArgumentParser(
        description="ERA5 Weather Data Analysis (v2 — explainable)",
        epilog=(
            "Notes:\n"
            " • Input columns matched case-insensitively. Extra columns are ignored.\n"
            " • Timestamps are parsed as UTC to avoid DST ambiguity.\n"
            " • If you omit the .csv suffix in paths, the script will also try with '.csv'.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--berlin_csv", type=Path, required=True, help="Path to Berlin CSV (timestamp,u10m,v10m[,lat,lon])")
    parser.add_argument("--munich_csv", type=Path, required=True, help="Path to Munich CSV (timestamp,u10m,v10m[,lat,lon])")
    parser.add_argument("--out_dir", type=Path, default=Path("./labs/lab3"), help="Directory to save outputs (default: ./labs/lab3)")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {out.resolve()}")

    # Load and combine
    berlin = load_city_csv(args.berlin_csv, "Berlin")
    munich = load_city_csv(args.munich_csv, "Munich")
    df = pd.concat([berlin, munich], ignore_index=True)

    # Aggregations
    monthly, seasonal, diurnal = compute_aggregates(df)

    # Descriptive stats and extremes
    stats = city_summary_stats(df)
    extremes = top_extremes(df, top_n=5)

    # Save tables
    stats.to_csv(out / "city_wind_speed_stats.csv", index=False)
    monthly.to_csv(out / "monthly_wind_speed.csv", index=False)
    seasonal.to_csv(out / "seasonal_wind_speed.csv", index=False)
    diurnal.to_csv(out / "diurnal_wind_speed_utc.csv", index=False)
    extremes.to_csv(out / "top5_hourly_extremes.csv", index=False)
    logging.info("Saved CSV tables.")

    # Plots
    plot_monthly(monthly, out)
    plot_seasonal(seasonal, out)
