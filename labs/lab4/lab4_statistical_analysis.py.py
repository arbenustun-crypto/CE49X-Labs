"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions

This script is designed to be run from:
    CE49X-Fall25/labs/lab4/lab4_statistical_analysis.py

It assumes datasets are in:
    CE49X-Fall25/datasets/

Datasets and columns:
- concrete_strength.csv   -> batch_id, age_days, mix_type, strength_mpa
- material_properties.csv -> material_type, test_number, yield_strength_mpa
- structural_loads.csv    -> timestamp, load_kN, component_type

Main goals:
- Compute descriptive statistics for concrete strength.
- Fit a normal distribution to concrete strength.
- Compare yield strengths across different materials.
- Model engineering probability scenarios (Binomial, Poisson, Normal, Exponential).
- Apply Bayes' theorem to a damage detection scenario.
- Generate plots and a text report summarizing the results.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom, poisson, expon

# Use a simple whitegrid style for all plots
sns.set(style="whitegrid")


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data(file_name: str) -> pd.DataFrame | None:
    """
    Load dataset from CSV file in the root-level datasets/ folder.

    Because this script sits in `labs/lab4/`, we go two levels up to
    reach the repository root:

        this file:      CE49X-Fall25/labs/lab4/lab4_statistical_analysis.py
        parents[0] ->   lab4
        parents[1] ->   labs
        parents[2] ->   CE49X-Fall25 (repo root)

    Then we append "datasets" and the file name.

    Example
    -------
    load_data("concrete_strength.csv")
    -> loads CE49X-Fall25/datasets/concrete_strength.csv
    """
    try:
        # Resolve the path of THIS script
        repo_root = Path(__file__).resolve().parents[2]
        datasets_dir = repo_root / "datasets"

        csv_path = datasets_dir / file_name
        if not csv_path.exists():
            print(f"[load_data] File not found: {csv_path}")
            return None

        df = pd.read_csv(csv_path)
        print(f"[load_data] Loaded '{file_name}' with shape {df.shape}")
        return df

    except Exception as e:
        print(f"[load_data] Error loading {file_name}: {e}")
        return None


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def calculate_descriptive_stats(data: pd.DataFrame,
                                column: str = "strength_mpa") -> pd.Series:
    """
    Compute standard descriptive statistics for a single numeric column.

    Includes:
    - count, mean, median, mode
    - min, max, range
    - variance, standard deviation
    - quartiles (Q1, Q2/median, Q3)
    - interquartile range (IQR)
    - skewness (shape asymmetry)
    - kurtosis (tail heaviness)

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the column.
    column : str
        Name of the numeric column to analyze.

    Returns
    -------
    pd.Series
        Series containing all computed statistics.
    """
    # Drop missing values to avoid NaN issues
    s = data[column].dropna()

    # Pandas mode() can return multiple values; we take the first one
    mode_vals = s.mode()
    mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan

    stats_dict = {
        "count": s.count(),
        "mean": s.mean(),
        "median": s.median(),
        "mode": mode_val,
        "min": s.min(),
        "max": s.max(),
        "range": s.max() - s.min(),
        "variance": s.var(ddof=1),      # sample variance (n-1)
        "std_dev": s.std(ddof=1),       # sample std dev (n-1)
        "q1": s.quantile(0.25),
        "q2_median": s.quantile(0.50),
        "q3": s.quantile(0.75),
        "iqr": s.quantile(0.75) - s.quantile(0.25),
        "skewness": s.skew(),           # positive -> right tail
        "kurtosis": s.kurtosis()        # >0 heavy tails, <0 light tails
    }

    result = pd.Series(stats_dict)
    print(f"[calculate_descriptive_stats] Stats for {column}:\n{result}")
    return result


def plot_distribution(data: pd.DataFrame,
                      column: str,
                      title: str,
                      save_path: str | None = None) -> None:
    """
    Plot histogram + KDE for a numeric column and mark:
    - mean
    - ±1σ, ±2σ, ±3σ (useful to visually relate to the empirical rule)

    This is used for the concrete strength distribution.
    """
    s = data[column].dropna()

    # We reuse our descriptive stats function to get mean & std
    desc = calculate_descriptive_stats(data, column)
    mean = desc["mean"]
    std = desc["std_dev"]

    plt.figure(figsize=(8, 5))

    # Histogram + kernel density estimate (continuous smoothed curve)
    sns.histplot(s, kde=True, stat="density")
    plt.axvline(mean, linestyle="--", label=f"Mean = {mean:.2f}")

    # Mark ±1σ, ±2σ, ±3σ around the mean
    for k, ls in zip([1, 2, 3], [":", "-.", (0, (5, 5))]):
        plt.axvline(mean + k * std, linestyle=ls, alpha=0.7,
                    label=f"+{k}σ = {mean + k * std:.2f}")
        plt.axvline(mean - k * std, linestyle=ls, alpha=0.7)

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_distribution] Saved figure to {save_path}")
    plt.close()


def plot_boxplot(data: pd.DataFrame,
                 column: str,
                 title: str,
                 save_path: str | None = None) -> None:
    """
    Simple boxplot for a single numeric column.
    Shows:
    - median line
    - Q1, Q3 (box)
    - whiskers & outliers
    """
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=data[column].dropna())
    plt.ylabel(column)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_boxplot] Saved figure to {save_path}")
    plt.close()


# ============================================================
# 3. DISTRIBUTION FITTING (NORMAL)
# ============================================================

def fit_distribution(data: pd.DataFrame,
                     column: str,
                     distribution_type: str = "normal") -> dict:
    """
    Fit a probability distribution to the given column.

    Currently supports:
    - "normal": fit μ and σ using MLE (via scipy.stats.norm.fit)

    Returns a dict describing the fitted parameters.
    """
    s = data[column].dropna()

    if distribution_type.lower() == "normal":
        # norm.fit returns MLE estimates of mean and std dev
        mu, sigma = norm.fit(s)
        print(f"[fit_distribution] Normal fit for {column}: μ={mu:.3f}, σ={sigma:.3f}")
        return {"type": "normal", "mean": mu, "std": sigma}

    # If someone asks a different dist, we raise an error (not used in this lab)
    raise ValueError(f"Unsupported distribution_type: {distribution_type}")


def plot_distribution_fitting(data: pd.DataFrame,
                              column: str,
                              fitted_dist: dict | None = None,
                              save_path: str | None = None) -> None:
    """
    Visualize the fitted distribution over the data:

    - Plot histogram of the real data (density).
    - Overlay the fitted normal PDF with the estimated μ and σ.

    This directly addresses the "Does your data follow a normal distribution?"
    question visually.
    """
    s = data[column].dropna()

    if fitted_dist is None:
        fitted_dist = fit_distribution(data, column, "normal")

    mu = fitted_dist["mean"]
    sigma = fitted_dist["std"]

    plt.figure(figsize=(8, 5))

    # Histogram (empirical distribution)
    sns.histplot(s, stat="density", bins=20, alpha=0.6, label="Real data")

    # Fitted normal PDF (theoretical distribution)
    x = np.linspace(s.min(), s.max(), 200)
    pdf = norm.pdf(x, mu, sigma)
    plt.plot(x, pdf, linewidth=2, label=f"Fitted Normal (μ={mu:.2f}, σ={sigma:.2f})")

    plt.title(f"Distribution Fitting for {column}")
    plt.xlabel(column)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_distribution_fitting] Saved figure to {save_path}")
    plt.close()


# ============================================================
# 4. PROBABILITY HELPER FUNCTIONS
# ============================================================

def calculate_probability_binomial(n: int, p: float, k: int) -> dict:
    """
    Compute Binomial PMF and CDF for given parameters.

    Binomial model:
    - n: number of trials (e.g., 100 components tested)
    - p: probability of "success" (e.g., defect probability)
    - k: number of successes

    Returns:
    - PMF at k: P(X = k)
    - CDF up to k: P(X <= k)
    """
    pmf = binom.pmf(k, n, p)
    cdf_leq_k = binom.cdf(k, n, p)
    print(f"[Binomial] n={n}, p={p}, k={k}: PMF={pmf:.4f}, CDF(≤k)={cdf_leq_k:.4f}")
    return {"pmf": pmf, "cdf_leq_k": cdf_leq_k}


def calculate_probability_normal(mean: float,
                                 std: float,
                                 x_lower: float | None = None,
                                 x_upper: float | None = None) -> float:
    """
    Compute probabilities under a Normal(μ, σ) distribution.

    Cases:
    - If only x_lower is given:  P(X >= x_lower)
    - If only x_upper is given:  P(X <= x_upper)
    - If both are given:         P(x_lower <= X <= x_upper)

    This is used in the steel yield strength example.
    """
    dist = norm(loc=mean, scale=std)

    if x_lower is not None and x_upper is None:
        prob = 1 - dist.cdf(x_lower)
        print(f"[Normal] P(X >= {x_lower}) = {prob:.4f}")
        return prob
    elif x_lower is None and x_upper is not None:
        prob = dist.cdf(x_upper)
        print(f"[Normal] P(X <= {x_upper}) = {prob:.4f}")
        return prob
    elif x_lower is not None and x_upper is not None:
        prob = dist.cdf(x_upper) - dist.cdf(x_lower)
        print(f"[Normal] P({x_lower} <= X <= {x_upper}) = {prob:.4f}")
        return prob
    else:
        raise ValueError("Provide at least one of x_lower or x_upper.")


def calculate_probability_poisson(lambda_param: float, k: int) -> dict:
    """
    Compute Poisson PMF and CDF for given λ and k.

    Poisson model:
    - λ (lambda_param): average rate (e.g., trucks/hour)
    - k: number of events

    Returns:
    - PMF at k: P(X = k)
    - CDF up to k: P(X <= k)
    """
    pmf = poisson.pmf(k, lambda_param)
    cdf_leq_k = poisson.cdf(k, lambda_param)
    print(f"[Poisson] λ={lambda_param}, k={k}: PMF={pmf:.4f}, CDF(≤k)={cdf_leq_k:.4f}")
    return {"pmf": pmf, "cdf_leq_k": cdf_leq_k}


def calculate_probability_exponential(mean: float, x: float) -> dict:
    """
    Exponential distribution: time-to-failure model.

    Parameterization here: mean = expected lifetime = scale parameter.

    Computes:
    - CDF at x: P(X <= x)  (failure before time x)
    - Survival: P(X > x)   (component still working after time x)
    """
    scale = mean
    cdf = expon.cdf(x, scale=scale)
    survival = 1 - cdf
    print(f"[Exponential] mean={mean}, x={x}: P(X<=x)={cdf:.4f}, P(X>x)={survival:.4f}")
    return {"cdf_leq_x": cdf, "survival_gt_x": survival}


def apply_bayes_theorem(prior: float,
                        sensitivity: float,
                        specificity: float) -> dict:
    """
    Apply Bayes' theorem in a diagnostic test scenario.

    Scenario:
    - "Damage" = structure has true damage.
    - Test result = positive/negative.

    Inputs:
    - prior      = P(Damage)
    - sensitivity = P(Test+ | Damage)
    - specificity = P(Test- | No Damage)

    We compute:
    - P(Test+)              = total probability of a positive test
    - posterior             = P(Damage | Test+)
    - false_positive_rate   = P(Test+ | No Damage)

    This directly answers: "Given a positive test, what is the probability of
    actual damage?" which is important for risk-based decisions.
    """
    p_damage = prior
    p_no_damage = 1 - p_damage

    # Conditional probabilities
    p_pos_given_damage = sensitivity
    p_neg_given_no_damage = specificity
    p_pos_given_no_damage = 1 - specificity

    # Law of total probability for P(Test+)
    p_positive = (p_pos_given_damage * p_damage +
                  p_pos_given_no_damage * p_no_damage)

    # Bayes' theorem for posterior
    posterior = (p_pos_given_damage * p_damage) / p_positive

    result = {
        "prior_damage": p_damage,
        "posterior_damage_given_positive": posterior,
        "p_positive": p_positive,
        "false_positive_rate": p_pos_given_no_damage
    }

    print("[Bayes] Prior damage = {:.3f}, Posterior damage | positive = {:.3f}"
          .format(p_damage, posterior))
    return result


# ============================================================
# 5. MATERIAL COMPARISON (YIELD STRENGTH BY TYPE)
# ============================================================

def plot_material_comparison(data: pd.DataFrame,
                             column: str,
                             group_column: str,
                             save_path: str | None = None) -> None:
    """
    Create a comparative boxplot for different materials.

    For this lab:
    - group_column = "material_type"
    - column       = "yield_strength_mpa"

    This shows how the yield strength distribution changes between
    materials (Steel, Concrete, Aluminum, Composite).
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=group_column, y=column, data=data)
    plt.title(f"{column} by {group_column}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_material_comparison] Saved figure to {save_path}")
    plt.close()


# ============================================================
# 6. GENERIC DISTRIBUTION PLOTS (PDF/PMF OVERVIEW)
# ============================================================

def plot_probability_distributions(save_path: str | None = None) -> None:
    """
    Create a summary figure showing:

    - Standard Normal PDF
    - Exponential PDF (mean = 1)
    - Binomial PMF (n=20, p=0.2)
    - Poisson PMF (λ=5)

    This is a visual comparison of different distribution families:
    - continuous vs discrete
    - symmetric (Normal) vs skewed (Exponential)
    - count models (Binomial, Poisson)
    """
    x_norm = np.linspace(-3, 3, 200)
    x_exp = np.linspace(0, 5, 200)
    k_vals = np.arange(0, 20)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Normal PDF
    axes[0, 0].plot(x_norm, norm.pdf(x_norm))
    axes[0, 0].set_title("Standard Normal PDF (μ=0, σ=1)")

    # Exponential PDF (mean=1)
    axes[0, 1].plot(x_exp, expon.pdf(x_exp, scale=1))
    axes[0, 1].set_title("Exponential PDF (mean=1)")

    # Binomial PMF n=20, p=0.2
    axes[1, 0].stem(k_vals, binom.pmf(k_vals, 20, 0.2))
    axes[1, 0].set_title("Binomial PMF (n=20, p=0.2)")

    # Poisson PMF λ=5
    axes[1, 1].stem(k_vals, poisson.pmf(k_vals, 5))
    axes[1, 1].set_title("Poisson PMF (λ=5)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_probability_distributions] Saved figure to {save_path}")
    plt.close()


def plot_bayes_tree_diagram(bayes_result: dict,
                            save_path: str | None = None) -> None:
    """
    Simple "tree-style" probability summary for the Bayes scenario.

    Instead of a full graphical tree, we place the key probabilities on
    a text-only figure, which is enough to explain the relationships.
    """
    prior = bayes_result["prior_damage"]
    posterior = bayes_result["posterior_damage_given_positive"]

    plt.figure(figsize=(6, 4))
    plt.axis("off")

    text = (
        f"P(Damage) = {prior:.3f}\n"
        f"P(Damage | Test +) = {posterior:.3f}\n"
        f"P(Test +) = {bayes_result['p_positive']:.3f}\n"
        f"False positive rate = {bayes_result['false_positive_rate']:.3f}"
    )

    plt.text(0.05, 0.8, "Bayes Theorem – Damage Detection", fontsize=12, weight="bold")
    plt.text(0.05, 0.6, text, fontsize=11)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[plot_bayes_tree_diagram] Saved figure to {save_path}")
    plt.close()


# ============================================================
# 7. REPORT GENERATION
# ============================================================

def create_statistical_report(data: dict,
                              output_file: str = "lab4_statistical_report.txt") -> None:
    """
    Create a text report summarizing key findings.

    The 'data' dict is expected to contain:
    - 'concrete_stats': pd.Series with descriptive stats for concrete strength.
    - 'material_stats': pd.DataFrame with stats by material type.
    - 'probability_results': dict with scenario probabilities.
    - 'bayes_result': dict with Bayes' theorem outputs.

    The report is written as a simple text file for easy reading and submission.
    """
    lines: list[str] = []

    lines.append("Lab 4 Statistical Report\n")
    lines.append("=" * 40 + "\n\n")

    # Concrete strength descriptive stats
    if "concrete_stats" in data and data["concrete_stats"] is not None:
        lines.append("Concrete Strength Descriptive Statistics\n")
        lines.append("-" * 40 + "\n")
        lines.append(str(data["concrete_stats"]) + "\n\n")

    # Material property stats (grouped by material_type)
    if "material_stats" in data and data["material_stats"] is not None:
        lines.append("Material Property Statistics (by material_type)\n")
        lines.append("-" * 40 + "\n")
        lines.append(str(data["material_stats"]) + "\n\n")

    # Probability calculations for engineering scenarios
    if "probability_results" in data and data["probability_results"] is not None:
        lines.append("Probability Scenarios\n")
        lines.append("-" * 40 + "\n")
        for name, val in data["probability_results"].items():
            lines.append(f"{name}: {val}\n")
        lines.append("\n")

    # Bayes theorem damage detection scenario
    if "bayes_result" in data and data["bayes_result"] is not None:
        lines.append("Bayes' Theorem – Structural Damage Detection\n")
        lines.append("-" * 40 + "\n")
        for k, v in data["bayes_result"].items():
            lines.append(f"{k}: {v}\n")
        lines.append("\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"[create_statistical_report] Report written to {output_file}")


# ============================================================
# 8. MAIN ORCHESTRATION
# ============================================================

def main() -> None:
    """
    Main function that ties all parts together.

    Steps:
    1. Load the three datasets.
    2. Perform concrete strength analysis (descriptive stats + fitting).
    3. Compare material yield strengths by type.
    4. Solve probability scenarios (Binomial, Poisson, Normal, Exponential).
    5. Apply Bayes' theorem to a damage detection test.
    6. Generate overview plots for distributions.
    7. Build a simple "dashboard" figure.
    8. Generate the textual report.
    """
    # ---------- Load datasets ----------
    concrete = load_data("concrete_strength.csv")          # strength_mpa
    loads = load_data("structural_loads.csv")              # load_kN (not deeply used here)
    materials = load_data("material_properties.csv")       # yield_strength_mpa

    # ---------- Task 1: Concrete Strength Analysis ----------
    concrete_stats = None
    if concrete is not None:
        # Compute descriptive statistics for compressive strength
        concrete_stats = calculate_descriptive_stats(concrete, "strength_mpa")

        # Histogram + KDE + ±σ
        plot_distribution(
            concrete,
            column="strength_mpa",
            title="Concrete Strength Distribution",
            save_path="concrete_strength_distribution.png",
        )

        # Boxplot for quartiles and outliers
        plot_boxplot(
            concrete,
            column="strength_mpa",
            title="Concrete Strength Boxplot",
            save_path="concrete_strength_boxplot.png",
        )

        # Fit a normal distribution to concrete strength and visualize it
        fit_info = fit_distribution(concrete, "strength_mpa", "normal")
        plot_distribution_fitting(
            concrete,
            column="strength_mpa",
            fitted_dist=fit_info,
            save_path="distribution_fitting.png",
        )

    # ---------- Task 2: Material Comparison ----------
    material_stats = None
    if materials is not None:
        # Group by material_type and summarize yield_strength_mpa
        material_stats = (
            materials
            .groupby("material_type")["yield_strength_mpa"]
            .agg(["count", "mean", "std", "min", "max"])
        )

        # Comparative boxplot across material types
        plot_material_comparison(
            materials,
            column="yield_strength_mpa",
            group_column="material_type",
            save_path="material_comparison_boxplot.png",
        )

    # ---------- Task 3: Probability Modeling ----------
    probability_results: dict[str, object] = {}

    # Binomial scenario: n=100, p=0.05, probability of 3 defects etc.
    probability_results["binomial_exact_3"] = calculate_probability_binomial(100, 0.05, 3)
    probability_results["binomial_leq_5"] = {
        "cdf_leq_5": binom.cdf(5, 100, 0.05)
    }

    # Poisson scenario: λ = 10 trucks/hour
    probability_results["poisson_exact_8"] = calculate_probability_poisson(10, 8)
    probability_results["poisson_gt_15"] = {
        "p_gt_15": 1 - poisson.cdf(15, 10)
    }

    # Normal scenario: steel yield strength with mean=250, std=15
    probability_results["normal_gt_280"] = {
        "p_gt_280": calculate_probability_normal(250, 15, x_lower=280)
    }
    probability_results["normal_95th_percentile"] = {
        "x_95": norm.ppf(0.95, loc=250, scale=15)
    }

    # Exponential scenario: component lifetime mean=1000 hours
    probability_results["exp_fail_before_500"] = calculate_probability_exponential(1000, 500)
    probability_results["exp_survive_beyond_1500"] = {
        "survival_gt_1500": calculate_probability_exponential(1000, 1500)["survival_gt_x"]
    }

    # ---------- Task 4: Bayes' Theorem Application ----------
    bayes_result = apply_bayes_theorem(
        prior=0.05,       # 5% base rate of damage
        sensitivity=0.95, # P(Test+ | Damage)
        specificity=0.90  # P(Test- | No Damage)
    )
    plot_bayes_tree_diagram(bayes_result, save_path="bayes_tree_diagram.png")

    # ---------- Probability distribution overview plots ----------
    plot_probability_distributions(save_path="probability_distributions.png")

    # ---------- Simple "dashboard" summary figure ----------
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.05, 0.8, "Statistical Summary Dashboard", fontsize=14, weight="bold")

    if concrete_stats is not None:
        plt.text(
            0.05, 0.55,
            f"Concrete Strength\n"
            f"Mean = {concrete_stats['mean']:.2f} MPa\n"
            f"Std dev = {concrete_stats['std_dev']:.2f} MPa\n"
            f"Skewness = {concrete_stats['skewness']:.2f}",
            fontsize=11,
        )

    if "normal_gt_280" in probability_results:
        plt.text(
            0.05, 0.3,
            f"P(Steel strength > 280 MPa)\n"
            f"= {probability_results['normal_gt_280']['p_gt_280']:.3f}",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig("statistical_summary_dashboard.png", dpi=300)
    plt.close()
    print("[main] Saved statistical_summary_dashboard.png")

    # ---------- Final Report ----------
    report_data = {
        "concrete_stats": concrete_stats,
        "material_stats": material_stats,
        "probability_results": probability_results,
        "bayes_result": bayes_result,
    }
    create_statistical_report(report_data, output_file="lab4_statistical_report.txt")


# Standard Python entry point guard
if __name__ == "__main__":
    main()
