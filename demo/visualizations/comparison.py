from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from plotnine import aes, annotate, geom_histogram, geom_hline, geom_line, ggplot, labs, theme, theme_minimal


def performance_comparison_report(
    model1_rewards: np.ndarray,
    model1_regrets: np.ndarray,
    model1_cumulative_regret: np.ndarray,
    model1_optimal_rates: np.ndarray,
    model2_rewards: np.ndarray,
    model2_regrets: np.ndarray,
    model2_cumulative_regret: np.ndarray,
    model2_optimal_rates: np.ndarray,
) -> str:
    """
    Build a text report comparing the performance between two models.

    Returns:
        str: Multi-line string with the performance comparison.

    """
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append("PERFORMANCE COMPARISON")
    lines.append("=" * 60)

    # Model 1
    lines.append("")
    lines.append("Model 1:")
    lines.append(f"  Mean reward: {model1_rewards.mean():.4f}")
    lines.append(f"  Mean regret: {model1_regrets.mean():.4f}")
    lines.append(f"  Total cumulative regret: {model1_cumulative_regret[-1]:.4f}")
    lines.append(f"  Final optimal arm rate: {model1_optimal_rates[-1] * 100:.2f}%")

    # Model 2
    lines.append("")
    lines.append("Model 2:")
    lines.append(f"  Mean reward: {model2_rewards.mean():.4f}")
    lines.append(f"  Mean regret: {model2_regrets.mean():.4f}")
    lines.append(f"  Total cumulative regret: {model2_cumulative_regret[-1]:.4f}")
    lines.append(f"  Mean optimal arm rate: {np.mean(model2_optimal_rates) * 100:.2f}%")

    # Improvement
    regret_reduction = (1 - model1_cumulative_regret[-1] / model2_cumulative_regret[-1]) * 100
    lines.append("")
    lines.append("Improvement:")
    lines.append(f"  Regret reduction: {regret_reduction:.2f}%")

    return "\n".join(lines)


def generate_comparison_plots(
    optimal_rate_dict: Dict[str, np.ndarray],
    methods: List[str],
    regrets_dict: Dict[str, np.ndarray],
    cumulative_regrets_dict: Dict[str, np.ndarray],
    rolling_regret_dict: Dict[str, np.ndarray],
    n_samples: int,
    n_rounds: int,
    n_arms: int,
    rolling_window: int,
) -> Tuple[ggplot, ggplot, ggplot, ggplot]:
    """
    Generate comparison plots for the performance of different methods.

    Args:
        optimal_rate_dict (Dict[str, np.ndarray]): A dictionary where the key is a method name (e.g., 'DQN') and the value is the optimal rate array for that method.
        methods (List[str]): List of method names corresponding to the optimal rates, regrets, and cumulative regrets.
        regrets_dict (Dict[str, np.ndarray]): A dictionary of instantaneous regrets for each method.
        cumulative_regrets_dict (Dict[str, np.ndarray]): A dictionary of cumulative regrets for each method.
        rolling_regret_dict (Dict[str, np.ndarray]): A dictionary of rolling regrets for each method.
        n_samples (int): The number of samples for the regret-related data.
        n_rounds (int): The number of rounds for the optimal rate-related data.
        n_arms (int): The number of arms in the environment.
        rolling_window (int): The number of samples for the rolling average data.

    Returns:
        Tuple[ggplot, ggplot, ggplot]: A tuple of three ggplot objects:
            - `p_cumulative`: Plot for cumulative regret over samples.
            - `p_optimal_rate`: Plot for optimal arm selection rate over rounds.
            - `p_regret`: Plot for the distribution of instantaneous regrets.

    """
    # Create comparison dataframes
    df_comparison_dict = _create_comparison_dataframes(
        optimal_rate_dict=optimal_rate_dict,
        methods=methods,
        regrets_dict=regrets_dict,
        rolling_regret_dict=rolling_regret_dict,
        cumulative_regrets_dict=cumulative_regrets_dict,
        n_samples=n_samples,
        n_rounds=n_rounds,
    )

    # Plot for cumulative regret
    p_cumulative_regret = (
        ggplot(df_comparison_dict["cumulative_regret"], aes(x="sample", y="cumulative_regret", color="method"))
        + geom_line(size=1)
        + labs(title=f'Cumulative Regret: {" vs ".join(methods)}', x="Sample", y="Cumulative Regret", color="Method")
        + theme_minimal()
        + theme(legend_position="bottom")
    )

    # Plot for optimal arm selection rate over time
    p_optimal_rate = (
        ggplot(df_comparison_dict["optimal_rate"], aes(x="round", y="optimal_rate", color="method"))
        + geom_line(size=1)
        + geom_hline(yintercept=1 / n_arms, linetype="dashed", color="gray")
        + annotate("text", x=n_rounds * 0.8, y=1 / n_arms + 0.05, label=f"Random baseline ({1 / n_arms:.2f})", color="gray", size=9)
        + labs(title="Optimal Arm Selection Rate Over Time", x="Round", y="Optimal Selection Rate", color="Method")
        + theme_minimal()
        + theme(legend_position="bottom")
    )

    # Plot for distribution of instantaneous regrets
    p_inst_regret = (
        ggplot(df_comparison_dict["regret"], aes(x="regret", fill="method"))
        + geom_histogram(bins=50, alpha=0.6, position="identity")
        + labs(title="Distribution of Instantaneous Regret", x="Instantaneous Regret", y="Frequency", fill="Method")
        + theme_minimal()
        + theme(legend_position="bottom")
    )

    p_rolling = (
        ggplot(df_comparison_dict["rolling_regret"], aes(x="sample", y="rolling_regret", color="method"))
        + geom_line(size=1)
        + labs(title=f"Rolling Average Regret (window={rolling_window})", x="Sample", y="Rolling Avg Regret", color="Method")
        + theme_minimal()
        + theme(legend_position="bottom")
    )

    return p_cumulative_regret, p_optimal_rate, p_inst_regret, p_rolling


def _create_comparison_dataframes(
    optimal_rate_dict: Dict[str, np.ndarray],
    methods: List[str],
    regrets_dict: Dict[str, np.ndarray],
    cumulative_regrets_dict: Dict[str, np.ndarray],
    rolling_regret_dict: Dict[str, np.ndarray],
    n_samples: int,
    n_rounds: int,
) -> Dict[str, pd.DataFrame]:
    """
    Create dataframes for comparison of different methods' performance metrics.

    Args:
        optimal_rate_dict (Dict[str, np.ndarray]): Dictionary where each key is a method name and the value is the corresponding optimal rate array.
        methods (List[str]): List of method names corresponding to the optimal_rate_arrays.
        regrets_dict (Dict[str, np.ndarray]): Dictionary containing regret arrays for each method.
        cumulative_regrets_dict (Dict[str, np.ndarray]): Dictionary containing cumulative regret arrays for each method.
        rolling_regret_dict (Dict[str, np.ndarray]): Dictionary containing rolling average regret arrays for each method.
        n_samples (int): The number of samples.
        n_rounds (int): The number of rounds.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing 'cumulative_regret', 'regret', and 'optimal_rate' dataframes.

    """
    # Create the cumulative regret dataframe
    df_cumulative = pd.DataFrame(
        {
            "sample": list(range(n_samples)) * len(methods),
            "cumulative_regret": np.concatenate([cumulative_regrets_dict[method] for method in methods]),
            "method": [method for method in methods for _ in range(n_samples)],
        }
    )

    # Create the regret dataframe
    df_regret = pd.DataFrame({"regret": np.concatenate([regrets_dict[method] for method in methods]), "method": [method for method in methods for _ in range(n_samples)]})

    # Create the optimal rate dataframe
    df_optimal_rate = pd.DataFrame(
        {
            "round": list(range(n_rounds)) * len(methods),
            "optimal_rate": np.concatenate([optimal_rate_dict[method] for method in methods]),
            "method": [method for method in methods for _ in range(n_rounds)],
        }
    )

    # Create the rolling average regret dataframe
    df_rolling = pd.DataFrame(
        {
            "sample": list(range(n_samples)) * len(methods),
            "rolling_regret": np.concatenate([rolling_regret_dict[method] for method in methods]),
            "method": [method for method in methods for _ in range(n_samples)],
        }
    ).dropna()

    return {"cumulative_regret": df_cumulative, "regret": df_regret, "optimal_rate": df_optimal_rate, "rolling_regret": df_rolling}
