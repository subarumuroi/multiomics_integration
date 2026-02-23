"""Visualization: shared plotting functions for all methods."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Consistent color scheme for banana ripeness
CLASS_COLORS = {"Green": "#2ca02c", "Ripe": "#ff7f0e", "Overripe": "#d62728"}
CLASS_ORDER = ["Green", "Ripe", "Overripe"]


def _setup_figure(figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_fig(fig, path, dpi=150):
    """Save figure, creating directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scores plots
# ---------------------------------------------------------------------------

def plot_scores(scores_df, y, title="", ax=None, save_path=None):
    """2D scatter of sample scores colored by class.
    
    Parameters
    ----------
    scores_df : DataFrame with comp1, comp2 columns (or first two columns)
    y : array of class labels
    """
    if ax is None:
        fig, ax = _setup_figure()
    else:
        fig = ax.get_figure()
    
    cols = scores_df.columns[:2]
    for cls in CLASS_ORDER:
        mask = np.array(y) == cls
        if mask.any():
            ax.scatter(
                scores_df.loc[mask, cols[0]],
                scores_df.loc[mask, cols[1]],
                c=CLASS_COLORS.get(cls, "gray"),
                label=cls,
                s=80,
                edgecolors="black",
                linewidth=0.5,
                zorder=3,
            )
    
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# VIP / importance bar plots
# ---------------------------------------------------------------------------

def plot_vip(vip_df, top_n=15, title="VIP Scores", ax=None, save_path=None):
    """Horizontal bar plot of VIP scores. Features ≥ 1.0 highlighted."""
    df = vip_df.head(top_n).copy()
    
    if ax is None:
        fig, ax = _setup_figure(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()
    
    colors = ["#1f77b4" if v >= 1.0 else "#aec7e8" for v in df["VIP"]]
    ax.barh(range(len(df)), df["VIP"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="VIP = 1")
    ax.set_xlabel("VIP Score")
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


def plot_importance(importance_df, top_n=15, title="Feature Importance", 
                    value_col="Importance", ax=None, save_path=None):
    """Generic horizontal bar plot for feature importance."""
    df = importance_df.head(top_n).copy()
    
    if ax is None:
        fig, ax = _setup_figure(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()
    
    ax.barh(range(len(df)), df[value_col], color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    ax.set_title(title)
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", ax=None, save_path=None):
    """Heatmap confusion matrix."""
    if labels is None:
        labels = CLASS_ORDER
    
    if ax is None:
        fig, ax = _setup_figure(figsize=(5, 4))
    else:
        fig = ax.get_figure()
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# DIABLO multi-block
# ---------------------------------------------------------------------------

def plot_diablo_scores(diablo_model, y, save_dir=None):
    """Plot scores for each block in a DIABLO model."""
    n_blocks = len(diablo_model.block_names_)
    fig, axes = plt.subplots(1, n_blocks, figsize=(5 * n_blocks, 5))
    if n_blocks == 1:
        axes = [axes]
    
    for ax, name in zip(axes, diablo_model.block_names_):
        scores = diablo_model.block_scores_[name]
        scores_df = pd.DataFrame(scores[:, :2], columns=["comp1", "comp2"])
        plot_scores(scores_df, y, title=name, ax=ax)
    
    fig.suptitle("DIABLO Block Scores", fontsize=14, y=1.02)
    fig.tight_layout()
    
    if save_dir:
        save_fig(fig, Path(save_dir) / "diablo_scores.png")
    return fig


def plot_block_correlations(corr_df, title="Inter-Block Correlations", save_path=None):
    """Heatmap of correlations between block scores."""
    fig, ax = _setup_figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".3f", cmap="RdYlBu_r", vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title(title)
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Consensus features
# ---------------------------------------------------------------------------

def plot_consensus_features(consensus_df, title="Consensus Features", save_path=None):
    """Bar plot of features found important across multiple methods."""
    if consensus_df.empty:
        print("No consensus features to plot.")
        return None, None
    
    fig, ax = _setup_figure(figsize=(10, max(4, len(consensus_df) * 0.4)))
    ax.barh(range(len(consensus_df)), consensus_df["n_methods"],
            color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(consensus_df)))
    ax.set_yticklabels(consensus_df["Feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Methods")
    ax.set_title(title)
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax
