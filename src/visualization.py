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


# ---------------------------------------------------------------------------
# Stability selection
# ---------------------------------------------------------------------------

def plot_stability(stability_df, top_n=20, title="Stability Selection", ax=None, save_path=None):
    """Bar plot of feature selection frequency from bootstrap stability."""
    df = stability_df.head(top_n).copy()
    
    if ax is None:
        fig, ax = _setup_figure(figsize=(8, max(4, top_n * 0.35)))
    else:
        fig = ax.get_figure()
    
    colors = ["#2ca02c" if s else "#aec7e8" for s in df["Stable"]]
    ax.barh(range(len(df)), df["Selection_Frequency"], color=colors,
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["Feature"], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.8, color="red", linestyle="--", linewidth=1, label="Stable (0.8)")
    ax.set_xlabel("Bootstrap Selection Frequency")
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def plot_permutation_null(perm_result, title="Permutation Test", save_path=None):
    """Histogram of null distribution with true accuracy marked."""
    fig, ax = _setup_figure()
    
    ax.hist(perm_result["null_distribution"], bins=30, color="#aec7e8",
            edgecolor="black", linewidth=0.5, density=True, label="Null distribution")
    ax.axvline(perm_result["true_accuracy"], color="red", linewidth=2,
               linestyle="--", label=f"True accuracy ({perm_result['true_accuracy']:.3f})")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\np = {perm_result['p_value']:.4f}")
    ax.legend()
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# WGCNA
# ---------------------------------------------------------------------------

def plot_module_trait(module_trait_df, title="Module-Trait Correlations", save_path=None):
    """Heatmap-style bar plot of module-trait correlations."""
    if module_trait_df.empty:
        return None, None
    
    fig, ax = _setup_figure(figsize=(8, max(3, len(module_trait_df) * 0.5)))
    
    colors = ["#d62728" if p < 0.05 else "#aec7e8" for p in module_trait_df["P_Value"]]
    bars = ax.barh(range(len(module_trait_df)), module_trait_df["Correlation"],
                   color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(module_trait_df)))
    labels = [f"Module {int(m)} ({n} feat)" 
              for m, n in zip(module_trait_df["Module"], module_trait_df["N_Features"])]
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Correlation with Ripening Stage")
    ax.set_title(title)
    
    # Add p-value annotations
    for i, (_, row) in enumerate(module_trait_df.iterrows()):
        ax.text(row["Correlation"] + 0.02 * np.sign(row["Correlation"]),
                i, f"p={row['P_Value']:.3f}", va="center", fontsize=8)
    
    if save_path:
        save_fig(fig, save_path)
    return fig, ax


# ---------------------------------------------------------------------------
# Method convergence grid
# ---------------------------------------------------------------------------

LAYER_ORDER = ["central_carbon", "amino_acids", "aromatics", "proteomics"]
LAYER_DISPLAY = {
    "central_carbon": "Central Carbon",
    "amino_acids": "Amino Acids",
    "aromatics": "Aromatics",
    "proteomics": "Proteomics",
}
METHOD_ORDER = ["sPLS-DA", "RF", "Ordinal", "DIABLO"]
METHOD_SUFFIX_MAP = {"splsda": "sPLS-DA", "rf": "RF", "ordinal": "Ordinal"}


def _short_feature_name(feature):
    """Shorten long feature names for display."""
    s = str(feature)
    if " : Conc. (nM)" in s:
        s = s.replace(" : Conc. (nM)", "")
        parts = s.split("_")
        if len(parts) >= 2:
            core = "_".join(parts[1:])
            return core.replace("_0.0", "").replace(" 0.0", "").strip()
    if s.endswith(" Area"):
        return s[:-5]
    if ";" in s and len(s) > 20:
        return s.split(";")[0]
    return s


def _infer_layer(methods_str):
    """Return the omics layer a feature belongs to from its methods string."""
    for layer in LAYER_ORDER:
        if layer in methods_str:
            return layer
    return "unknown"


def _fmt_stab(val):
    """Format a stability value for display, treating NaN and 0 as not applicable."""
    if np.isnan(val) or val == 0.0:
        return "\u2014", "#bbbbbb", "normal"
    txt = f"{val:.2f}"
    if val >= 0.8:
        return txt, "#2166ac", "bold"
    return txt, "#666666", "normal"


def plot_convergence_grid(consensus_df, splsda_stability_map=None,
                          diablo_stability_map=None, save_path=None, dpi=300):
    """Plot method convergence grid for consensus features.

    Parameters
    ----------
    consensus_df : DataFrame
        Must have columns: Feature, n_methods, methods.
    splsda_stability_map : dict, optional
        {feature_name: selection_frequency} from sPLS-DA stability selection.
    diablo_stability_map : dict, optional
        {feature_name: selection_frequency} from DIABLO stability selection.
    save_path : str or Path, optional
        If given, saves PNG (at *dpi*) and SVG.
    """
    from matplotlib.patches import Patch

    splsda_stability_map = splsda_stability_map or {}
    diablo_stability_map = diablo_stability_map or {}

    df = consensus_df.copy()
    df["layer"] = df["methods"].apply(_infer_layer)
    df["short"] = df["Feature"].apply(_short_feature_name)

    # Show 4/4 and 3/4 features
    show = df[df["n_methods"] >= 3].sort_values(
        ["layer", "n_methods"], ascending=[True, False]
    ).reset_index(drop=True)

    # Parse method membership
    rows = []
    for _, r in show.iterrows():
        present = {m: False for m in METHOD_ORDER}
        for token in r["methods"].split(", "):
            token = token.strip()
            if token.startswith("diablo_"):
                present["DIABLO"] = True
            else:
                for suffix, mname in METHOD_SUFFIX_MAP.items():
                    if token.endswith(f"_{suffix}"):
                        present[mname] = True
                        break
        rows.append({
            "Feature": r["short"],
            "Layer": LAYER_DISPLAY.get(r["layer"], r["layer"]),
            "layer_key": r["layer"],
            "n_methods": r["n_methods"],
            **present,
        })
    grid = pd.DataFrame(rows)
    grid["Stab_sPLSDA"] = [splsda_stability_map.get(f, np.nan) for f in show["Feature"]]
    grid["Stab_DIABLO"] = [diablo_stability_map.get(f, np.nan) for f in show["Feature"]]

    n_feat = len(grid)
    fig_h = max(7.5, 0.38 * n_feat + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    layers = grid["Layer"].values
    features = grid["Feature"].values
    n_arr = grid["n_methods"].values
    stab_s = grid["Stab_sPLSDA"].values
    stab_d = grid["Stab_DIABLO"].values
    lkeys = grid["layer_key"].values

    # Professional monochrome palette
    fill_color = "#2166ac"
    empty_color = "#f0f0f0"
    text_dark = "#333333"
    text_mid = "#666666"
    bracket_color = "#999999"

    # X layout — generous spacing to avoid overlaps
    n_methods_count = len(METHOD_ORDER)
    col_stab_s = n_methods_count + 0.6          # sPLS-DA stab
    col_stab_d = col_stab_s + 0.9               # DIABLO stab
    col_n = col_stab_d + 0.7                    # n column
    feat_x = -0.2                               # feature label right edge
    bracket_x = feat_x - 2.8                    # bracket well left of longest names
    layer_x = bracket_x - 0.3                   # layer label right edge

    cw, ch = 0.85, 0.75

    # Y positions (top-down, gaps between layers)
    gap = 0.45
    y_pos = []
    prev = None
    y = 0
    for i in range(n_feat):
        if prev is not None and layers[i] != prev:
            y += gap
        y_pos.append(y)
        y += 1.0
        prev = layers[i]
    y_pos = np.array(y_pos)
    y_top, y_bot = -1.2, y_pos[-1] + 1.0

    # --- Grid cells ---
    for i in range(n_feat):
        for j, m in enumerate(METHOD_ORDER):
            sel = grid.iloc[i][m]
            rect = plt.Rectangle(
                (j + (1 - cw) / 2, y_pos[i] - ch / 2), cw, ch,
                facecolor=fill_color if sel else empty_color,
                edgecolor="white" if sel else "#d0d0d0",
                linewidth=0.6, alpha=0.9 if sel else 0.35,
            )
            ax.add_patch(rect)

    # --- Feature labels (monochrome) ---
    for i in range(n_feat):
        ax.text(feat_x, y_pos[i], features[i], ha="right", va="center",
                fontsize=8.5, color=text_dark,
                fontweight="bold" if n_arr[i] == 4 else "normal")

    # --- Stability columns ---
    # Use combined single-line headers to avoid overlap
    ax.text(col_stab_s, y_top, "sPLS-DA\nStab.", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=text_dark, linespacing=1.3)
    for i in range(n_feat):
        txt, col, fw = _fmt_stab(stab_s[i])
        ax.text(col_stab_s, y_pos[i], txt, ha="center", va="center",
                fontsize=8, color=col, fontweight=fw)

    ax.text(col_stab_d, y_top, "DIABLO\nStab.", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=text_dark, linespacing=1.3)
    for i in range(n_feat):
        txt, col, fw = _fmt_stab(stab_d[i])
        ax.text(col_stab_d, y_pos[i], txt, ha="center", va="center",
                fontsize=8, color=col, fontweight=fw)

    # --- n column ---
    ax.text(col_n, y_top, "n", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color=text_dark)
    for i in range(n_feat):
        ax.text(col_n, y_pos[i], str(n_arr[i]), ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=fill_color if n_arr[i] == 4 else text_mid)

    # --- Layer brackets + labels (shifted left, no overlap) ---
    for ld in dict.fromkeys(layers):
        idxs = [i for i in range(n_feat) if layers[i] == ld]
        y_min_g = y_pos[idxs[0]] - 0.45
        y_max_g = y_pos[idxs[-1]] + 0.45
        # Bracket
        ax.plot([bracket_x, bracket_x], [y_min_g, y_max_g],
                color=bracket_color, lw=1.2, clip_on=False)
        ax.plot([bracket_x, bracket_x + 0.12], [y_min_g, y_min_g],
                color=bracket_color, lw=1.2, clip_on=False)
        ax.plot([bracket_x, bracket_x + 0.12], [y_max_g, y_max_g],
                color=bracket_color, lw=1.2, clip_on=False)
        # Label to the left of the bracket
        ax.text(layer_x, (y_min_g + y_max_g) / 2, ld, ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=text_dark)

    # --- Method column headers (monochrome) ---
    for j, m in enumerate(METHOD_ORDER):
        ax.text(j + 0.5, y_top, m, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=text_dark)

    # --- Legend ---
    ax.legend(
        handles=[
            Patch(facecolor=fill_color, edgecolor="white", label="Selected (top 15 per layer)"),
            Patch(facecolor=empty_color, edgecolor="#d0d0d0", label="Not selected"),
        ],
        loc="lower right", fontsize=8.5, framealpha=0.9, edgecolor="#cccccc",
    )

    # --- Footnote ---
    footnote = (
        "Each cell indicates whether the feature ranked in a method\u2019s "
        "top 15 within its own omics layer. Stab. columns show bootstrap "
        "stability selection frequency (100 resamples); bold \u2265 0.80. "
        "\u2014 = not selected (VIP < 1) in any bootstrap."
    )
    ax.text(0.5, -0.02, footnote, transform=ax.transAxes, ha="center", va="top",
            fontsize=7.5, color=text_mid, style="italic",
            wrap=True)

    ax.set_xlim(layer_x - 1.5, col_n + 0.6)
    ax.set_ylim(y_bot + 0.5, y_top - 0.8)
    ax.axis("off")
    ax.set_title("Method Convergence: Features Selected Across Independent Methods",
                 fontsize=13, fontweight="bold", pad=18, color=text_dark)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        svg_path = p.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
    return fig, ax
