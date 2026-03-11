"""Visualization: shared plotting functions for all methods."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram

try:
    from .utils import _infer_layer
except ImportError:
    from utils import _infer_layer

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


def plot_candidate_drivers(candidate_df, top_n=20,
                           title="Candidate Driver Summary",
                           save_path=None):
    """Compact evidence matrix for top candidate features.

    The figure is designed for communication rather than exploration:
    each row is a feature and columns show cross-method support plus the
    most important WGCNA evidence.
    """
    if candidate_df is None or candidate_df.empty:
        return None, None

    df = candidate_df.copy().head(top_n)
    df["short"] = df["Feature"].apply(_short_feature_name)
    df["layer_display"] = df["layer"].map(LAYER_DISPLAY).fillna(df["layer"])

    layer_colors = {
        "central_carbon": "#1f77b4",
        "amino_acids": "#2ca02c",
        "aromatics": "#ff7f0e",
        "proteomics": "#9467bd",
        "unknown": "#7f7f7f",
    }

    columns = [
        ("Layer", 1.15),
        ("Methods", 1.0),
        ("WGCNA", 0.9),
        ("Hub", 0.7),
        ("Module", 0.9),
        ("Trait r", 1.15),
        ("Hub score", 1.2),
        ("Total", 0.85),
    ]
    x_positions = np.cumsum([0] + [w for _, w in columns[:-1]])
    width_total = sum(w for _, w in columns)

    n = len(df)
    row_h = 0.72
    fig_h = max(5.5, n * 0.42 + 1.8)
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    y_positions = np.arange(n)
    ax.set_xlim(-4.8, width_total + 0.2)
    ax.set_ylim(-1.2, n - 0.4)
    ax.invert_yaxis()

    # Background stripes
    for i, y in enumerate(y_positions):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((-4.8, y - row_h / 2), width_total + 5.0, row_h,
                                       facecolor="#fafafa", edgecolor="none", zorder=0))

    # Headers
    for (label, width), x0 in zip(columns, x_positions):
        ax.text(x0 + width / 2, -0.95, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#333333")

    # Row labels and cells
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        ax.text(-0.15, y, row["short"], ha="right", va="center",
                fontsize=8.5, color="#222222",
                fontweight="bold" if row["integrated_evidence_score"] >= 5 else "normal")

        # Layer color chip + text
        x0, w = x_positions[0], columns[0][1]
        layer_color = layer_colors.get(row["layer"], "#7f7f7f")
        ax.add_patch(plt.Rectangle((x0 + 0.08, y - 0.22), 0.24, 0.44,
                                   facecolor=layer_color, edgecolor="black", linewidth=0.4))
        ax.text(x0 + 0.40, y, row["layer_display"], ha="left", va="center", fontsize=8, color="#333333")

        # Methods count
        x0, w = x_positions[1], columns[1][1]
        ax.text(x0 + w / 2, y, str(int(row["n_methods"])), ha="center", va="center", fontsize=9)

        # WGCNA supported + Hub
        for idx_col, key in enumerate(["wgcna_supported", "wgcna_is_hub"], start=2):
            x0, w = x_positions[idx_col], columns[idx_col][1]
            val = bool(row[key])
            ax.add_patch(plt.Rectangle((x0 + 0.18, y - 0.22), w - 0.36, 0.44,
                                       facecolor="#2166ac" if val else "#f0f0f0",
                                       edgecolor="#c7c7c7", linewidth=0.5))
            ax.text(x0 + w / 2, y, "✓" if val else "—", ha="center", va="center",
                    fontsize=9, color="white" if val else "#999999", fontweight="bold")

        # Module id
        x0, w = x_positions[4], columns[4][1]
        mod = row.get("wgcna_module", np.nan)
        mod_txt = f"M{int(mod)}" if pd.notna(mod) and float(mod) > 0 else "—"
        ax.text(x0 + w / 2, y, mod_txt, ha="center", va="center", fontsize=8.5, color="#333333")

        # Trait correlation
        x0, w = x_positions[5], columns[5][1]
        r = row.get("wgcna_module_trait_correlation", np.nan)
        p = row.get("wgcna_module_trait_p_value", np.nan)
        if pd.notna(r):
            if p < 0.05:
                color = "#b2182b" if r < 0 else "#2166ac"
                weight = "bold"
            else:
                color = "#666666"
                weight = "normal"
            ax.text(x0 + w / 2, y, f"{r:.2f}", ha="center", va="center",
                    fontsize=8.5, color=color, fontweight=weight)
        else:
            ax.text(x0 + w / 2, y, "—", ha="center", va="center", fontsize=8.5, color="#999999")

        # Hub score
        x0, w = x_positions[6], columns[6][1]
        hs = row.get("wgcna_hub_score", np.nan)
        ax.text(x0 + w / 2, y, f"{hs:.2f}" if pd.notna(hs) else "—",
                ha="center", va="center", fontsize=8.5, color="#333333" if pd.notna(hs) else "#999999")

        # Total score
        x0, w = x_positions[7], columns[7][1]
        ax.add_patch(plt.Rectangle((x0 + 0.08, y - 0.22), w - 0.16, 0.44,
                                   facecolor="#e5f5e0", edgecolor="#c7c7c7", linewidth=0.5))
        ax.text(x0 + w / 2, y, str(int(row["integrated_evidence_score"])),
                ha="center", va="center", fontsize=9, fontweight="bold", color="#1b7837")

    # Column separators
    for x0 in list(x_positions) + [width_total]:
        ax.plot([x0, x0], [-0.5, n - 0.5], color="#dddddd", linewidth=0.7, zorder=0)

    # Notes
    note = (
        "Methods = number of supervised methods selecting the feature; "
        "WGCNA = feature belongs to a non-grey module; Hub = top intramodular feature; "
        "Trait r = correlation of the feature's module eigengene with ripening stage."
    )
    ax.text(-4.75, n + 0.1, note, ha="left", va="bottom", fontsize=8, color="#555555")

    ax.set_title(title, fontsize=13, pad=18)
    ax.axis("off")
    fig.tight_layout()

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

MODULE_PALETTE = [
    "#bdbdbd",  # grey / unassigned
    "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
    "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6",
    "#ffff99", "#8dd3c7", "#80b1d3", "#bebada", "#fb8072",
]


def _module_color(module_id):
    """Map integer module ids to display colors."""
    if module_id == 0:
        return MODULE_PALETTE[0]
    return MODULE_PALETTE[(int(module_id) - 1) % (len(MODULE_PALETTE) - 1) + 1]


def plot_scale_free_fit(scale_free_df, selected_power=None, selected_r2=None,
                        target_r2=0.8, threshold_met=None,
                        title="Scale-Free Topology Fit", save_path=None):
    """Plot signed R² and mean connectivity across candidate powers."""
    if scale_free_df is None or len(scale_free_df) == 0:
        return None, None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(scale_free_df["power"], scale_free_df["r_squared"], marker="o", color="#1f77b4")
    axes[0].axhline(target_r2, color="red", linestyle="--", linewidth=1, label=f"Target R² = {target_r2:.1f}")
    if selected_power is not None:
        sel_label = "Selected power"
        if threshold_met is not None:
            sel_label += " (threshold met)" if threshold_met else " (best available)"
        axes[0].axvline(selected_power, color="#2ca02c", linestyle=":", linewidth=1.5, label=sel_label)
        if selected_r2 is not None:
            axes[0].scatter([selected_power], [selected_r2], color="#2ca02c", zorder=5)
    axes[0].set_xlabel("Soft-Threshold Power")
    axes[0].set_ylabel("Signed Scale-Free Fit (R²)")
    axes[0].set_title("Scale-Free Fit")
    axes[0].legend()

    axes[1].plot(scale_free_df["power"], scale_free_df["mean_connectivity"], marker="o", color="#2ca02c")
    if selected_power is not None:
        axes[1].axvline(selected_power, color="#2ca02c", linestyle=":", linewidth=1.5)
    axes[1].set_xlabel("Soft-Threshold Power")
    axes[1].set_ylabel("Mean Connectivity")
    axes[1].set_title("Mean Connectivity")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig, axes


def plot_wgcna_dendrogram(linkage_matrix, module_assignments, title="WGCNA Dendrogram",
                          save_path=None):
    """Plot feature dendrogram with a module-color strip underneath."""
    if linkage_matrix is None or module_assignments is None or module_assignments.empty:
        return None, None

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 0.6], hspace=0.05)
    ax_tree = fig.add_subplot(gs[0, 0])
    ax_colors = fig.add_subplot(gs[1, 0])

    dend = dendrogram(linkage_matrix, no_labels=True, color_threshold=0, above_threshold_color="black", ax=ax_tree)
    ax_tree.set_title(title)
    ax_tree.set_ylabel("TOM Dissimilarity")

    ordered_modules = module_assignments.iloc[dend["leaves"]]["Module"].to_numpy()
    color_row = np.array([[_module_color(m) for m in ordered_modules]], dtype=object)
    ax_colors.imshow([list(range(len(ordered_modules)))], aspect="auto", cmap="gray", alpha=0)
    for i, color in enumerate(color_row[0]):
        ax_colors.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 1, color=color, linewidth=0))
    ax_colors.set_xlim(-0.5, len(ordered_modules) - 0.5)
    ax_colors.set_ylim(-0.5, 0.5)
    ax_colors.set_yticks([0])
    ax_colors.set_yticklabels(["Module"])
    ax_colors.set_xticks([])
    for spine in ax_colors.spines.values():
        spine.set_visible(False)

    if save_path:
        save_fig(fig, save_path)
    return fig, (ax_tree, ax_colors)


def plot_module_sizes(module_assignments, title="WGCNA Module Sizes", save_path=None):
    """Bar plot of detected module sizes including grey/unassigned."""
    if module_assignments is None or module_assignments.empty:
        return None, None

    counts = module_assignments["Module"].value_counts().sort_index().reset_index()
    counts.columns = ["Module", "Count"]
    labels = ["grey" if m == 0 else f"M{int(m)}" for m in counts["Module"]]
    colors = [_module_color(m) for m in counts["Module"]]

    fig, ax = _setup_figure(figsize=(8, 4))
    ax.bar(range(len(counts)), counts["Count"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Number of Features")
    ax.set_title(title)

    if save_path:
        save_fig(fig, save_path)
    return fig, ax

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
