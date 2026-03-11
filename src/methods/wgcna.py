"""
Weighted Gene Co-expression Network Analysis (WGCNA) adapted for metabolomics/proteomics.

Builds a weighted correlation network between features, identifies co-abundance
modules, tests module-trait associations, and identifies hub features.

Reference: Langfelder & Horvath (2008) BMC Bioinformatics.

NOTE: WGCNA was designed for n >= 15-20. With small n, pairwise correlations
are noisy and module detection less stable. Results should be interpreted as
exploratory and cross-referenced with supervised methods.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr


VALID_CORR_METHODS = {"pearson", "spearman"}
VALID_NETWORK_TYPES = {"unsigned", "signed", "signed_hybrid"}


# ---------------------------------------------------------------------------
# Correlation & adjacency
# ---------------------------------------------------------------------------

def compute_correlation_matrix(X, method="pearson"):
    """Compute feature-feature correlation matrix.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    method : 'pearson' or 'spearman'
        Spearman is more robust for small n.
    
    Returns
    -------
    corr : ndarray (n_features, n_features) — correlation matrix
    """
    if method == "spearman":
        corr = np.corrcoef(pd.DataFrame(X).rank().values.T)
    else:
        corr = np.corrcoef(X.T)
    # Fix numerical issues
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1, 1)
    return corr


def _validate_corr_method(method):
    """Validate correlation method name."""
    if method not in VALID_CORR_METHODS:
        raise ValueError(f"Unknown correlation method '{method}'. Expected one of {sorted(VALID_CORR_METHODS)}.")


def _validate_network_type(network_type):
    """Validate WGCNA network type."""
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(
            f"Unknown network_type '{network_type}'. Expected one of {sorted(VALID_NETWORK_TYPES)}."
        )


def adjacency_from_correlation(corr, power=6, network_type="unsigned"):
    """Convert a correlation matrix to a WGCNA adjacency matrix.

    Parameters
    ----------
    corr : ndarray (p, p)
        Feature-feature correlation matrix.
    power : int
        Soft-thresholding power.
    network_type : {'unsigned', 'signed', 'signed_hybrid'}
        WGCNA network type.

    Returns
    -------
    adjacency : ndarray (p, p)
    """
    _validate_network_type(network_type)

    corr = np.asarray(corr, dtype=float)
    corr = np.clip(corr, -1.0, 1.0)

    if network_type == "unsigned":
        adj = np.abs(corr) ** power
    elif network_type == "signed":
        adj = ((1.0 + corr) / 2.0) ** power
    else:  # signed_hybrid
        adj = np.where(corr > 0, corr ** power, 0.0)

    adj = (adj + adj.T) / 2.0
    np.fill_diagonal(adj, 0.0)
    adj = np.clip(adj, 0.0, 1.0)
    return adj


def _compute_module_eigengene(X_mod):
    """Return an oriented first principal component for a module."""
    X_mod = np.asarray(X_mod, dtype=float)
    X_centered = X_mod - X_mod.mean(axis=0)

    if X_centered.ndim != 2 or X_centered.shape[1] < 1:
        raise ValueError("X_mod must be a 2D array with at least one feature.")

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigengene = U[:, 0] * S[0]

    # Orient eigengene consistently with the module's average centered profile.
    avg_profile = X_centered.mean(axis=1)
    if np.std(avg_profile) > 0 and np.std(eigengene) > 0:
        orient_r = np.corrcoef(eigengene, avg_profile)[0, 1]
        if np.isfinite(orient_r) and orient_r < 0:
            eigengene = -eigengene

    return eigengene


def compute_module_eigengenes(X, module_assignments):
    """Compute module eigengenes for all non-grey modules."""
    eigengenes = {}

    for mod in sorted(module_assignments["Module"].unique()):
        if mod == 0:
            continue

        mod_idx = np.where((module_assignments["Module"] == mod).values)[0]
        if len(mod_idx) < 2:
            continue

        try:
            eigengenes[mod] = _compute_module_eigengene(X[:, mod_idx])
        except np.linalg.LinAlgError:
            continue

    return eigengenes


def pick_soft_threshold(X, powers=None, method="pearson", network_type="unsigned",
                        target_r2=0.8):
    """Determine soft-thresholding power for scale-free topology.
    
    Tests multiple powers and returns the one that best fits a scale-free
    topology (R² of log(k) vs log(p(k)) regression).
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    powers : list of int, optional
        Candidate powers to test. Default: [2, 3, ..., 20]
    method : str
        Correlation method.
    target_r2 : float
        Target signed scale-free fit threshold used as a guideline.
    
    Returns
    -------
    dict with 'power', 'r_squared', 'all_results' (DataFrame)
    """
    _validate_corr_method(method)
    _validate_network_type(network_type)

    if powers is None:
        powers = list(range(2, 21))
    
    corr = compute_correlation_matrix(X, method=method)
    results = []
    
    for beta in powers:
        adj = adjacency_from_correlation(corr, power=beta, network_type=network_type)
        
        # Connectivity (degree) of each node
        k = adj.sum(axis=0)
        
        # Scale-free fit: regress log(p(k)) ~ log(k)
        # Bin the connectivity distribution
        k_pos = k[k > 0]
        if len(k_pos) < 5:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        
        # Histogram-based approach
        n_bins = max(5, min(20, len(k_pos) // 3))
        hist, bin_edges = np.histogram(k_pos, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove empty bins
        mask = hist > 0
        if mask.sum() < 3:
            results.append({"power": beta, "r_squared": 0.0, "mean_connectivity": k.mean()})
            continue
        
        log_k = np.log10(bin_centers[mask])
        log_pk = np.log10(hist[mask].astype(float))
        
        # Linear regression
        slope, intercept = np.polyfit(log_k, log_pk, 1)
        predicted = slope * log_k + intercept
        ss_res = np.sum((log_pk - predicted) ** 2)
        ss_tot = np.sum((log_pk - log_pk.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # For scale-free, we want negative slope and high R²
        r2_signed = -np.sign(slope) * r2
        
        results.append({
            "power": beta, 
            "r_squared": r2_signed, 
            "mean_connectivity": k.mean(),
        })
    
    df = pd.DataFrame(results)
    
    # Pick first power where R² exceeds the guideline, or else the best available.
    good = df[df["r_squared"] > target_r2]
    if len(good) > 0:
        best = good.iloc[0]
        selection_rule = "first_above_target"
    else:
        best = df.loc[df["r_squared"].idxmax()]
        selection_rule = "best_available"
    
    return {
        "power": int(best["power"]),
        "r_squared": best["r_squared"],
        "target_r2": float(target_r2),
        "selection_rule": selection_rule,
        "threshold_met": bool(len(good) > 0),
        "all_results": df,
    }


def compute_adjacency(X, power=6, method="pearson", network_type="unsigned"):
    """Compute a WGCNA adjacency matrix from pairwise correlations.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    power : int — soft-thresholding power
    method : str — correlation method
    network_type : {'unsigned', 'signed', 'signed_hybrid'}
        WGCNA network type.
    
    Returns
    -------
    adjacency : ndarray (n_features, n_features)
    """
    _validate_corr_method(method)
    _validate_network_type(network_type)

    corr = compute_correlation_matrix(X, method=method)
    return adjacency_from_correlation(corr, power=power, network_type=network_type)


def compute_tom(adjacency):
    """Topological Overlap Matrix (TOM).
    
    TOM measures interconnectedness: features sharing many neighbors 
    have high TOM even if their direct correlation is moderate.
    
    TOM_ij = (l_ij + a_ij) / (min(k_i, k_j) + 1 - a_ij)
    where l_ij = sum_u(a_iu * a_uj), k_i = sum_u(a_iu)
    
    Parameters
    ----------
    adjacency : ndarray (p, p) — adjacency matrix
    
    Returns
    -------
    tom : ndarray (p, p) — TOM similarity matrix
    """
    adjacency = np.asarray(adjacency, dtype=float)
    adjacency = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)
    adjacency = np.clip(adjacency, 0.0, 1.0)

    p = adjacency.shape[0]
    k = adjacency.sum(axis=0)  # connectivity
    
    # l_ij = sum_u(a_iu * a_uj) = (A @ A)_ij
    L = adjacency @ adjacency
    
    # Vectorized TOM computation
    # min(k_i, k_j) for all pairs
    k_row = k[np.newaxis, :]  # (1, p)
    k_col = k[:, np.newaxis]  # (p, 1)
    k_min = np.minimum(k_row, k_col)
    
    numerator = L + adjacency
    denominator = k_min + 1 - adjacency
    
    # Avoid division by zero
    tom = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    tom = (tom + tom.T) / 2.0
    tom = np.clip(tom, 0.0, 1.0)
    np.fill_diagonal(tom, 1.0)
    return tom


# ---------------------------------------------------------------------------
# Module detection
# ---------------------------------------------------------------------------

def detect_modules(tom, feature_names=None, min_module_size=3, cut_height=None):
    """Identify feature modules via hierarchical clustering on 1 - TOM.
    
    Parameters
    ----------
    tom : ndarray (p, p) — TOM similarity matrix
    feature_names : list of str
    min_module_size : int — minimum features per module (smaller assigned to grey)
    cut_height : float, optional — tree cut threshold on TOM dissimilarity
    
    Returns
    -------
    module_assignments : DataFrame with Feature, Module columns
    linkage_matrix : scipy linkage matrix (for dendrogram)
    """
    p = tom.shape[0]
    names = feature_names or [f"f{i}" for i in range(p)]
    
    # Distance = 1 - TOM
    dist = 1 - tom
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)
    
    # Convert to condensed form for scipy
    dist_condensed = squareform(dist, checks=False)
    
    # Hierarchical clustering (average linkage, as in WGCNA)
    Z = linkage(dist_condensed, method="average")
    
    # Default cut height for TOM dissimilarity.
    # WGCNA commonly uses a dynamic tree cut; here we use a conservative,
    # TOM-based cut followed by eigengene merging to approximate that flow
    # without an additional dependency.
    if cut_height is None:
        if p <= 25:
            cut_height = 0.40
        elif p <= 100:
            cut_height = 0.35
        else:
            cut_height = 0.25
    
    labels = fcluster(Z, t=cut_height, criterion="distance")
    
    # Merge small modules into Module 0 ("grey" / unassigned)
    module_counts = pd.Series(labels).value_counts()
    small_modules = module_counts[module_counts < min_module_size].index
    labels = np.array([0 if lab in small_modules else lab for lab in labels])
    
    # Renumber modules sequentially (1, 2, 3, ...) with 0 = unassigned
    unique_mods = sorted(set(labels) - {0})
    remap = {0: 0}
    for i, m in enumerate(unique_mods, 1):
        remap[m] = i
    labels = np.array([remap[lab] for lab in labels])
    
    df = pd.DataFrame({"Feature": names, "Module": labels})
    return df, Z


def merge_modules_by_eigengene(X, module_assignments, threshold=0.25):
    """Merge modules whose eigengenes are highly correlated.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    module_assignments : DataFrame with Feature, Module columns
    threshold : float
        Merge modules when eigengene dissimilarity <= threshold.
        A common WGCNA choice is 0.25, corresponding to correlation >= 0.75.

    Returns
    -------
    DataFrame with updated module labels.
    """
    merged = module_assignments.copy()
    eigengenes = compute_module_eigengenes(X, merged)
    if len(eigengenes) < 2:
        return merged

    modules = sorted(eigengenes)
    me_matrix = np.column_stack([eigengenes[m] for m in modules])
    me_corr = np.corrcoef(me_matrix.T)
    me_corr = np.nan_to_num(me_corr, nan=0.0)
    np.fill_diagonal(me_corr, 1.0)

    me_diss = 1.0 - np.abs(me_corr)
    me_diss = np.clip((me_diss + me_diss.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(me_diss, 0.0)

    Z_me = linkage(squareform(me_diss, checks=False), method="average")
    me_labels = fcluster(Z_me, t=threshold, criterion="distance")

    module_to_cluster = {mod: lab for mod, lab in zip(modules, me_labels)}
    unique_clusters = sorted(set(me_labels))
    cluster_to_new = {cluster: i + 1 for i, cluster in enumerate(unique_clusters)}

    merged["Module"] = merged["Module"].map(
        lambda mod: 0 if mod == 0 else cluster_to_new[module_to_cluster[mod]]
    )
    return merged


# ---------------------------------------------------------------------------
# Module-trait associations
# ---------------------------------------------------------------------------

def module_trait_correlation(X, y_encoded, module_assignments, method="pearson"):
    """Test association between module eigengenes and trait (ripening stage).
    
    Module eigengene = first PC of features in that module.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y_encoded : ndarray of ordinal integers (e.g. 0, 1, 2)
    module_assignments : DataFrame with Feature, Module columns
    method : 'pearson' or 'spearman'
    
    Returns
    -------
    DataFrame with Module, n_features, correlation, p_value
    """
    _validate_corr_method(method)

    modules = sorted(module_assignments["Module"].unique())
    results = []
    eigengenes = compute_module_eigengenes(X, module_assignments)
    
    for mod in modules:
        if mod == 0:
            continue  # skip unassigned

        mod_idx = np.where((module_assignments["Module"] == mod).values)[0]
        if mod not in eigengenes or len(mod_idx) < 2:
            continue

        eigengene = eigengenes[mod]
        
        # Correlate with trait
        if method == "spearman":
            r, p = spearmanr(eigengene, y_encoded)
        else:
            r, p = pearsonr(eigengene, y_encoded)
        
        results.append({
            "Module": mod,
            "N_Features": len(mod_idx),
            "Correlation": r,
            "P_Value": p,
            "Abs_Correlation": abs(r),
        })
    
    if not results:
        return pd.DataFrame(columns=["Module", "N_Features", "Correlation", "P_Value", "Abs_Correlation"]), eigengenes

    df = pd.DataFrame(results).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)
    return df, eigengenes


# ---------------------------------------------------------------------------
# Hub features
# ---------------------------------------------------------------------------

def identify_hub_features(X, module_assignments, adjacency, y_encoded=None, top_n=5,
                          method="pearson"):
    """Identify hub features: high intramodular connectivity + trait correlation.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    module_assignments : DataFrame with Feature, Module
    adjacency : ndarray (p, p)
    y_encoded : ndarray, optional — if provided, compute feature-trait correlation
    top_n : int — hub features per module
    
    Returns
    -------
    DataFrame with Feature, Module, Intramodular_Connectivity, Trait_Correlation, Is_Hub
    """
    _validate_corr_method(method)

    all_features = module_assignments["Feature"].tolist()
    records = []
    
    modules = sorted(module_assignments["Module"].unique())
    
    for mod in modules:
        if mod == 0:
            continue
        
        mod_mask = module_assignments["Module"] == mod
        mod_indices = np.where(mod_mask.values)[0]
        mod_features = module_assignments.loc[mod_mask, "Feature"].tolist()
        
        if len(mod_indices) < 2:
            continue
        
        # Intramodular connectivity: sum of adjacency within module
        adj_sub = adjacency[np.ix_(mod_indices, mod_indices)]
        k_in = adj_sub.sum(axis=0)  # connectivity within module
        k_in_norm = k_in / k_in.max() if k_in.max() > 0 else k_in
        
        for i, feat_idx in enumerate(mod_indices):
            record = {
                "Feature": all_features[feat_idx],
                "Module": mod,
                "Intramodular_Connectivity": k_in[i],
                "Normalized_Connectivity": k_in_norm[i],
            }
            
            if y_encoded is not None:
                if method == "spearman":
                    r, p = spearmanr(X[:, feat_idx], y_encoded)
                else:
                    r, p = pearsonr(X[:, feat_idx], y_encoded)
                record["Trait_Correlation"] = r
                record["Trait_P_Value"] = p
                # Gene significance (GS) in WGCNA parlance
                record["Gene_Significance"] = abs(r)
            
            records.append(record)
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Hub score: product of normalized connectivity and gene significance
    if y_encoded is not None and "Gene_Significance" in df.columns:
        df["Hub_Score"] = df["Normalized_Connectivity"] * df["Gene_Significance"]
    else:
        df["Hub_Score"] = df["Normalized_Connectivity"]
    
    # Mark top hubs per module
    df["Is_Hub"] = False
    for mod in df["Module"].unique():
        mod_mask = df["Module"] == mod
        mod_df = df.loc[mod_mask].nlargest(top_n, "Hub_Score")
        df.loc[mod_df.index, "Is_Hub"] = True
    
    return df.sort_values(["Module", "Hub_Score"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full WGCNA pipeline
# ---------------------------------------------------------------------------

def run_wgcna(X, y_encoded, feature_names=None, power=None, corr_method="spearman",
              network_type="unsigned", min_module_size=None,
              module_cut_height=None, merge_cut_height=0.25, top_n_hubs=5):
    """Run full WGCNA pipeline on a single omics block.
    
    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y_encoded : ndarray of ordinal integers (0, 1, 2)
    feature_names : list of str
    power : int, optional — soft-thresholding power (auto-detected if None)
    corr_method : 'pearson' or 'spearman' (spearman recommended for small n)
    network_type : {'unsigned', 'signed', 'signed_hybrid'}
        WGCNA network type. Default is unsigned to preserve prior behavior.
    min_module_size : int, optional — auto-scaled if None using a capped WGCNA-style rule
    module_cut_height : float, optional — tree cut threshold on TOM dissimilarity
    merge_cut_height : float — module eigengene dissimilarity threshold for merging
    top_n_hubs : int — hub features to flag per module
    
    Returns
    -------
    dict with keys:
        'power' : int — selected soft threshold power
        'modules' : DataFrame — feature to module assignments
        'module_trait' : DataFrame — module-trait correlations
        'hubs' : DataFrame — hub features with connectivity & trait correlation
        'adjacency' : ndarray — adjacency matrix
        'tom' : ndarray — TOM matrix
        'linkage' : ndarray — hierarchical clustering linkage
        'scale_free_fit' : DataFrame — power scan summary
        'scale_free_selection' : dict — selected power metadata
        'eigengenes' : dict — {module: eigengene array}
    """
    _validate_corr_method(corr_method)
    _validate_network_type(network_type)

    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    p = X.shape[1]
    
    # WGCNA commonly uses minModuleSize around 20-30 for large datasets.
    # For very small example blocks we allow smaller modules, but cap the
    # default at 20 so large omics layers are not over-pruned.
    if min_module_size is None:
        min_module_size = min(20, max(3, p // 10))
    
    # 1. Pick soft threshold
    threshold_result = None
    if power is None:
        threshold_result = pick_soft_threshold(X, method=corr_method, network_type=network_type)
        power = threshold_result["power"]
        reason = "threshold met" if threshold_result["threshold_met"] else "best available"
        print(
            f"    Selected soft-threshold power: {power} "
            f"(R² = {threshold_result['r_squared']:.3f}; {reason})"
        )
    
    # 2. Compute adjacency and TOM
    adj = compute_adjacency(X, power=power, method=corr_method, network_type=network_type)
    tom = compute_tom(adj)
    
    # 3. Detect modules
    effective_module_cut_height = module_cut_height
    if effective_module_cut_height is None:
        if p <= 25:
            effective_module_cut_height = 0.40
        elif p <= 100:
            effective_module_cut_height = 0.35
        else:
            effective_module_cut_height = 0.25

    modules_df, Z = detect_modules(
        tom,
        feature_names=names,
        min_module_size=min_module_size,
        cut_height=effective_module_cut_height,
    )

    if module_cut_height is None and len(set(modules_df["Module"]) - {0}) == 0:
        for fallback_cut in (0.40, 0.50, 0.60, 0.70):
            modules_df, Z = detect_modules(
                tom,
                feature_names=names,
                min_module_size=min_module_size,
                cut_height=fallback_cut,
            )
            if len(set(modules_df["Module"]) - {0}) > 0:
                effective_module_cut_height = fallback_cut
                break

    modules_df = merge_modules_by_eigengene(X, modules_df, threshold=merge_cut_height)
    n_modules = len(set(modules_df["Module"]) - {0})
    n_unassigned = (modules_df["Module"] == 0).sum()
    print(f"    Detected {n_modules} modules ({n_unassigned} unassigned features)")
    
    # 4. Module-trait correlations
    mod_trait_df, eigengenes = module_trait_correlation(X, y_encoded, modules_df, method=corr_method)
    
    # 5. Hub features
    hubs_df = identify_hub_features(
        X, modules_df, adj, y_encoded=y_encoded, top_n=top_n_hubs, method=corr_method
    )
    
    return {
        "power": power,
        "network_type": network_type,
        "module_cut_height": effective_module_cut_height,
        "merge_cut_height": merge_cut_height,
        "modules": modules_df,
        "module_trait": mod_trait_df,
        "hubs": hubs_df,
        "adjacency": adj,
        "tom": tom,
        "linkage": Z,
        "scale_free_fit": None if threshold_result is None else threshold_result["all_results"],
        "scale_free_selection": None if threshold_result is None else {
            "power": threshold_result["power"],
            "r_squared": threshold_result["r_squared"],
            "target_r2": threshold_result["target_r2"],
            "selection_rule": threshold_result["selection_rule"],
            "threshold_met": threshold_result["threshold_met"],
        },
        "eigengenes": eigengenes,
    }


# ---------------------------------------------------------------------------
# WGCNA-based dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_by_wgcna(X, wgcna_result, strategy="eigengenes_and_hubs"):
    """Create a reduced feature matrix using WGCNA module structure.

    Uses WGCNA as an *unsupervised* dimensionality-reduction step:
    correlated features are collapsed into module eigengenes, and optionally
    supplemented by hub features that retain individual identity.

    Because WGCNA never sees the class labels during module construction,
    this introduces **no data leakage** and can safely precede supervised
    methods such as sPLS-DA, DIABLO, RF, or ordinal regression.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        The same matrix that was passed to ``run_wgcna``.
    wgcna_result : dict
        Return value of ``run_wgcna``.
    strategy : {'eigengenes_only', 'hubs_only', 'eigengenes_and_hubs'}
        - ``eigengenes_only``: one column per non-grey module (1st PC).
        - ``hubs_only``: columns for hub features only.
        - ``eigengenes_and_hubs``: both (default).

    Returns
    -------
    X_reduced : ndarray (n_samples, p_reduced)
    feature_names : list of str
        Human-readable names; eigengene columns are prefixed ``ME_``.
    meta : dict
        ``n_original``, ``n_reduced``, ``strategy``, ``module_counts``.
    """
    valid = {"eigengenes_only", "hubs_only", "eigengenes_and_hubs"}
    if strategy not in valid:
        raise ValueError(f"strategy must be one of {sorted(valid)}, got '{strategy}'")

    modules_df = wgcna_result["modules"]
    eigengenes = wgcna_result["eigengenes"]
    hubs_df = wgcna_result["hubs"]
    all_features = modules_df["Feature"].tolist()

    cols = []
    names = []

    # --- Eigengene columns ---
    if strategy in ("eigengenes_only", "eigengenes_and_hubs"):
        for mod_id in sorted(eigengenes):
            cols.append(eigengenes[mod_id].reshape(-1, 1))
            names.append(f"ME_{mod_id}")

    # --- Hub feature columns ---
    if strategy in ("hubs_only", "eigengenes_and_hubs"):
        hub_names = set()
        if hubs_df is not None and not hubs_df.empty:
            hub_rows = hubs_df[hubs_df["Is_Hub"]]
            hub_names = set(hub_rows["Feature"].tolist())

        for feat in hub_names:
            if feat in all_features:
                idx = all_features.index(feat)
                cols.append(X[:, idx].reshape(-1, 1))
                names.append(feat)

    if not cols:
        # Fallback: no modules detected — return empty reduction
        return np.empty((X.shape[0], 0)), [], {
            "n_original": X.shape[1],
            "n_reduced": 0,
            "strategy": strategy,
            "module_counts": {},
        }

    X_reduced = np.hstack(cols)

    module_counts = (
        modules_df[modules_df["Module"] != 0]["Module"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return X_reduced, names, {
        "n_original": X.shape[1],
        "n_reduced": X_reduced.shape[1],
        "strategy": strategy,
        "module_counts": module_counts,
    }
