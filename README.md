# Multi-Omics Integration

Python package for multi-omics classification and feature selection, combining supervised single-layer analyses with joint integration and unsupervised network methods. Designed for small-sample multi-omics studies where statistical rigour requires complementary approaches.

Implements five complementary methods — **sPLS-DA / DIABLO**, **Random Forest**, **Ordinal Regression**, **Bootstrap Stability Selection**, and **WGCNA** — with **permutation testing** for significance assessment.

## Project Structure

```
multiomics_integration/
├── data/                           # Banana ripeness dataset (4 omics layers, n=9)
│   ├── badata-amino-acids.csv
│   ├── badata-aromatics.csv
│   ├── badata-metabolomics.csv     # Central carbon metabolism
│   └── badata-proteomics-imputed.csv
├── src/
│   ├── ingestion.py                # Load, impute, transform, scale, align
│   ├── visualization.py            # All plotting (scores, VIP, stability, permutation, WGCNA)
│   ├── utils.py                    # I/O helpers, consensus feature identification
│   └── methods/
│       ├── plsda.py                # sPLS-DA, DIABLO, permutation tests, stability selection
│       ├── random_forest.py        # RF with SHAP, permutation importance
│       ├── ordinal.py              # mord LogisticAT/IT/SE
│       └── wgcna.py                # Weighted correlation network analysis
├── examples/
│   └── banana_workflow.py          # End-to-end demo: single → multi → consensus
├── tests/
│   ├── validate_plsda.py           # Numerical validation vs mixOmics R reference
│   └── reference_data/             # R reference outputs
├── results/                        # Generated outputs (gitignored)
├── setup.py
└── requirements.txt
```

## Methods

### Supervised Methods

| Method | Module | Type | Purpose |
|--------|--------|------|---------|
| sPLS-DA | `plsda.SPLSDA` | Single-block | Sparse PLS-DA via NIPALS with L1 soft-thresholding — identifies discriminant features |
| DIABLO | `plsda.DIABLO` | Multi-block | Multi-block sPLS-DA with design matrix — integrates across omics layers |
| Random Forest | `random_forest.train_rf` | Single / concat | Balanced RF with Gini importance, SHAP values, and permutation importance |
| Ordinal Regression | `ordinal.train_ordinal` | Single / concat | mord LogisticAT/IT/SE — respects class ordering (Green < Ripe < Overripe) |

### Statistical Validation

| Method | Module | Purpose |
|--------|--------|---------|
| Permutation Testing | `plsda.permutation_test_splsda`, `permutation_test_diablo` | Label-shuffling null distribution with Clopper-Pearson early stopping |
| sPLS-DA Stability | `plsda.stability_selection_splsda` | 100 stratified bootstrap resamples of single-block sPLS-DA; features with VIP ≥ 1 in ≥ 80% are "Stable" |
| DIABLO Stability | `plsda.stability_selection_diablo` | 100 stratified bootstrap resamples of the joint multi-block model; per-block stability frequencies |

### Unsupervised Network Analysis

| Method | Module | Purpose |
|--------|--------|---------|
| WGCNA | `wgcna.run_wgcna` | Pairwise correlation → soft thresholding → TOM → hierarchical clustering → module eigengenes → module-trait correlation → hub identification |

Notes:
- In the provided workflow, WGCNA is run **independently for each single-omics layer**; it is **not** used as a joint multi-omics network model.
- `run_wgcna()` supports `network_type="unsigned"`, `"signed"`, or `"signed_hybrid"`; the banana example uses **unsigned**.
- Module detection uses hierarchical clustering on `1 - TOM`, followed by **eigengene-based module merging** (`merge_cut_height=0.25` by default).
- The example workflow generates biology-facing WGCNA visuals, including a **scale-free fit diagnostic**, **feature dendrogram with module colors**, **module-size summary**, and **module-trait association plot**.

## Preprocessing Pipeline

1. **Drop sparse features** — remove features with > 50% missing values
2. **Half-min imputation** — MNAR-appropriate (assumes missingness below detection limit)
3. **Log transform** — per-column offset for negatives/zeros, then natural log
4. **Pareto scaling** — divide by √std (balances high/low-abundance features without compressing variance as much as z-scoring)

## Installation

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -e .
```

Or with pinned versions:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
python examples/banana_workflow.py
```

This runs the full analysis pipeline:
1. **Single-omics** (per layer): sPLS-DA, RF + SHAP, ordinal regression, stability selection, permutation testing, WGCNA
2. **Multi-omics**: DIABLO integration, concatenated RF/ordinal baselines, DIABLO permutation testing, DIABLO stability selection
3. **Consensus**: features appearing across multiple methods' top-15 lists, method convergence grid visualisation

For WGCNA, the example currently uses:
- `corr_method="spearman"`
- `network_type="unsigned"`
- automatic soft-threshold power selection
- per-layer analysis only (no joint multi-omics WGCNA)

## Outputs

Results are saved to `results/` with this structure:

```
results/
├── method_comparison.csv           # All methods' LOO accuracies
├── consensus_features.csv          # Features selected by ≥ 2 methods, enriched with WGCNA support columns
├── candidate_driver_features.csv   # Ranked shortlist combining cross-method consensus + WGCNA module/hub evidence
├── candidate_driver_features.png   # Visual summary of integrated candidate-driver evidence
├── consensus_features.png
├── single_omics/<layer>/
│   ├── splsda_scores.png           # sPLS-DA sample scores (2D)
│   ├── splsda_vip.png              # VIP bar plot
│   ├── splsda_vip_scores.csv
│   ├── splsda_stability.csv        # Bootstrap selection frequencies
│   ├── splsda_stability.png        # Stability bar plot
│   ├── splsda_permutation_test.json
│   ├── splsda_permutation_null.png # Null distribution histogram
│   ├── rf_importance.png           # Gini importance bar plot
│   ├── rf_feature_importance.csv
│   ├── rf_shap_importance.csv      # Mean |SHAP| (≤ 1000 features)
│   ├── rf_permutation_importance.csv
│   ├── rf_confusion_matrix.png
│   ├── ordinal_coefficients.png
│   ├── ordinal_coefficients.csv
│   ├── ordinal_model_comparison.csv
│   ├── ordinal_confusion_matrix.png
│   └── wgcna/
│       ├── module_assignments.csv
│       ├── scale_free_fit.csv
│       ├── scale_free_fit.png
│       ├── module_dendrogram.png
│       ├── module_sizes.png
│       ├── module_trait_correlations.csv
│       ├── module_trait_correlations.png
│       ├── wgcna_parameters.json
│       └── hub_features.csv
└── multi_omics/
    ├── diablo_scores.png
    ├── diablo_vip_<block>.png
    ├── diablo_vip_scores.csv
    ├── block_correlations.csv
    ├── block_correlations.png
    ├── selected_features_<block>_comp<N>.csv
    ├── diablo_permutation_test.json
    ├── diablo_permutation_null.png
    ├── diablo_stability_<block>.csv  # DIABLO bootstrap stability per block
    ├── diablo_stability_<block>.png
    ├── method_convergence_grid.png   # Cross-method feature selection summary
    └── method_convergence_grid.svg
```

## Validation

sPLS-DA is validated against mixOmics R reference outputs:

```bash
python tests/validate_plsda.py
```

Checks: Spearman rank correlation of VIP scores, top-5 feature overlap, class separation ordering. Warnings for expected differences (sample alignment, high dimensionality).

WGCNA also has a lightweight smoke test:

```bash
python tests/validate_wgcna.py
```

Checks: adjacency/TOM sanity, returned metadata, and basic module detection behavior on the banana example dataset.

## Limitations

- **Small sample size** (n = 9): LOO CV is the only viable validation strategy; permutation p-values help assess significance but confidence intervals are wide.
- **WGCNA** was designed for n ≥ 15–20; results with n = 9 should be treated as exploratory.
- **Candidate-driver prioritisation** should be interpreted as exploratory evidence integration, not causal inference.
- **WGCNA implementation** follows the standard correlation → soft-threshold → TOM → clustering → eigengene → trait-association flow, but uses a pragmatic tree cut plus eigengene merging rather than the exact R `dynamicTreeCut` implementation.
- **sPLS-DA/DIABLO** implementation differs from mixOmics R (different SVD initialisation, deflation strategy) — validated to produce equivalent rankings.

## Dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `mord`, `shap`, `scipy`

## License

Apache 2.0