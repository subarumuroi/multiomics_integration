# Multi-Omics Integration

Consolidated Python package for multi-omics classification, combining single-layer and joint integration analyses. Implements three complementary methods — **sPLS-DA / DIABLO**, **Random Forest**, and **Ordinal Regression** — validated against mixOmics R reference outputs.

Consolidates the functionality of `downstream_analysis`, `ML_omics`, and `ML_multiomics` into one minimal, self-contained package.

## Project Structure

```
multiomics_integration/
├── data/                        # Banana ripeness dataset (4 omics layers)
│   ├── central_carbon.csv
│   ├── amino_acids.csv
│   ├── aromatics.csv
│   └── proteomics.csv
├── src/
│   ├── ingestion.py             # Load, impute, filter, scale (Pareto, half-min, log)
│   ├── visualization.py         # Scores plots, VIP bars, confusion matrices, consensus
│   ├── utils.py                 # I/O helpers, consensus feature identification
│   └── methods/
│       ├── plsda.py             # sPLS-DA (NIPALS + L1) and DIABLO (multi-block)
│       ├── random_forest.py     # RF with SHAP, permutation importance, permutation test
│       └── ordinal.py           # mord LogisticAT/IT/SE with ordinal class structure
├── examples/
│   └── banana_workflow.py       # End-to-end demo: single-omics → multi-omics → consensus
├── tests/
│   ├── validate_plsda.py        # Numerical validation vs mixOmics R reference
│   └── reference_data/          # R reference outputs for validation
├── results/                     # Generated outputs (gitignored)
└── requirements.txt
```

## Methods

| Method | Class / Function | Type | Description |
|--------|-----------------|------|-------------|
| sPLS-DA | `SPLSDA` | Single-block | Sparse PLS-DA via NIPALS with L1 soft-thresholding on loadings |
| DIABLO | `DIABLO` | Multi-block | Multi-block sPLS-DA with design matrix for inter-block covariance |
| Random Forest | `train_rf` | Single / concat | Balanced RF with SHAP and permutation importance |
| Ordinal Regression | `train_ordinal` | Single / concat | mord LogisticAT preserving class order (Green < Ripe < Overripe) |

## Preprocessing Pipeline

1. **Drop sparse features** (>50% missing)
2. **Half-min imputation** (MNAR-appropriate for metabolomics)
3. **Log₂ transform** (optional, stabilises variance)
4. **Pareto scaling** (divide by √std — balances high/low-abundance features)

## Installation

```bash
# On WSL, create venv on native filesystem (not /mnt/c/)
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
source ~/venv/bin/activate
python examples/banana_workflow.py
```

This runs all three methods on each omics layer (single-omics), then DIABLO + concatenated RF/ordinal (multi-omics), and finally identifies consensus features across methods. Results are saved to `results/`.

## Outputs

The workflow generates **66 files** across these categories:

| Output | Location | Format |
|--------|----------|--------|
| Scores plots | `results/single_omics/<layer>/` | PNG |
| VIP / importance bars | `results/single_omics/<layer>/` | PNG + CSV |
| Confusion matrices | `results/single_omics/<layer>/` | PNG |
| DIABLO multi-block scores | `results/multi_omics/` | PNG |
| Block correlations | `results/multi_omics/` | PNG + CSV |
| Method comparison | `results/method_comparison.csv` | CSV |
| Consensus features | `results/consensus_features.csv` | CSV + PNG |

## Validation

sPLS-DA is validated against mixOmics R reference (run via `ML_multiomics`):

```bash
python tests/validate_plsda.py
```

Results: **7/12 PASS, 5/12 WARN, 0 FAIL**. Warnings are due to expected differences (aromatics sample trimming 12→9, proteomics high dimensionality).

## Dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `mord`, `shap`, `umap-learn`