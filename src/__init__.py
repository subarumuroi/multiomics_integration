"""multiomics_integration: consolidated multi-omics analysis package."""

from .ingestion import (
    load_omics, load_all_layers, prepare_block, prepare_multiblock,
    encode_ordinal, align_blocks,
)
from .methods.plsda import (
    SPLSDA, DIABLO, cross_validate_splsda, cross_validate_diablo,
    permutation_test_splsda, permutation_test_diablo,
    stability_selection_splsda, stability_selection_diablo,
)
from .methods.random_forest import (
    train_rf, cross_validate_rf, compute_shap_values,
    compute_permutation_importance, permutation_test_rf,
)
from .methods.ordinal import train_ordinal, cross_validate_ordinal, compare_ordinal_models, get_coefficient_df, permutation_test_ordinal
from .methods.wgcna import run_wgcna
from .visualization import (
    plot_scores, plot_vip, plot_importance, plot_confusion_matrix,
    plot_diablo_scores, plot_block_correlations, plot_consensus_features,
    plot_stability, plot_permutation_null, plot_module_trait,
    plot_convergence_grid, save_fig,
)
from .utils import create_results_dir, save_csv, save_json, find_consensus_features
