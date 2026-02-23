"""multiomics_integration: consolidated multi-omics analysis package."""

from .ingestion import (
    load_omics, load_all_layers, prepare_block, prepare_multiblock,
    encode_ordinal, decode_ordinal, align_blocks,
)
from .methods.plsda import SPLSDA, DIABLO, cross_validate_splsda, cross_validate_diablo
from .methods.random_forest import train_rf, cross_validate_rf, compute_shap_values
from .methods.ordinal import train_ordinal, cross_validate_ordinal, compare_ordinal_models
from .utils import create_results_dir, save_csv, save_json, find_consensus_features
