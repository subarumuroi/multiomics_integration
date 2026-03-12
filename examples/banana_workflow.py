#!/usr/bin/env python3
"""
Banana multi-omics workflow: end-to-end demo.

Runs single-omics (sPLS-DA, RF, ordinal) and multi-omics (DIABLO) analyses
on the banana ripening dataset (metabolomics + proteomics).

Outputs: scores plots, VIP bars, confusion matrices, feature importance CSVs,
         method comparison, consensus features.
"""

import sys
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion import load_all_layers, prepare_block, prepare_multiblock, encode_ordinal
from methods.plsda import (
    SPLSDA, DIABLO, cross_validate_splsda, cross_validate_diablo,
    permutation_test_splsda, permutation_test_diablo,
    stability_selection_splsda, stability_selection_diablo,
)
from methods.random_forest import train_rf, cross_validate_rf, compute_shap_values, compute_permutation_importance, permutation_test_rf
from methods.ordinal import cross_validate_ordinal, compare_ordinal_models, get_coefficient_df, permutation_test_ordinal
from methods.wgcna import run_wgcna, reduce_by_wgcna
from visualization import (
    plot_scores, plot_vip, plot_importance, plot_confusion_matrix,
    plot_diablo_scores, plot_block_correlations, plot_consensus_features,
    plot_candidate_drivers,
    plot_stability, plot_permutation_null, plot_module_trait,
    plot_scale_free_fit, plot_wgcna_dendrogram, plot_module_sizes,
    plot_convergence_grid,
)
from utils import (
    create_results_dir, save_csv, save_json,
    find_consensus_features, integrate_wgcna_evidence,
)

warnings.filterwarnings("ignore")

DATA_DIR = ROOT / "data"
RESULTS_VERSION = "ver2"
RESULTS_DIR = create_results_dir(str(ROOT), f"results_{RESULTS_VERSION}")

# Runtime profiles to keep exploratory runs manageable on the banana dataset.
#
# `ver2` is the main comparison requested by the user:
# full DIABLO vs proteomics-only WGCNA-reduced DIABLO, with permutation and
# stability enabled but without the expensive SHAP / RF permutation-importance
# extras and without the abandoned all-layers-reduced branch.
WORKFLOW_MODE = "ver2"

MODE_CONFIGS = {
    "diagnostic": {
        "run_shap": False,
        "run_rf_permutation_importance": False,
        "run_permutation_tests": False,
        "run_stability_selection": False,
        "run_wgcna": True,
        "wgcna_min_features": 100,
        "run_reduced_diablo_proteomics": True,
        "n_permutations": 0,
        "n_bootstrap": 0,
    },
    "standard": {
        "run_shap": False,
        "run_rf_permutation_importance": False,
        "run_permutation_tests": True,
        "run_stability_selection": False,
        "run_wgcna": True,
        "wgcna_min_features": 100,
        "run_reduced_diablo_proteomics": True,
        "n_permutations": 100,
        "n_bootstrap": 0,
    },
    "full": {
        "run_shap": True,
        "run_rf_permutation_importance": True,
        "run_permutation_tests": True,
        "run_stability_selection": True,
        "run_wgcna": True,
        "wgcna_min_features": 0,
        "run_reduced_diablo_proteomics": True,
        "n_permutations": 200,
        "n_bootstrap": 100,
    },
    "ver2": {
        "run_shap": False,
        "run_rf_permutation_importance": False,
        "run_permutation_tests": True,
        "run_stability_selection": True,
        "run_wgcna": True,
        "wgcna_min_features": 100,
        "run_reduced_diablo_proteomics": True,
        "n_permutations": 200,
        "n_bootstrap": 100,
    },
}

if WORKFLOW_MODE not in MODE_CONFIGS:
    raise ValueError(f"Unknown WORKFLOW_MODE: {WORKFLOW_MODE}")

RUN_CONFIG = MODE_CONFIGS[WORKFLOW_MODE]


def run_single_omics(blocks, results_dir):
    """Run sPLS-DA, RF, and ordinal on each omics layer independently."""
    all_importance = {}
    summary_rows = []
    wgcna_results = {}
    
    for layer_name, df in blocks.items():
        print(f"\n{'='*60}")
        print(f"  Single-omics: {layer_name}")
        print(f"{'='*60}")
        
        layer_dir = results_dir / "single_omics" / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        X, y, feature_names, sample_names = prepare_block(df)
        y_enc = encode_ordinal(y)
        
        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # --- sPLS-DA (sparse) ---
        splsda_keepX = [min(5, X.shape[1]), min(5, X.shape[1])]
        print(f"  Running sPLS-DA (keepX={splsda_keepX})...")
        splsda = SPLSDA(n_components=2, keepX=splsda_keepX)
        splsda.fit(X, y, feature_names=feature_names)
        
        scores_df = splsda.get_scores_df(sample_names)
        vip_df = splsda.get_vip_df()
        
        plot_scores(scores_df, y, title=f"sPLS-DA: {layer_name}",
                    save_path=layer_dir / "splsda_scores.png")
        plot_vip(vip_df, top_n=15, title=f"sPLS-DA VIP: {layer_name}",
                 save_path=layer_dir / "splsda_vip.png")
        save_csv(vip_df, layer_dir / "splsda_vip_scores.csv")
        
        cv_splsda = cross_validate_splsda(X, y, n_components=2, keepX=splsda_keepX)
        print(f"  sPLS-DA LOO accuracy: {cv_splsda['accuracy']:.3f}")
        summary_rows.append({"Layer": layer_name, "Method": "sPLS-DA", "Accuracy": cv_splsda["accuracy"], "Type": "Single"})
        
        all_importance[f"{layer_name}_splsda"] = vip_df.rename(columns={"VIP": "Score"})[["Feature", "Score"]]
        
        # --- Random Forest ---
        print("  Running Random Forest...")
        rf_model, rf_imp = train_rf(X, y, feature_names=feature_names)
        cv_rf = cross_validate_rf(X, y)
        
        plot_importance(rf_imp, top_n=15, title=f"RF Importance: {layer_name}",
                        save_path=layer_dir / "rf_importance.png")
        plot_confusion_matrix(cv_rf["confusion_matrix"], title=f"RF CM: {layer_name}",
                              save_path=layer_dir / "rf_confusion_matrix.png")
        save_csv(rf_imp, layer_dir / "rf_feature_importance.csv")
        
        # SHAP
        if RUN_CONFIG["run_shap"]:
            try:
                shap_vals, shap_df = compute_shap_values(rf_model, X, feature_names=feature_names)
                save_csv(shap_df, layer_dir / "rf_shap_importance.csv")
            except Exception as e:
                print(f"  SHAP failed: {e}")
        
        # Permutation importance
        if RUN_CONFIG["run_rf_permutation_importance"]:
            perm_imp = compute_permutation_importance(rf_model, X, y, feature_names=feature_names)
            save_csv(perm_imp, layer_dir / "rf_permutation_importance.csv")
        
        print(f"  RF LOO accuracy: {cv_rf['accuracy']:.3f}")
        summary_rows.append({"Layer": layer_name, "Method": "RF", "Accuracy": cv_rf["accuracy"], "Type": "Single"})
        
        all_importance[f"{layer_name}_rf"] = rf_imp.rename(columns={"Importance": "Score"})[["Feature", "Score"]]
        
        # RF Permutation Test
        if RUN_CONFIG["run_permutation_tests"]:
            print(f"  Running RF permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
            perm_rf = permutation_test_rf(X, y, n_permutations=RUN_CONFIG["n_permutations"],
                                           early_stop=True, min_perms=50, check_every=25)
            save_json({
                "true_accuracy": perm_rf["true_accuracy"],
                "p_value": perm_rf["p_value"],
                "mean_null": perm_rf["mean_null"],
                "std_null": perm_rf["std_null"],
                "n_permutations_run": perm_rf["n_permutations_run"],
                "stopped_early": perm_rf["stopped_early"],
            }, layer_dir / "rf_permutation_test.json")
            plot_permutation_null(perm_rf, title=f"RF Permutation: {layer_name}",
                                  save_path=layer_dir / "rf_permutation_null.png")
            early_note = f" (stopped early at {perm_rf['n_permutations_run']})" if perm_rf["stopped_early"] else ""
            print(f"  RF permutation p-value: {perm_rf['p_value']:.4f}{early_note}")
            summary_rows[-1]["Perm_P_Value"] = perm_rf["p_value"]
        
        # --- Ordinal Regression ---
        print("  Running Ordinal Regression...")
        cv_ord = cross_validate_ordinal(X, y_enc)
        coef_df = get_coefficient_df(cv_ord["mean_coefficients"], feature_names=feature_names)
        
        plot_importance(coef_df, top_n=15, title=f"Ordinal Coefficients: {layer_name}",
                        value_col="Abs_Coefficient", save_path=layer_dir / "ordinal_coefficients.png")
        plot_confusion_matrix(cv_ord["confusion_matrix"], title=f"Ordinal CM: {layer_name}",
                              save_path=layer_dir / "ordinal_confusion_matrix.png")
        save_csv(coef_df, layer_dir / "ordinal_coefficients.csv")
        
        # Compare AT/IT/SE
        comparison = compare_ordinal_models(X, y_enc)
        save_csv(comparison, layer_dir / "ordinal_model_comparison.csv")
        
        print(f"  Ordinal LOO accuracy: {cv_ord['accuracy']:.3f}, MAE: {cv_ord['mae']:.3f}")
        summary_rows.append({"Layer": layer_name, "Method": "Ordinal (AT)", "Accuracy": cv_ord["accuracy"], 
                             "MAE": cv_ord["mae"], "Type": "Single"})
        
        all_importance[f"{layer_name}_ordinal"] = coef_df.rename(columns={"Abs_Coefficient": "Score"})[["Feature", "Score"]]
        
        # Ordinal Permutation Test
        if RUN_CONFIG["run_permutation_tests"]:
            print(f"  Running Ordinal permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
            perm_ord = permutation_test_ordinal(X, y_enc, n_permutations=RUN_CONFIG["n_permutations"],
                                                 early_stop=True, min_perms=50, check_every=25)
            save_json({
                "true_accuracy": perm_ord["true_accuracy"],
                "p_value": perm_ord["p_value"],
                "mean_null": perm_ord["mean_null"],
                "std_null": perm_ord["std_null"],
                "n_permutations_run": perm_ord["n_permutations_run"],
                "stopped_early": perm_ord["stopped_early"],
            }, layer_dir / "ordinal_permutation_test.json")
            plot_permutation_null(perm_ord, title=f"Ordinal Permutation: {layer_name}",
                                  save_path=layer_dir / "ordinal_permutation_null.png")
            early_note = f" (stopped early at {perm_ord['n_permutations_run']})" if perm_ord["stopped_early"] else ""
            print(f"  Ordinal permutation p-value: {perm_ord['p_value']:.4f}{early_note}")
            summary_rows[-1]["Perm_P_Value"] = perm_ord["p_value"]
        
        # --- Stability Selection for sPLS-DA ---
        if RUN_CONFIG["run_stability_selection"]:
            print(f"  Running sPLS-DA stability selection ({RUN_CONFIG['n_bootstrap']} bootstraps)...")
            stability_df = stability_selection_splsda(X, y, feature_names=feature_names,
                                                       n_components=2, keepX=splsda_keepX,
                                                       n_bootstrap=RUN_CONFIG["n_bootstrap"])
            save_csv(stability_df, layer_dir / "splsda_stability.csv")
            plot_stability(stability_df, top_n=20, title=f"Stability Selection: {layer_name}",
                           save_path=layer_dir / "splsda_stability.png")
            n_stable = stability_df["Stable"].sum()
            print(f"  Stable features (freq >= 0.8): {n_stable}/{len(stability_df)}")
        
        # --- Permutation Test for sPLS-DA ---
        if RUN_CONFIG["run_permutation_tests"]:
            print(f"  Running sPLS-DA permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
            perm_result = permutation_test_splsda(X, y, n_components=2, keepX=splsda_keepX,
                                                   n_permutations=RUN_CONFIG["n_permutations"],
                                                   early_stop=True, min_perms=50, check_every=25)
            save_json({
                "true_accuracy": perm_result["true_accuracy"],
                "p_value": perm_result["p_value"],
                "mean_null": perm_result["mean_null"],
                "std_null": perm_result["std_null"],
                "n_permutations_run": perm_result["n_permutations_run"],
                "stopped_early": perm_result["stopped_early"],
            }, layer_dir / "splsda_permutation_test.json")
            plot_permutation_null(perm_result, title=f"sPLS-DA Permutation: {layer_name}",
                                  save_path=layer_dir / "splsda_permutation_null.png")
            early_note = f" (stopped early at {perm_result['n_permutations_run']})" if perm_result["stopped_early"] else ""
            print(f"  sPLS-DA permutation p-value: {perm_result['p_value']:.4f}{early_note}")
            for row in summary_rows:
                if row["Layer"] == layer_name and row["Method"] == "sPLS-DA":
                    row["Perm_P_Value"] = perm_result["p_value"]
        
        # --- WGCNA ---
        if RUN_CONFIG["run_wgcna"]:
            wgcna_dir = layer_dir / "wgcna"
            if X.shape[1] <= RUN_CONFIG["wgcna_min_features"]:
                if wgcna_dir.exists():
                    shutil.rmtree(wgcna_dir)
                print(
                    f"  Skipping WGCNA for {layer_name}: "
                    f"{X.shape[1]} features (<= {RUN_CONFIG['wgcna_min_features']})"
                )
                continue

            print(f"  Running WGCNA (n={X.shape[0]}, p={X.shape[1]})...")
            wgcna_result = run_wgcna(
                X,
                y_enc,
                feature_names=feature_names,
                corr_method="spearman",
                network_type="unsigned",
            )
            
            wgcna_dir.mkdir(parents=True, exist_ok=True)
            save_csv(wgcna_result["modules"], wgcna_dir / "module_assignments.csv")
            save_csv(wgcna_result["module_trait"], wgcna_dir / "module_trait_correlations.csv")
            if wgcna_result["scale_free_fit"] is not None:
                selection = wgcna_result["scale_free_selection"] or {}
                save_csv(wgcna_result["scale_free_fit"], wgcna_dir / "scale_free_fit.csv")
                plot_scale_free_fit(
                    wgcna_result["scale_free_fit"],
                    selected_power=selection.get("power"),
                    selected_r2=selection.get("r_squared"),
                    target_r2=selection.get("target_r2", 0.8),
                    threshold_met=selection.get("threshold_met"),
                    title=f"Scale-Free Fit: {layer_name}",
                    save_path=wgcna_dir / "scale_free_fit.png",
                )
            save_json({
                "power": wgcna_result["power"],
                "network_type": wgcna_result["network_type"],
                "module_cut_height": wgcna_result["module_cut_height"],
                "merge_cut_height": wgcna_result["merge_cut_height"],
                "scale_free_selection": wgcna_result["scale_free_selection"],
            }, wgcna_dir / "wgcna_parameters.json")
            plot_wgcna_dendrogram(
                wgcna_result["linkage"],
                wgcna_result["modules"],
                title=f"WGCNA Dendrogram: {layer_name}",
                save_path=wgcna_dir / "module_dendrogram.png",
            )
            plot_module_sizes(
                wgcna_result["modules"],
                title=f"Module Sizes: {layer_name}",
                save_path=wgcna_dir / "module_sizes.png",
            )
            if not wgcna_result["module_trait"].empty:
                plot_module_trait(wgcna_result["module_trait"],
                                 title=f"Module-Trait: {layer_name}",
                                 save_path=wgcna_dir / "module_trait_correlations.png")
            if not wgcna_result["hubs"].empty:
                save_csv(wgcna_result["hubs"], wgcna_dir / "hub_features.csv")
                hub_features = wgcna_result["hubs"][wgcna_result["hubs"]["Is_Hub"]]
                print(f"  WGCNA: {len(set(wgcna_result['modules']['Module']) - {0})} modules, "
                      f"{len(hub_features)} hub features")
                if not wgcna_result["module_trait"].empty:
                    top_mod = wgcna_result["module_trait"].iloc[0]
                    print(f"  Top module-trait: Module {int(top_mod['Module'])} "
                          f"(r={top_mod['Correlation']:.3f}, p={top_mod['P_Value']:.4f})")

            wgcna_results[layer_name] = wgcna_result

    return summary_rows, all_importance, wgcna_results


def run_multi_omics(blocks, results_dir):
    """Run DIABLO (multi-block sPLS-DA) across all layers."""
    print(f"\n{'='*60}")
    print(f"  Multi-omics: DIABLO")
    print(f"{'='*60}")
    
    multi_dir = results_dir / "multi_omics"
    multi_dir.mkdir(parents=True, exist_ok=True)
    
    X_blocks, y, feature_names, sample_names = prepare_multiblock(blocks)
    
    for name, X in X_blocks.items():
        print(f"  {name}: {X.shape[1]} features")
    
    # Conservative keepX for small n
    keepX = {}
    for name, X in X_blocks.items():
        p = X.shape[1]
        keepX[name] = [min(5, p), min(5, p)]
    
    # Fit DIABLO
    print("  Fitting DIABLO...")
    diablo = DIABLO(n_components=2, keepX=keepX, design=0.1)
    diablo.fit(X_blocks, y, feature_names=feature_names)
    
    # Scores plots
    plot_diablo_scores(diablo, y, save_dir=multi_dir)
    
    # Block correlations
    plot_block_correlations(diablo.correlations_, save_path=multi_dir / "block_correlations.png")
    save_csv(diablo.correlations_, multi_dir / "block_correlations.csv", index=True)
    print(f"  Block correlations:\n{diablo.correlations_.to_string()}")
    
    # VIP per block
    all_vip = diablo.get_all_vip_df()
    save_csv(all_vip, multi_dir / "diablo_vip_scores.csv")
    
    for name in diablo.block_names_:
        vip_df = diablo.get_vip_df(name)
        plot_vip(vip_df, top_n=min(15, len(vip_df)), title=f"DIABLO VIP: {name}",
                 save_path=multi_dir / f"diablo_vip_{name}.png")
    
    # Selected features per block per component
    for name in diablo.block_names_:
        for comp in range(diablo.n_components_):
            sel = diablo.get_selected_features(name, component=comp)
            if not sel.empty:
                save_csv(sel, multi_dir / f"selected_features_{name}_comp{comp+1}.csv")
    
    # Cross-validation
    print("  Running DIABLO LOO CV...")
    cv_diablo = cross_validate_diablo(X_blocks, y, n_components=2, keepX=keepX, design=0.1)
    print(f"  DIABLO LOO accuracy: {cv_diablo['accuracy']:.3f}")
    
    # Also run RF on concatenated (early fusion baseline)
    print("  Running concatenated RF (early fusion baseline)...")
    X_concat = np.hstack([X_blocks[name] for name in diablo.block_names_])
    
    cv_concat = cross_validate_rf(X_concat, y)
    print(f"  Concatenated RF LOO accuracy: {cv_concat['accuracy']:.3f}")
    perm_concat_rf = None
    if RUN_CONFIG["run_permutation_tests"]:
        print(f"  Running concatenated RF permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
        perm_concat_rf = permutation_test_rf(X_concat, y, n_permutations=RUN_CONFIG["n_permutations"],
                                             early_stop=True, min_perms=50, check_every=25)
        save_json({
            "true_accuracy": perm_concat_rf["true_accuracy"],
            "p_value": perm_concat_rf["p_value"],
            "mean_null": perm_concat_rf["mean_null"],
            "std_null": perm_concat_rf["std_null"],
            "n_permutations_run": perm_concat_rf["n_permutations_run"],
            "stopped_early": perm_concat_rf["stopped_early"],
        }, multi_dir / "concat_rf_permutation_test.json")
        plot_permutation_null(perm_concat_rf, title="Concat RF Permutation Test",
                              save_path=multi_dir / "concat_rf_permutation_null.png")
        early_note = f" (stopped early at {perm_concat_rf['n_permutations_run']})" if perm_concat_rf["stopped_early"] else ""
        print(f"  Concat-RF permutation p-value: {perm_concat_rf['p_value']:.4f}{early_note}")
    
    # Also run ordinal on concatenated
    print("  Running concatenated ordinal regression...")
    y_enc = encode_ordinal(y)
    cv_ord_multi = cross_validate_ordinal(X_concat, y_enc)
    print(f"  Concatenated ordinal LOO accuracy: {cv_ord_multi['accuracy']:.3f}, MAE: {cv_ord_multi['mae']:.3f}")
    perm_concat_ord = None
    if RUN_CONFIG["run_permutation_tests"]:
        print(f"  Running concatenated ordinal permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
        perm_concat_ord = permutation_test_ordinal(X_concat, y_enc, n_permutations=RUN_CONFIG["n_permutations"],
                                                   early_stop=True, min_perms=50, check_every=25)
        save_json({
            "true_accuracy": perm_concat_ord["true_accuracy"],
            "p_value": perm_concat_ord["p_value"],
            "mean_null": perm_concat_ord["mean_null"],
            "std_null": perm_concat_ord["std_null"],
            "n_permutations_run": perm_concat_ord["n_permutations_run"],
            "stopped_early": perm_concat_ord["stopped_early"],
        }, multi_dir / "concat_ordinal_permutation_test.json")
        plot_permutation_null(perm_concat_ord, title="Concat Ordinal Permutation Test",
                              save_path=multi_dir / "concat_ordinal_permutation_null.png")
        early_note = f" (stopped early at {perm_concat_ord['n_permutations_run']})" if perm_concat_ord["stopped_early"] else ""
        print(f"  Concat-Ordinal permutation p-value: {perm_concat_ord['p_value']:.4f}{early_note}")
    
    # --- DIABLO Permutation Test ---
    perm_diablo = None
    if RUN_CONFIG["run_permutation_tests"]:
        print(f"  Running DIABLO permutation test (up to {RUN_CONFIG['n_permutations']} perms, early stopping)...")
        perm_diablo = permutation_test_diablo(X_blocks, y, n_components=2, keepX=keepX,
                                               design=0.1, n_permutations=RUN_CONFIG["n_permutations"],
                                               early_stop=True, min_perms=50, check_every=25)
        save_json({
            "true_accuracy": perm_diablo["true_accuracy"],
            "p_value": perm_diablo["p_value"],
            "mean_null": perm_diablo["mean_null"],
            "std_null": perm_diablo["std_null"],
            "n_permutations_run": perm_diablo["n_permutations_run"],
            "stopped_early": perm_diablo["stopped_early"],
        }, multi_dir / "diablo_permutation_test.json")
        plot_permutation_null(perm_diablo, title="DIABLO Permutation Test",
                              save_path=multi_dir / "diablo_permutation_null.png")
        early_note = f" (stopped early at {perm_diablo['n_permutations_run']})" if perm_diablo["stopped_early"] else ""
        print(f"  DIABLO permutation p-value: {perm_diablo['p_value']:.4f}{early_note}")
    
    summary_rows = [
        {"Layer": "all", "Method": "DIABLO", "Accuracy": cv_diablo["accuracy"],
            "Perm_P_Value": perm_diablo["p_value"] if perm_diablo is not None else np.nan, "Type": "Joint Integration"},
        {"Layer": "all", "Method": "Concat-RF", "Accuracy": cv_concat["accuracy"],
            "Perm_P_Value": perm_concat_rf["p_value"] if perm_concat_rf is not None else np.nan, "Type": "Early Fusion"},
        {"Layer": "all", "Method": "Concat-Ordinal", "Accuracy": cv_ord_multi["accuracy"], 
            "MAE": cv_ord_multi["mae"], "Perm_P_Value": perm_concat_ord["p_value"] if perm_concat_ord is not None else np.nan, "Type": "Early Fusion"},
    ]
    
    importance_dfs = {}
    for name in diablo.block_names_:
        vip_df = diablo.get_vip_df(name)
        importance_dfs[f"diablo_{name}"] = vip_df.rename(columns={"VIP": "Score"})[["Feature", "Score"]]
    
    # --- DIABLO Stability Selection ---
    if RUN_CONFIG["run_stability_selection"]:
        print(f"  Running DIABLO stability selection ({RUN_CONFIG['n_bootstrap']} bootstraps)...")
        diablo_stab = stability_selection_diablo(
            X_blocks, y, feature_names=feature_names,
            n_components=2, keepX=keepX, design=0.1, n_bootstrap=RUN_CONFIG["n_bootstrap"],
        )
        for name, sdf in diablo_stab.items():
            save_csv(sdf, multi_dir / f"diablo_stability_{name}.csv")
            plot_stability(sdf, top_n=20, title=f"DIABLO Stability: {name}",
                           save_path=multi_dir / f"diablo_stability_{name}.png")
            n_stable = sdf["Stable"].sum()
            print(f"    {name}: {n_stable}/{len(sdf)} stable features")
    
    return summary_rows, importance_dfs


def run_reduced_diablo(blocks, wgcna_results, results_dir,
                       min_features_for_reduction=0, label=None):
    """Run DIABLO on WGCNA-reduced blocks for stabilised feature selection.

    Parameters
    ----------
    min_features_for_reduction : int
        Only reduce layers with more features than this threshold.
        Layers at or below this value pass through with raw features.
        Default 0 means reduce all layers.
    label : str or None
        Label suffix for output directory and method name.
        Default None uses 'wgcna_reduced'.
    """
    dir_name = label or "wgcna_reduced"
    method_label = f"DIABLO ({dir_name.replace('_', ' ')})"

    print(f"\n{'='*60}")
    print(f"  {method_label}")
    print(f"{'='*60}")

    reduced_dir = results_dir / "multi_omics" / dir_name
    reduced_dir.mkdir(parents=True, exist_ok=True)

    X_blocks, y, feature_names_full, sample_names = prepare_multiblock(blocks)
    X_reduced_blocks = {}
    reduced_feature_names = {}
    reduction_meta = {}

    for layer_name in X_blocks:
        p = X_blocks[layer_name].shape[1]

        # Skip reduction for small layers when threshold is set
        if p <= min_features_for_reduction:
            print(f"  {layer_name}: {p} features (<= {min_features_for_reduction}), keeping raw")
            X_reduced_blocks[layer_name] = X_blocks[layer_name]
            reduced_feature_names[layer_name] = feature_names_full[layer_name]
            continue

        if layer_name not in wgcna_results:
            print(f"  {layer_name}: no WGCNA result, keeping raw")
            X_reduced_blocks[layer_name] = X_blocks[layer_name]
            reduced_feature_names[layer_name] = feature_names_full[layer_name]
            continue

        X_red, feat_names, meta = reduce_by_wgcna(
            X_blocks[layer_name], wgcna_results[layer_name],
            strategy="eigengenes_and_hubs",
        )

        if X_red.shape[1] == 0:
            print(f"  {layer_name}: no modules detected, using raw features")
            X_reduced_blocks[layer_name] = X_blocks[layer_name]
            reduced_feature_names[layer_name] = feature_names_full[layer_name]
        else:
            X_reduced_blocks[layer_name] = X_red
            reduced_feature_names[layer_name] = feat_names
            reduction_meta[layer_name] = meta
            print(f"  {layer_name}: {meta['n_original']} -> {meta['n_reduced']} features")

    if not X_reduced_blocks:
        print("  No blocks available for reduced DIABLO, skipping.")
        return []

    save_json(reduction_meta, reduced_dir / "reduction_meta.json")

    # keepX: same conservative sparsity as full DIABLO for fair comparison
    keepX = {}
    for name, X_r in X_reduced_blocks.items():
        p = X_r.shape[1]
        keepX[name] = [min(5, p), min(5, p)]

    # Fit reduced DIABLO
    print(f"  Fitting {method_label}...")
    diablo = DIABLO(n_components=2, keepX=keepX, design=0.1)
    diablo.fit(X_reduced_blocks, y, feature_names=reduced_feature_names)

    plot_diablo_scores(diablo, y, save_dir=reduced_dir)
    plot_block_correlations(diablo.correlations_,
                            title=f"Block Correlations ({method_label})",
                            save_path=reduced_dir / "block_correlations.png")
    save_csv(diablo.correlations_, reduced_dir / "block_correlations.csv", index=True)

    # VIP scores
    all_vip = diablo.get_all_vip_df()
    save_csv(all_vip, reduced_dir / "diablo_vip_scores.csv")
    for name in diablo.block_names_:
        vip_df = diablo.get_vip_df(name)
        plot_vip(vip_df, top_n=min(15, len(vip_df)),
                 title=f"{method_label} VIP: {name}",
                 save_path=reduced_dir / f"diablo_vip_{name}.png")

    # LOO CV
    print(f"  Running {method_label} LOO CV...")
    cv = cross_validate_diablo(X_reduced_blocks, y, n_components=2,
                               keepX=keepX, design=0.1)
    print(f"  {method_label} LOO accuracy: {cv['accuracy']:.3f}")

    # Permutation test
    perm = None
    if RUN_CONFIG["run_permutation_tests"]:
        print(f"  Running {method_label} permutation test (up to {RUN_CONFIG['n_permutations']} perms)...")
        perm = permutation_test_diablo(X_reduced_blocks, y, n_components=2,
                                       keepX=keepX, design=0.1,
                                       n_permutations=RUN_CONFIG["n_permutations"],
                                       early_stop=True, min_perms=50,
                                       check_every=25)
        save_json({
            "true_accuracy": perm["true_accuracy"],
            "p_value": perm["p_value"],
            "mean_null": perm["mean_null"],
            "std_null": perm["std_null"],
            "n_permutations_run": perm["n_permutations_run"],
            "stopped_early": perm["stopped_early"],
        }, reduced_dir / "diablo_permutation_test.json")
        plot_permutation_null(perm, title=f"{method_label} Permutation Test",
                              save_path=reduced_dir / "diablo_permutation_null.png")
        early_note = (f" (stopped early at {perm['n_permutations_run']})"
                      if perm["stopped_early"] else "")
        print(f"  {method_label} permutation p-value: {perm['p_value']:.4f}{early_note}")

    # Stability selection
    if RUN_CONFIG["run_stability_selection"]:
        print(f"  Running {method_label} stability selection ({RUN_CONFIG['n_bootstrap']} bootstraps)...")
        stab = stability_selection_diablo(
            X_reduced_blocks, y, feature_names=reduced_feature_names,
            n_components=2, keepX=keepX, design=0.1, n_bootstrap=RUN_CONFIG["n_bootstrap"],
        )
        for name, sdf in stab.items():
            save_csv(sdf, reduced_dir / f"diablo_stability_{name}.csv")
            plot_stability(sdf, top_n=20,
                           title=f"{method_label} Stability: {name}",
                           save_path=reduced_dir / f"diablo_stability_{name}.png")
            n_stable = sdf["Stable"].sum()
            print(f"    {name}: {n_stable}/{len(sdf)} stable features")

    summary_rows = [
        {"Layer": "all", "Method": method_label,
         "Accuracy": cv["accuracy"],
            "Perm_P_Value": perm["p_value"] if perm is not None else np.nan,
         "Type": "WGCNA-Reduced Integration"},
    ]
    return summary_rows


def write_ver2_report(results_dir, all_summary, reduction_dir_name):
    """Write a compact markdown report for the ver2 comparison."""
    results_dir = Path(results_dir)
    multi_dir = results_dir / "multi_omics"
    reduced_dir = multi_dir / reduction_dir_name

    lines = [
        "# Banana workflow report (ver.2)",
        "",
        "## Setup",
        "",
        "- Comparison: full DIABLO vs proteomics-only WGCNA-reduced DIABLO",
        f"- Workflow mode: {WORKFLOW_MODE}",
        f"- Permutations: {RUN_CONFIG['n_permutations'] if RUN_CONFIG['run_permutation_tests'] else 0}",
        f"- Stability bootstraps: {RUN_CONFIG['n_bootstrap'] if RUN_CONFIG['run_stability_selection'] else 0}",
        f"- WGCNA threshold for reduction: > {RUN_CONFIG['wgcna_min_features']} features",
        "",
        "## Method comparison",
        "",
        "```text",
        all_summary.to_string(index=False),
        "```",
        "",
    ]

    reduction_meta_path = reduced_dir / "reduction_meta.json"
    if reduction_meta_path.exists():
        import json
        with open(reduction_meta_path, "r") as f:
            reduction_meta = json.load(f)
        lines.extend([
            "## Reduction summary",
            "",
        ])
        for layer_name, meta in reduction_meta.items():
            lines.append(f"- {layer_name}: {meta['n_original']} -> {meta['n_reduced']} features")
        lines.append("")

    full_perm_path = multi_dir / "diablo_permutation_test.json"
    reduced_perm_path = reduced_dir / "diablo_permutation_test.json"
    if full_perm_path.exists() or reduced_perm_path.exists():
        import json
        lines.extend([
            "## DIABLO permutation tests",
            "",
        ])
        if full_perm_path.exists():
            with open(full_perm_path, "r") as f:
                perm = json.load(f)
            lines.append(
                f"- Full DIABLO: accuracy={perm['true_accuracy']:.3f}, p-value={perm['p_value']:.4f}, "
                f"permutations={perm['n_permutations_run']}"
            )
        if reduced_perm_path.exists():
            with open(reduced_perm_path, "r") as f:
                perm = json.load(f)
            lines.append(
                f"- Proteomics-reduced DIABLO: accuracy={perm['true_accuracy']:.3f}, p-value={perm['p_value']:.4f}, "
                f"permutations={perm['n_permutations_run']}"
            )
        lines.append("")

    if RUN_CONFIG["run_stability_selection"]:
        lines.extend([
            "## Stability selection",
            "",
            "| Layer | Full DIABLO stable | Proteomics-reduced stable |",
            "| --- | ---: | ---: |",
        ])
        for layer_name in ["central_carbon", "amino_acids", "aromatics", "proteomics"]:
            full_path = multi_dir / f"diablo_stability_{layer_name}.csv"
            reduced_path = reduced_dir / f"diablo_stability_{layer_name}.csv"
            full_text = "n/a"
            reduced_text = "n/a"
            if full_path.exists():
                full_df = pd.read_csv(full_path)
                full_text = f"{int(full_df['Stable'].sum())}/{len(full_df)}"
            if reduced_path.exists():
                reduced_df = pd.read_csv(reduced_path)
                reduced_text = f"{int(reduced_df['Stable'].sum())}/{len(reduced_df)}"
            lines.append(f"| {layer_name} | {full_text} | {reduced_text} |")
        lines.append("")

    lines.extend([
        "## Interpretation",
        "",
        "This ver.2 run isolates WGCNA reduction to proteomics only, leaving the other blocks unchanged.",
        "That makes it easier to judge whether dimensionality reduction is helping the hardest block without perturbing the smaller metabolomics layers.",
        "",
    ])

    report_path = results_dir / "report_ver.2.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("Loading banana ripening dataset...")
    print(f"Workflow mode: {WORKFLOW_MODE}")
    print(
        "Enabled: "
        f"WGCNA={RUN_CONFIG['run_wgcna']}, "
        f"wgcna_min_features={RUN_CONFIG['wgcna_min_features']}, "
        f"permutations={RUN_CONFIG['run_permutation_tests']}, "
        f"stability={RUN_CONFIG['run_stability_selection']}, "
        f"reduced_proteomics={RUN_CONFIG['run_reduced_diablo_proteomics']}"
    )
    blocks = load_all_layers(str(DATA_DIR))
    print(f"Loaded {len(blocks)} omics layers: {list(blocks.keys())}")
    
    # Single-omics
    single_summary, single_importance, wgcna_results = run_single_omics(blocks, RESULTS_DIR)
    
    # Multi-omics
    multi_summary, multi_importance = run_multi_omics(blocks, RESULTS_DIR)

    # --- WGCNA-reduced DIABLO (proteomics only in ver2 mode) ---
    reduced_summary = []
    reduced_dir_name = "wgcna_reduced_proteomics"
    if RUN_CONFIG["run_reduced_diablo_proteomics"] and RUN_CONFIG["run_wgcna"]:
        reduced_summary = run_reduced_diablo(
            blocks, wgcna_results, RESULTS_DIR,
            min_features_for_reduction=RUN_CONFIG["wgcna_min_features"],
            label=reduced_dir_name,
        )
    
    # Combined summary
    all_summary = pd.DataFrame(
        single_summary + multi_summary + reduced_summary
    )
    save_csv(all_summary, RESULTS_DIR / "method_comparison.csv")
    save_csv(all_summary, RESULTS_DIR / "method_comparison_ver2.csv")
    write_ver2_report(RESULTS_DIR, all_summary, reduced_dir_name)
    print(f"\n{'='*60}")
    print("  Method Comparison")
    print(f"{'='*60}")
    print(all_summary.to_string(index=False))
    
    # Consensus features
    all_importance = {**single_importance, **multi_importance}
    consensus = find_consensus_features(all_importance, top_n=15)
    if not consensus.empty:
        if RUN_CONFIG["run_wgcna"]:
            consensus = integrate_wgcna_evidence(consensus, RESULTS_DIR)
        save_csv(consensus, RESULTS_DIR / "consensus_features.csv")
        plot_consensus_features(consensus, save_path=RESULTS_DIR / "consensus_features.png")
        candidate_cols = [
            "Feature", "n_methods", "methods",
        ]
        if RUN_CONFIG["run_wgcna"]:
            candidate_cols = [
                "Feature", "layer", "n_methods", "wgcna_supported", "wgcna_is_hub",
                "wgcna_module", "wgcna_module_trait_correlation", "wgcna_module_trait_p_value",
                "wgcna_hub_score", "integrated_evidence_score", "methods",
            ]
        candidate_features = consensus[candidate_cols].copy()
        save_csv(candidate_features, RESULTS_DIR / "candidate_driver_features.csv")
        plot_candidate_drivers(
            candidate_features,
            top_n=25,
            title="Candidate Driver Summary",
            save_path=RESULTS_DIR / "candidate_driver_features.png",
        )
        print(f"\nConsensus features (top across methods):")
        print(consensus.to_string(index=False))

        # Method convergence grid (3/4 and 4/4 features)
        if RUN_CONFIG["run_stability_selection"]:
            print("\n  Generating method convergence grid...")
            splsda_stability_map = {}
            for layer_name in blocks:
                stab_path = RESULTS_DIR / "single_omics" / layer_name / "splsda_stability.csv"
                if stab_path.exists():
                    stab_df = pd.read_csv(stab_path)
                    for _, sr in stab_df.iterrows():
                        splsda_stability_map[sr["Feature"]] = sr["Selection_Frequency"]
            diablo_stability_map = {}
            for layer_name in blocks:
                stab_path = RESULTS_DIR / "multi_omics" / f"diablo_stability_{layer_name}.csv"
                if stab_path.exists():
                    stab_df = pd.read_csv(stab_path)
                    for _, sr in stab_df.iterrows():
                        diablo_stability_map[sr["Feature"]] = sr["Selection_Frequency"]
            plot_convergence_grid(
                consensus,
                splsda_stability_map=splsda_stability_map,
                diablo_stability_map=diablo_stability_map,
                save_path=RESULTS_DIR / "multi_omics" / "method_convergence_grid.png",
            )
            print("  Saved method_convergence_grid.png + .svg")
        else:
            print("  Skipping convergence grid because stability selection is disabled")
    
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
