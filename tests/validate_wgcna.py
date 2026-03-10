#!/usr/bin/env python3
"""Lightweight WGCNA smoke test for the banana example dataset.

Checks:
  - adjacency and TOM matrix sanity
  - module assignment table shape
  - returned metadata fields
  - ability to detect at least one non-grey module on at least one block

This is intentionally a smoke test rather than a numerical gold-standard test,
since WGCNA output is sensitive to network choices and small-sample noise.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingestion import load_all_layers, prepare_block, encode_ordinal
from methods.wgcna import run_wgcna

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def check(label, status, detail=""):
    sym = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}[status]
    print(f"  [{sym}] {label}: {status}  {detail}")
    return status


def validate_wgcna():
    print("\n=== WGCNA Smoke Test ===")
    blocks = load_all_layers(str(ROOT / "data"))
    results = []
    detected_any = False

    for layer_name, df in blocks.items():
        X, y, feature_names, _ = prepare_block(df)
        y_enc = encode_ordinal(y)
        res = run_wgcna(
            X,
            y_enc,
            feature_names=feature_names,
            corr_method="spearman",
            network_type="unsigned",
        )

        adj = res["adjacency"]
        tom = res["tom"]
        modules = res["modules"]
        n_modules = len(set(modules["Module"]) - {0})
        detected_any = detected_any or (n_modules > 0)

        conditions = [
            np.allclose(adj, adj.T),
            np.allclose(np.diag(adj), 0.0),
            np.all((adj >= 0) & (adj <= 1)),
            np.allclose(tom, tom.T),
            np.allclose(np.diag(tom), 1.0),
            np.all((tom >= 0) & (tom <= 1)),
            len(modules) == X.shape[1],
            "network_type" in res and "power" in res and "eigengenes" in res,
        ]

        if all(conditions):
            status = check(
                layer_name,
                PASS,
                f"power={res['power']}, modules={n_modules}, network={res['network_type']}",
            )
        else:
            status = check(layer_name, FAIL, "matrix or metadata sanity check failed")
        results.append(status)

    if detected_any:
        results.append(check("module detection", PASS, "at least one omics layer produced a non-grey module"))
    else:
        results.append(check("module detection", WARN, "no non-grey modules detected on any layer"))

    n_fail = sum(r == FAIL for r in results)
    n_warn = sum(r == WARN for r in results)
    print(f"\nSummary: {len(results) - n_fail - n_warn} PASS, {n_warn} WARN, {n_fail} FAIL")

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    validate_wgcna()
