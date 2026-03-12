# Banana workflow report (ver.2)

## Setup

- Comparison: full DIABLO vs proteomics-only WGCNA-reduced DIABLO
- Workflow mode: ver2
- Permutations: 200
- Stability bootstraps: 100
- WGCNA threshold for reduction: > 100 features

## Method comparison

```text
         Layer                            Method  Accuracy                      Type  Perm_P_Value  MAE
central_carbon                           sPLS-DA  1.000000                    Single      0.014925  NaN
central_carbon                                RF  1.000000                    Single      0.014925  NaN
central_carbon                      Ordinal (AT)  1.000000                    Single      0.014925  0.0
   amino_acids                           sPLS-DA  1.000000                    Single      0.014925  NaN
   amino_acids                                RF  1.000000                    Single      0.014925  NaN
   amino_acids                      Ordinal (AT)  1.000000                    Single      0.014925  0.0
     aromatics                           sPLS-DA  1.000000                    Single      0.013245  NaN
     aromatics                                RF  1.000000                    Single      0.013245  NaN
     aromatics                      Ordinal (AT)  1.000000                    Single      0.013245  0.0
    proteomics                           sPLS-DA  0.666667                    Single      0.059701  NaN
    proteomics                                RF  1.000000                    Single      0.014925  NaN
    proteomics                      Ordinal (AT)  1.000000                    Single      0.014925  0.0
           all                            DIABLO  1.000000         Joint Integration      0.014925  NaN
           all                         Concat-RF  1.000000              Early Fusion      0.014925  NaN
           all                    Concat-Ordinal  1.000000              Early Fusion      0.013245  0.0
           all DIABLO (wgcna reduced proteomics)  1.000000 WGCNA-Reduced Integration      0.014925  NaN
```

## Reduction summary

- proteomics: 5319 -> 12 features

## DIABLO permutation tests

- Full DIABLO: accuracy=1.000, p-value=0.0149, permutations=200
- Proteomics-reduced DIABLO: accuracy=1.000, p-value=0.0149, permutations=200

## Stability selection

| Layer | Full DIABLO stable | Proteomics-reduced stable |
| --- | ---: | ---: |
| central_carbon | 8/33 | 5/33 |
| amino_acids | 10/21 | 10/21 |
| aromatics | 3/99 | 3/99 |
| proteomics | 0/5319 | 4/12 |

## Interpretation

This ver.2 run isolates WGCNA reduction to proteomics only, leaving the other blocks unchanged.
That makes it easier to judge whether dimensionality reduction is helping the hardest block without perturbing the smaller metabolomics layers.
