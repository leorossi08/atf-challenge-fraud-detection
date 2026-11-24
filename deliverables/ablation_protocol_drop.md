# Attribution of the metric drop (same models, one factor changed per step)

Random-split rows are the mean of 3 seeds; walk-forward rows use all Jul–Dec 2025 out-of-time predictions.

## logistic_regression

| Step | Condition | PR-AUC | Δ PR-AUC | ROC-AUC | F1 @0.50 | F1 @threshold | Δ F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | Legacy: legacy features, random 80/20, threshold tuned on test | 0.6919 |  | 0.9852 | 0.6210 | 0.6497 |  |
| S2 | + new features (constants fit on train, calendar features, no raw month/day) | 0.6951 | +0.0032 | 0.9854 | 0.6235 | 0.6514 | +0.0017 |
| S3 | + score only the Jul–Dec rows of that random test set (month mix) | 0.7062 | +0.0111 | 0.9856 | 0.6289 | 0.6605 | +0.0091 |
| S4 | + train on the past only, walk-forward Jul–Dec, threshold still tuned on test | 0.6732 | -0.0330 | 0.9829 | 0.5924 | 0.6322 | -0.0283 |
| S5 | + walk-forward threshold from prior months only (= current protocol) | 0.6732 | +0.0000 | 0.9829 | 0.5924 | 0.6302 | -0.0020 |
| S4o | (side check) legacy features + walk-forward, threshold tuned on test | 0.6731 |  | 0.9830 | 0.5984 | 0.6324 |  |

Total S1 → S5: PR-AUC -0.0187, F1 -0.0195. S1 seed spread (PR-AUC): 0.6954, 0.7018, 0.6784.

## xgboost

| Step | Condition | PR-AUC | Δ PR-AUC | ROC-AUC | F1 @0.50 | F1 @threshold | Δ F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | Legacy: legacy features, random 80/20, threshold tuned on test | 0.6935 |  | 0.9854 | 0.6238 | 0.6558 |  |
| S2 | + new features (constants fit on train, calendar features, no raw month/day) | 0.6925 | -0.0010 | 0.9854 | 0.6231 | 0.6551 | -0.0007 |
| S3 | + score only the Jul–Dec rows of that random test set (month mix) | 0.7036 | +0.0111 | 0.9856 | 0.6278 | 0.6620 | +0.0068 |
| S4 | + train on the past only, walk-forward Jul–Dec, threshold still tuned on test | 0.6871 | -0.0165 | 0.9848 | 0.6069 | 0.6498 | -0.0122 |
| S5 | + walk-forward threshold from prior months only (= current protocol) | 0.6871 | +0.0000 | 0.9848 | 0.6069 | 0.6480 | -0.0018 |
| S4o | (side check) legacy features + walk-forward, threshold tuned on test | 0.6855 |  | 0.9847 | 0.6090 | 0.6471 |  |

Total S1 → S5: PR-AUC -0.0064, F1 -0.0078. S1 seed spread (PR-AUC): 0.6948, 0.7036, 0.6820.

## lightgbm

| Step | Condition | PR-AUC | Δ PR-AUC | ROC-AUC | F1 @0.50 | F1 @threshold | Δ F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | Legacy: legacy features, random 80/20, threshold tuned on test | 0.6939 |  | 0.9854 | 0.6248 | 0.6548 |  |
| S2 | + new features (constants fit on train, calendar features, no raw month/day) | 0.6924 | -0.0015 | 0.9854 | 0.6243 | 0.6540 | -0.0008 |
| S3 | + score only the Jul–Dec rows of that random test set (month mix) | 0.7041 | +0.0117 | 0.9857 | 0.6274 | 0.6633 | +0.0093 |
| S4 | + train on the past only, walk-forward Jul–Dec, threshold still tuned on test | 0.6874 | -0.0167 | 0.9848 | 0.6077 | 0.6495 | -0.0138 |
| S5 | + walk-forward threshold from prior months only (= current protocol) | 0.6874 | +0.0000 | 0.9848 | 0.6077 | 0.6478 | -0.0017 |
| S4o | (side check) legacy features + walk-forward, threshold tuned on test | 0.6875 |  | 0.9848 | 0.6091 | 0.6494 |  |

Total S1 → S5: PR-AUC -0.0065, F1 -0.0070. S1 seed spread (PR-AUC): 0.6936, 0.7040, 0.6842.

