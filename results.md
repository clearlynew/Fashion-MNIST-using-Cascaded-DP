## Results (Hima)

## ML1
| DP Enabled | Optimizer | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
| ---------- | --------- | ---------------- | ------------ | ------------ | ------------- | ------ | ----------- | --------- | ------ | -------- | -------- | ----------------- | ------------------- |
| No         | Adam      | 0.0              | 1.0          | 32           | Default       | 8      | N/A         | N/A       | 0.3788 | 0.8644   | 0.8621   | 144.55            | 2.41                |
| No         | SGD       | 0.0              | 1.0          | 32           | Default       | 8      | N/A         | N/A       | 0.3862 | 0.8578   | 0.8556   | 146.34            | 2.44                |
| Yes        | Adam      | 0.5              | 1.0          | 32           | Default       | 8      | 5.6255      | 3.33e-05  | 0.9032 | 0.7743   | 0.7632   | 264.76            | 4.41                |
| Yes        | Adam      | 1.0              | 1.0          | 32           | Default       | 8      | 0.6756      | 3.33e-05  | 0.8742 | 0.7325   | 0.7160   | 265.71            | 4.43                |
| Yes        | Adam      | 3.0              | 1.0          | 32           | Default       | 8      | 0.1153      | 3.33e-05  | 0.9129 | 0.6283   | 0.5782   | 257.39            | 4.29                |
| Yes        | SGD       | 0.5              | 1.0          | 32           | Default       | 8      | 5.6255      | 3.33e-05  | 0.8991 | 0.6302   | 0.5904   | 260.14            | 4.34                |
| Yes        | SGD       | 1.0              | 1.0          | 32           | Default       | 8      | 0.6756      | 3.33e-05  | 0.9421 | 0.5964   | 0.5291   | 254.11            | 4.24                |
| Yes        | SGD       | 3.0              | 1.0          | 32           | Default       | 8      | 0.1153      | 3.33e-05  | 0.9553 | 0.6023   | 0.5566   | 260.39            | 4.34                |

## ML2
| DP Enabled | Optimizer | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
| ---------- | --------- | ---------------- | ------------ | ------------ | ------------- | ------ | ----------- | --------- | ------ | -------- | -------- | ----------------- | ------------------- |
| No         | Adam      | 0.0              | 1.0          | 32           | Default       | 8      | N/A         | N/A       | 0.3788 | 0.8644   | 0.8621   | 141.14            | 2.35                |
| No         | SGD       | 0.0              | 1.0          | 32           | Default       | 8      | N/A         | N/A       | 0.3862 | 0.8578   | 0.8556   | 142.92            | 2.38                |
| Yes        | Adam      | 0.5              | 1.0          | 32           | Default       | 8      | 5.6255      | 3.33e-05  | 0.9032 | 0.7743   | 0.7632   | 261.99            | 4.37                |
| Yes        | Adam      | 1.0              | 1.0          | 32           | Default       | 8      | 0.6756      | 3.33e-05  | 0.8742 | 0.7325   | 0.7160   | 264.60            | 4.41                |
| Yes        | Adam      | 3.0              | 1.0          | 32           | Default       | 8      | 0.1153      | 3.33e-05  | 0.9129 | 0.6283   | 0.5782   | 254.70            | 4.25                |
| Yes        | SGD       | 0.5              | 1.0          | 32           | Default       | 8      | 5.6255      | 3.33e-05  | 0.8991 | 0.6302   | 0.5904   | 263.83            | 4.40                |
| Yes        | SGD       | 1.0              | 1.0          | 32           | Default       | 8      | 0.6756      | 3.33e-05  | 0.9421 | 0.5964   | 0.5291   | 251.19            | 4.19                |
| Yes        | SGD       | 3.0              | 1.0          | 32           | Default       | 8      | 0.1153      | 3.33e-05  | 0.9553 | 0.6023   | 0.5566   | 258.13            | 4.30                |

## AutoTuned Model
| Node | DP Enabled | Optimizer         | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
| ---- | ---------- | ----------------- | ---------------- | ------------ | ------------ | ------------- | ------ | ----------- | --------- | ------ | -------- | -------- | ----------------- | ------------------- |
| ML1  | Yes        | Adam (Auto-Tuned) | 0.4305           | 1.5081       | 32           | Default       | 8      | 9.9957      | 3.33e-05  | 0.8356 | 0.7759   | 0.7656   | 260.70            | 4.34                |
| ML2  | Yes        | Adam (Auto-Tuned) | 0.4305           | 1.5081       | 32           | Default       | 8      | 9.9957      | 3.33e-05  | 0.8356 | 0.7759   | 0.7656   | 259.96            | 4.33                |
| ML1  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6218           | 1.5081       | 32           | Default       | 20     | 3.1496      | 3.33e-05  | 0.7984 | 0.8118   | 0.8070   | 618.01            | 10.30               |
| ML2  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6218           | 1.5081       | 32           | Default       | 20     | 3.1496      | 3.33e-05  | 0.7984 | 0.8118   | 0.8070   | 619.66            | 10.33               |

## Adaptive Tuning Summary
| Node | Initial Noise | Final Noise | Initial Clip | Final Clip | Adjustments Made | Final ε |
| ---- | ------------- | ----------- | ------------ | ---------- | ---------------- | ------- |
| ML1  | 0.4640        | 0.6218      | 1.5081       | 1.5081     | 6                | 3.1496  |
| ML2  | 0.4640        | 0.6218      | 1.5081       | 1.5081     | 6                | 3.1496  |

## ML1 Adjustment Log
| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 15    | 0.8008              | 0.4640 → 0.4872 |
| 16    | 0.8047              | 0.4872 → 0.5116 |
| 17    | 0.8075              | 0.5116 → 0.5371 |
| 18    | 0.8054              | 0.5371 → 0.5640 |
| 19    | 0.8104              | 0.5640 → 0.5922 |
| 20    | 0.8124              | 0.5922 → 0.6218 |

## ML2 Adjustment Log
| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 15    | 0.8009              | 0.4640 → 0.4872 |
| 16    | 0.8024              | 0.4872 → 0.5116 |
| 17    | 0.8079              | 0.5116 → 0.5371 |
| 18    | 0.8073              | 0.5371 → 0.5640 |
| 19    | 0.8095              | 0.5640 → 0.5922 |
| 20    | 0.8121              | 0.5922 → 0.6218 |

## Cascaded DP results
| Experiment                      | ML Node      | DP  | Cascaded DP | Noise | Epochs | Training Time (s) | Test Loss | Test Accuracy (%) | Macro F1 (%) | ε (Epsilon) | δ (Delta) | DP Drop Epoch | Gradient Stability Threshold | Accuracy Plateau Threshold |
| ------------------------------- | ------------ | --- | ----------- | ----- | ------ | ----------------: | --------: | ----------------: | -----------: | ----------: | --------- | ------------: | ---------------------------: | -------------------------: |
| Baseline                        | ML1 (Node 0) | No  | No          | 0.0   | 50     |            770.02 |    0.3176 |             89.29 |        89.27 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Baseline                        | ML2 (Node 1) | No  | No          | 0.0   | 50     |            767.15 |    0.3176 |             89.29 |        89.27 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Cascaded DP (Loose Thresholds)  | ML1 (Node 0) | Yes | Yes         | 0.5   | 50     |            871.63 |    0.3225 |             88.70 |        88.59 |      5.6255 | 3.33×10⁻⁵ |             8 |                         0.04 |                      0.001 |
| Cascaded DP (Loose Thresholds)  | ML2 (Node 1) | Yes | Yes         | 0.5   | 50     |            873.87 |    0.3225 |             88.70 |        88.59 |      5.4473 | 3.33×10⁻⁵ |             7 |                         0.04 |                      0.001 |
| Cascaded DP (Strict Thresholds) | ML1 (Node 0) | Yes | Yes         | 0.5   | 50     |            948.00 |    0.3275 |             88.34 |        88.20 |      6.5167 | 3.33×10⁻⁵ |            13 |                         0.01 |                     0.0005 |
| Cascaded DP (Strict Thresholds) | ML2 (Node 1) | Yes | Yes         | 0.5   | 50     |            943.91 |    0.3275 |             88.34 |        88.20 |      6.5167 | 3.33×10⁻⁵ |            13 |                         0.01 |                     0.0005 |
| Standard DP                     | ML1 (Node 0) | Yes | No          | 0.5   | 50     |           1500.08 |    0.8773 |             82.93 |        82.64 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Standard DP                     | ML2 (Node 1) | Yes | No          | 0.5   | 50     |           1499.37 |    0.8773 |             82.93 |        82.64 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |

| Experiment                   | ML Node      | DP  | Cascaded DP | Noise | Epochs | Training Time (s) | Test Loss | Test Accuracy (%) | Macro F1 (%) | ε (Epsilon) | δ (Delta) | DP Drop Epoch | SNR Stability Threshold | Accuracy Plateau Threshold |
| ---------------------------- | ------------ | --- | ----------- | ----- | ------ | ----------------- | --------- | ----------------- | ------------ | ----------- | --------- | ------------- | ----------------------- | -------------------------- |
| Cascaded DP (SNR Thresholds) | ML1 (Node 0) | Yes | Yes         | 0.5   | 50     | 949.65            | 0.3252    | 88.57             | 88.50        | 6.6948      | 3.33×10⁻⁵ | 14            | 0.02                    | 0.005                      |
| Cascaded DP (SNR Thresholds) | ML2 (Node 1) | Yes | Yes         | 0.5   | 50     | 950.46            | 0.3252    | 88.57             | 88.50        | 6.6948      | 3.33×10⁻⁵ | 14            | 0.02                    | 0.005                      |


<img width="2385" height="875" alt="comparison_plot_extended" src="https://github.com/user-attachments/assets/ff57e973-7ae7-4211-ba03-8a361ac61e7d" />

| Experiment                       | ML Node      | DP  | Cascaded DP | Noise | Epochs | Training Time (s) | Test Loss | Test Accuracy (%) | Macro F1 (%) | ε (Epsilon) | δ (Delta) | DP Drop Epoch | Gradient Stability Threshold | Accuracy Plateau Threshold |
| -------------------------------- | ------------ | --- | ----------- | ----- | ------ | ----------------: | --------: | ----------------: | -----------: | ----------: | --------- | ------------: | ---------------------------: | -------------------------: |
| Label Skew Baseline              | ML1 (Node 0) | No  | No          | 0.0   | 50     |            753.27 |    0.3223 |             88.55 |        88.51 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Label Skew Baseline              | ML2 (Node 1) | No  | No          | 0.0   | 50     |            754.04 |    0.3223 |             88.55 |        88.51 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Label Skew + Standard DP         | ML1 (Node 0) | Yes | No          | 0.5   | 50     |           1383.93 |    0.8686 |             82.64 |        82.60 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Label Skew + Standard DP         | ML2 (Node 1) | Yes | No          | 0.5   | 50     |           1380.00 |    0.8686 |             82.64 |        82.60 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Label Skew + Cascaded DP         | ML1 (Node 0) | Yes | Yes         | 0.5   | 50     |            961.81 |    0.3266 |             88.15 |        88.14 |      7.0512 | 3.33×10⁻⁵ |            16 |                         0.01 |                     0.0005 |
| Label Skew + Cascaded DP         | ML2 (Node 1) | Yes | Yes         | 0.5   | 50     |            958.32 |    0.3266 |             88.15 |        88.14 |      7.0512 | 3.33×10⁻⁵ |            16 |                         0.01 |                     0.0005 |
| Extreme Label Skew Baseline      | ML1 (Node 0) | No  | No          | 0.0   | 50     |            747.11 |    0.5765 |             78.10 |        79.18 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Extreme Label Skew Baseline      | ML2 (Node 1) | No  | No          | 0.0   | 50     |            743.52 |    0.5765 |             78.10 |        79.18 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Extreme Label Skew + Standard DP | ML1 (Node 0) | Yes | No          | 0.5   | 50     |           1407.24 |    1.0890 |             76.42 |        77.07 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Extreme Label Skew + Standard DP | ML2 (Node 1) | Yes | No          | 0.5   | 50     |           1406.02 |    1.0890 |             76.42 |        77.07 |     10.2567 | 3.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Extreme Label Skew + Cascaded DP | ML1 (Node 0) | Yes | Yes         | 0.5   | 50     |           1019.55 |    0.6876 |             74.11 |        75.28 |      7.4632 | 3.33×10⁻⁵ |            19 |                         0.01 |                     0.0005 |
| Extreme Label Skew + Cascaded DP | ML2 (Node 1) | Yes | Yes         | 0.5   | 50     |           1019.32 |    0.6876 |             74.11 |        75.28 |      7.5533 | 3.33×10⁻⁵ |            20 |                         0.01 |                     0.0005 |

| Node         | DP Drop Epoch | Val Acc (t-4) | Val Acc (t-3) | Val Acc (t-2) | Val Acc (t-1) | Val Acc (t) | Rolling Mean |
| ------------ | ------------: | ------------: | ------------: | ------------: | ------------: | ----------: | -----------: |
| ML1 (Node 0) |            16 |        76.89% |        77.25% |        77.39% |        77.65% |      78.08% |       6.4467 |
| ML2 (Node 1) |            16 |        69.34% |        69.95% |        72.32% |        73.34% |      73.88% |      10.1055 |

| Node         | DP Drop Epoch | Val Acc (t-4) | Val Acc (t-3) | Val Acc (t-2) | Val Acc (t-1) | Val Acc (t) | Rolling Mean |
| ------------ | ------------: | ------------: | ------------: | ------------: | ------------: | ----------: | -----------: |
| ML1 (Node 0) |            19 |        70.62% |        71.76% |        70.85% |        70.05% |      72.72% |      10.1318 |
| ML2 (Node 1) |            20 |        54.55% |        54.22% |        54.75% |        55.76% |      56.26% |      23.8160 |


| Experiment            | ML Node      | Samples | Weight | DP  | Cascaded DP | Noise | Epochs | Training Time (s) | Test Loss | Test Accuracy (%) | Macro F1 (%) | ε (Epsilon) | δ (Delta) | DP Drop Epoch | Gradient Stability Threshold | Accuracy Plateau Threshold |
| --------------------- | ------------ | ------: | -----: | --- | ----------- | ----- | ------ | ----------------: | --------: | ----------------: | -----------: | ----------: | --------- | ------------: | ---------------------------: | -------------------------: |
| Unequal Baseline      | ML1 (Node 0) |   48000 |     80 | No  | No          | 0.0   | 50     |           1189.14 |    0.3376 |             88.25 |        88.18 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Unequal Baseline      | ML2 (Node 1) |   12000 |     20 | No  | No          | 0.0   | 50     |           1190.05 |    0.3376 |             88.25 |        88.18 |         N/A | N/A       |           N/A |                          N/A |                        N/A |
| Unequal + Standard DP | ML1 (Node 0) |   48000 |     80 | Yes | No          | 0.5   | 50     |           1971.44 |    0.8709 |             81.34 |        81.15 |      8.7053 | 2.08×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Unequal + Standard DP | ML2 (Node 1) |   12000 |     20 | Yes | No          | 0.5   | 50     |           1970.99 |    0.8709 |             81.34 |        81.15 |     15.1514 | 8.33×10⁻⁵ |           N/A |                          N/A |                        N/A |
| Unequal + Cascaded DP | ML1 (Node 0) |   48000 |     80 | Yes | Yes         | 0.5   | 50     |           1497.38 |    0.4230 |             85.10 |        85.01 |      5.6011 | 2.08×10⁻⁵ |            13 |                         0.01 |                     0.0005 |
| Unequal + Cascaded DP | ML2 (Node 1) |   12000 |     20 | Yes | Yes         | 0.5   | 50     |           1495.67 |    0.4232 |             85.10 |        85.01 |     15.1514 | 8.33×10⁻⁵ |          N/A* |                         0.01 |                     0.0005 |

| Node         | Samples | Weight | DP Drop Epoch | Val Acc (t-4) | Val Acc (t-3) | Val Acc (t-2) | Val Acc (t-1) | Val Acc (t) | Rolling Mean |
| ------------ | ------: | -----: | ------------: | ------------: | ------------: | ------------: | ------------: | ----------: | -----------: |
| ML1 (Node 0) |   48000 |     80 |            13 |        79.83% |        80.19% |        80.51% |        80.92% |      80.80% |       5.8699 |
| ML2 (Node 1) |   12000 |     20 |          N/A* |           N/A |           N/A |           N/A |           N/A |         N/A |          N/A |



---

## Results (Anushka)

## ML1 Results

| Experiment | Optimizer | DP Enabled | Noise Multiplier | Epsilon | Delta | Loss | Accuracy | F1-Score | Training Time (s) | Training Time (min) |
|------------|-----------|------------|------------------|---------|--------|--------|----------|----------|-------------------|---------------------|
| Baseline | Adam | No | 0.0 | — | — | 0.3761 | 0.8633 | 0.8626 | 150.61 | 2.51 |
| DP-0.5 | Adam | Yes | 0.5 | 5.6255 | 3.3333e-05 | 0.8943 | 0.7656 | 0.7499 | 299.94 | 5.00 |
| DP-1.0 | Adam | Yes | 1.0 | 0.6756 | 3.3333e-05 | 0.9069 | 0.7050 | 0.6855 | 290.18 | 4.84 |
| DP-3.0 | Adam | Yes | 3.0 | 0.1153 | 3.3333e-05 | 0.9477 | 0.6132 | 0.5708 | 292.58 | 4.88 |
| Baseline | SGD | No | 0.0 | — | — | 0.3874 | 0.8568 | 0.8564 | 155.94 | 2.60 |
| DP-0.5 | SGD | Yes | 0.5 | 5.6255 | 3.3333e-05 | 0.9546 | 0.5893 | 0.5120 | 310.76 | 5.18 |
| DP-1.0 | SGD | Yes | 1.0 | 0.6756 | 3.3333e-05 | 0.9232 | 0.6321 | 0.5796 | 295.10 | 4.92 |
| DP-3.0 | SGD | Yes | 3.0 | 0.1153 | 3.3333e-05 | 0.9453 | 0.6082 | 0.5599 | 293.22 | 4.89 |


## ML2 Results

| Experiment | Optimizer | DP Enabled | Noise Multiplier | Epsilon | Delta | Loss | Accuracy | F1-Score | Training Time (s) | Training Time (min) |
|------------|-----------|------------|------------------|---------|--------|--------|----------|----------|-------------------|---------------------|
| Baseline | Adam | No | 0.0 | — | — | 0.3761 | 0.8633 | 0.8626 | 148.47 | 2.47 |
| DP-0.5 | Adam | Yes | 0.5 | 5.6255 | 3.3333e-05 | 0.8943 | 0.7656 | 0.7499 | 291.32 | 4.86 |
| DP-1.0 | Adam | Yes | 1.0 | 0.6756 | 3.3333e-05 | 0.9069 | 0.7050 | 0.6855 | 299.80 | 5.00 |
| DP-3.0 | Adam | Yes | 3.0 | 0.1153 | 3.3333e-05 | 0.9477 | 0.6132 | 0.5708 | 291.58 | 4.86 |
| Baseline | SGD | No | 0.0 | — | — | 0.3874 | 0.8568 | 0.8564 | 147.96 | 2.47 |
| DP-0.5 | SGD | Yes | 0.5 | 5.6255 | 3.3333e-05 | 0.9546 | 0.5893 | 0.5120 | 290.23 | 4.84 |
| DP-1.0 | SGD | Yes | 1.0 | 0.6756 | 3.3333e-05 | 0.9232 | 0.6321 | 0.5796 | 289.93 | 4.83 |
| DP-3.0 | SGD | Yes | 3.0 | 0.1153 | 3.3333e-05 | 0.9453 | 0.6082 | 0.5599 | 297.15 | 4.95 |

## Auto-Tuned Adam Results

| Node | DP Enabled | Optimizer                  | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
| ---- | ---------- | -------------------------- | ---------------- | ------------ | ------------ | ------------- | ------ | ----------- | --------- | ------ | -------- | -------- | ----------------- | ------------------- |
| ML1  | Yes        | Adam (Auto-Tuned)          | 0.4305           | 1.5081       | 32           | Default       | 8      | 9.9957      | 3.33e-05  | 0.8816 | 0.7666   | 0.7497   | 359.27            | 5.99                |
| ML2  | Yes        | Adam (Auto-Tuned)          | 0.4305           | 1.5081       | 32           | Default       | 8      | 9.9957      | 3.33e-05  | 0.8816 | 0.7666   | 0.7497   | 361.63            | 6.03                |
| ML1  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6218           | 1.5081       | 32           | Default       | 20     | 3.1496      | 3.33e-05  | 0.8496 | 0.8102   | 0.8084   | 905.01            | 15.08               |
| ML2  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6218           | 1.5081       | 32           | Default       | 20     | 3.1496      | 3.33e-05  | 0.8496 | 0.8102   | 0.8084   | 827.47            | 13.79               |

## Adaptive Tuning Summary (8 Epochs)

| Node | Initial Noise | Final Noise | Initial Clip | Final Clip | Adjustments Made | Final ε |
|------|---------------|-------------|--------------|------------|------------------|---------|
| ML1 | 0.4305 | 0.4305 | 1.5081 | 1.5081 | 0 | 9.9957 |
| ML2 | 0.4305 | 0.4305 | 1.5081 | 1.5081 | 0 | 9.9957 |

## ML1 Adjustment Log (8 Epochs)

No adaptive adjustments were triggered during training.

## ML2 Adjustment Log (8 Epochs)

No adaptive adjustments were triggered during training.

## Adaptive Tuning Summary (20 epochs)

| Node | Initial Noise | Final Noise | Initial Clip | Final Clip | Adjustments Made | Final ε |
| ---- | ------------- | ----------- | ------------ | ---------- | ---------------- | ------- |
| ML1  | 0.4640        | 0.6218      | 1.5081       | 1.5081     | 6                | 3.1496  |
| ML2  | 0.4640        | 0.6218      | 1.5081       | 1.5081     | 6                | 3.1496  |

## ML1 Adjustment Log (20 epochs)

| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 15    | 0.8053              | 0.4640 → 0.4872 |
| 16    | 0.8064              | 0.4872 → 0.5116 |
| 17    | 0.8083              | 0.5116 → 0.5371 |
| 18    | 0.8045              | 0.5371 → 0.5640 |
| 19    | 0.8073              | 0.5640 → 0.5922 |
| 20    | 0.8117              | 0.5922 → 0.6218 |

## ML2 Adjustment Log (20 epochs)

| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 14    | 0.8014              | 0.4640 → 0.4872 |
| 16    | 0.8062              | 0.4872 → 0.5116 |
| 17    | 0.8072              | 0.5116 → 0.5371 |
| 18    | 0.8075              | 0.5371 → 0.5640 |
| 19    | 0.8088              | 0.5640 → 0.5922 |
| 20    | 0.8087              | 0.5922 → 0.6218 |

## Cascaded DP

| Method                     | Node | DP Enabled | Cascaded DP | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | DP Drop Epoch | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
|----------------------------|------|------------|-------------|------------------|--------------|--------------|---------------|--------|-------------|-----------|---------------|--------|----------|----------|-------------------|---------------------|
| Cascaded DP       | 0    | Yes        | Yes         | 0.5              | 1.0          | 32           | 0.001         | 50     | 5.8038      | 3.33e-05  | 9             | 0.3170 | 0.8880   | 0.8878   | 948.06            | 15.80               |
| Cascaded DP       | 1    | Yes        | Yes         | 0.5              | 1.0          | 32           | 0.001         | 50     | 5.6255      | 3.33e-05  | 8             | 0.3170 | 0.8880   | 0.8878   | 935.62            | 15.59               |
| SNR-Gated Cascaded DP      | 0    | Yes        | Yes         | 0.5              | 1.0          | 32           | 0.001         | 50     | 10.2567     | 3.33e-05  | N/A           | 0.8688 | 0.8281   | 0.8250   | 1687.24           | 28.12               |
| SNR-Gated Cascaded DP      | 1    | Yes        | Yes         | 0.5              | 1.0          | 32           | 0.001         | 50     | 10.2567     | 3.33e-05  | N/A           | 0.8688 | 0.8281   | 0.8250   | 1692.68           | 28.21               |

---

## Results (Sidharth)

## ML1

| Experiment    | DP  | Noise | Epsilon (ε) | Delta (δ) | Optimizer | Accuracy | F1 Score | Loss   | Training Time (s) | Training Time (min) |
| ------------- | --- | ----- | ----------- | --------- | --------- | -------- | -------- | ------ | ----------------- | ------------------- |
| Baseline Adam | No  | 0.0   | —           | —         | Adam      | 0.8308   | 0.8308   | 0.4641 | 162.35            | 2.71                |
| DP Adam 0.5   | Yes | 0.5   | 5.6255      | 3.33e-05  | Adam      | 0.7746   | 0.7657   | 0.9266 | 427.08            | 7.12                |
| DP Adam 0.7   | Yes | 0.7   | 1.767       | 3.33e-05  | Adam      | 0.7502   | 0.7402   | 0.8824 | 427.87            | 7.13                |
| DP Adam 1.0   | Yes | 1.0   | 0.6756      | 3.33e-05  | Adam      | 0.7159   | 0.6929   | 0.9095 | 435.45            | 7.26                |
| DP Adam 3.0   | Yes | 3.0   | 0.1153      | 3.33e-05  | Adam      | 0.6219   | 0.5857   | 0.9152 | 407.00            | 6.78                |

## ML2

| Experiment    | DP  | Noise | Epsilon (ε) | Delta (δ) | Optimizer | Accuracy | F1 Score | Loss   | Training Time (s) | Training Time (min) |
| ------------- | --- | ----- | ----------- | --------- | --------- | -------- | -------- | ------ | ----------------- | ------------------- |
| Baseline Adam | No  | 0.0   | —           | —         | Adam      | 0.8264   | 0.8261   | 0.4848 | 147.80            | 2.46                |
| DP Adam 0.5   | Yes | 0.5   | 5.6255      | 3.33e-05  | Adam      | 0.7746   | 0.7657   | 0.9266 | 423.53            | 7.06                |
| DP Adam 0.7   | Yes | 0.7   | 1.767       | 3.33e-05  | Adam      | 0.7502   | 0.7402   | 0.8824 | 421.98            | 7.03                |
| DP Adam 1.0   | Yes | 1.0   | 0.6756      | 3.33e-05  | Adam      | 0.7159   | 0.6929   | 0.9095 | 432.14            | 7.20                |
| DP Adam 3.0   | Yes | 3.0   | 0.1153      | 3.33e-05  | Adam      | 0.6219   | 0.5857   | 0.9152 | 397.31            | 6.62                |

## Auto-Tuned Adam Results

| Node | DP Enabled | Optimizer                  | Noise Multiplier | L2 Norm Clip | Microbatches | Learning Rate | Epochs | Epsilon (ε) | Delta (δ) | Loss   | Accuracy | F1 Score | Training Time (s) | Training Time (min) |
| ---- | ---------- | -------------------------- | ---------------- | ------------ | ------------ | ------------- | ------ | ----------- | --------- | ------ | -------- | -------- | ----------------- | ------------------- |
| ML1  | Yes        | Adam (Auto-Tuned)          | 0.4640           | 1.5081       | 32           | Default       | 8      | 9.9981      | 3.33e-05  | —      | —        | —        | —                 | —                   |
| ML2  | Yes        | Adam (Auto-Tuned)          | 0.4640           | 1.5081       | 32           | Default       | 8      | 9.9981      | 3.33e-05  | —      | —        | —        | —                 | —                   |
| ML1  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6855           | 1.5081       | 32           | Default       | 20     | 2.4006      | 3.33e-05  | 0.8040 | 0.8125   | 0.8083   | 1265.41           | 21.09               |
| ML2  | Yes        | Adam (Auto-Tuned Adaptive) | 0.6529           | 1.5081       | 32           | Default       | 20     | 2.6574      | 3.33e-05  | 0.8040 | 0.8125   | 0.8083   | 1260.84           | 21.01               |

## Adaptive Tuning Summary (20 Epochs)

| Node | Initial Noise | Final Noise | Initial Clip | Final Clip | Adjustments Made | Final ε |
| ---- | ------------- | ----------- | ------------ | ---------- | ---------------- | ------- |
| ML1  | 0.4640        | 0.6855      | 1.5081       | 1.5081     | 8                | 2.4006  |
| ML2  | 0.4640        | 0.6529      | 1.5081       | 1.5081     | 7                | 2.6574  |

## ML1 Adjustment Log (20 Epochs)

| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 13    | 0.8018              | 0.4640 → 0.4872 |
| 14    | 0.8032              | 0.4872 → 0.5116 |
| 15    | 0.8049              | 0.5116 → 0.5371 |
| 16    | 0.8103              | 0.5371 → 0.5640 |
| 17    | 0.8099              | 0.5640 → 0.5922 |
| 18    | 0.8106              | 0.5922 → 0.6218 |
| 19    | 0.8083              | 0.6218 → 0.6529 |
| 20    | 0.8122              | 0.6529 → 0.6855 |

## ML2 Adjustment Log (20 Epochs)

| Epoch | Validation Accuracy | Noise Change    |
| ----- | ------------------- | --------------- |
| 14    | 0.8046              | 0.4640 → 0.4872 |
| 15    | 0.8063              | 0.4872 → 0.5116 |
| 16    | 0.8064              | 0.5116 → 0.5371 |
| 17    | 0.8098              | 0.5371 → 0.5640 |
| 18    | 0.8120              | 0.5640 → 0.5922 |
| 19    | 0.8138              | 0.5922 → 0.6218 |
| 20    | 0.8115              | 0.6218 → 0.6529 |



## Cascaded DP Comparison


![Cascaded DP Comparison](./results_sidharth(cascaded)/comparison_plot_cascaded.png)

| Experiment | Accuracy (%) | Training Time (s) | DP Drop Epoch | ε (Epsilon) |
| --- | ---: | ---: | ---: | ---: |
| Baseline (No DP) | 89.29 | 769 | N/A | N/A |
| Standard DP (noise=0.5) | 82.93 | 1500 | N/A | 5.6255 |
| Cascaded DP (Loose thresholds) | 88.70 | 873 | 7–8 | 5.4473–5.6255 |
| Cascaded DP (Strict thresholds) | 88.34 | 946 | 13 | 6.5167 |
| SNR Cascaded DP | 88.76 | 1546 | 15 | 6.8731 |

---

## Results (Goutham)

The following parameters are fixed:
- Microbatches : 32
- L2 Norm Clip : 1.0
- Number of epochs : 8
- Learning Rate: 0.01 for SGD and 0.001 for Adam

## ML1

| Experiment | Optimizer | DP Enabled | Noise Multiplier | Epsilon | Delta      | Loss   | Accuracy | F1-Score | Training Time (s) | Training Time (min) |
| ---------- | --------- | ---------- | ---------------- | ------- | ---------- | ------ | -------- | -------- | ----------------- | ------------------- |
| Baseline   | Adam      | No         | 0.0              | —       | —          | 0.3682 | 0.8659   | 0.8623   | 148.75            | 2.48                |
| DP-0.5     | Adam      | Yes        | 0.5              | 5.6255  | 3.3333e-05 | 0.8919 | 0.7719   | 0.7572   | 411.16            | 6.85                |
| DP-1.0     | Adam      | Yes        | 1.0              | 0.6756  | 3.3333e-05 | 0.8633 | 0.7353   | 0.7189   | 596.05            | 9.93                |
| DP-3.0     | Adam      | Yes        | 3.0              | 0.1153  | 3.3333e-05 | 0.8990 | 0.6321   | 0.5971   | 399.92            | 6.67                |
| Baseline   | SGD       | No         | 0.0              | —       | —          | 0.3894 | 0.8565   | 0.8550   | 156.61            | 2.61                |
| DP-0.5     | SGD       | Yes        | 0.5              | 5.6255  | 3.3333e-05 | 0.9054 | 0.6294   | 0.5882   | 442.16            | 7.37                |
| DP-1.0     | SGD       | Yes        | 1.0              | 0.6756  | 3.3333e-05 | 0.9005 | 0.6089   | 0.5530   | 512.27            | 8.54                |
| DP-3.0     | SGD       | Yes        | 3.0              | 0.1153  | 3.3333e-05 | 1.0116 | 0.5847   | 0.5118   | 399.39            | 6.66                |

## ML2

| Experiment | Optimizer | DP Enabled | Noise Multiplier | Epsilon | Delta      | Loss   | Accuracy | F1-Score | Training Time (s) | Training Time (min) |
| ---------- | --------- | ---------- | ---------------- | ------- | ---------- | ------ | -------- | -------- | ----------------- | ------------------- |
| Baseline   | Adam      | No         | 0.0              | —       | —          | 0.3682 | 0.8659   | 0.8623   | 146.30            | 2.44                |
| DP-0.5     | Adam      | Yes        | 0.5              | 5.6255  | 3.3333e-05 | 0.8919 | 0.7719   | 0.7572   | 395.14            | 6.59                |
| DP-1.0     | Adam      | Yes        | 1.0              | 0.6756  | 3.3333e-05 | 0.8633 | 0.7353   | 0.7189   | 550.94            | 9.18                |
| DP-3.0     | Adam      | Yes        | 3.0              | 0.1153  | 3.3333e-05 | 0.8990 | 0.6321   | 0.5971   | 407.92            | 6.80                |
| Baseline   | SGD       | No         | 0.0              | —       | —          | 0.3894 | 0.8565   | 0.8550   | 145.68            | 2.43                |
| DP-0.5     | SGD       | Yes        | 0.5              | 5.6255  | 3.3333e-05 | 0.9054 | 0.6294   | 0.5882   | 391.37            | 6.52                |
| DP-1.0     | SGD       | Yes        | 1.0              | 0.6756  | 3.3333e-05 | 0.9005 | 0.6089   | 0.5530   | 507.77            | 8.46                |
| DP-3.0     | SGD       | Yes        | 3.0              | 0.1153  | 3.3333e-05 | 1.0116 | 0.5847   | 0.5118   | 389.43            | 6.49                |

## Cascaded DP Experiments and Results

The following parameters are fixed:
- Microbatches : 32
- L2 Norm Clip : 1.0
- Number of epochs : 50
- Learning Rate: 0.001 (Adam)

### Node 0
| Experiment              | DP Type        | Noise Multiplier | DP Enabled | Accuracy | F1 Score | Loss   | Epsilon (ε) | Delta (δ) | DP Drop Epoch | Training Time (s) | Training Time (min) |
| ----------------------- | -------------- | ---------------- | ---------- | -------- | -------- | ------ | ----------- | --------- | ------------- | ----------------- | ------------------- |
| No DP                   | None           | __               | No         | 0.8885   | 0.8883   | 0.3217 | __          | __        | __            | 789.98            | 13.17               |
| Cascaded DP (0.5)       | Cascaded       | 0.5              | Yes        | 0.8839   | 0.8824   | 0.3249 | 5.8038      | 3.33e-05  | 9             | 1189.13           | 19.82               |
| Cascaded DP + SNR (0.5) | Cascaded + SNR | 0.5              | Yes        | 0.8838   | 0.8839   | 0.3323 | 7.6434      | 3.33e-05  | 21            | 1438.27           | 23.97               |
| Cascaded DP + SNR (0.8) | Cascaded + SNR | 0.8              | Yes        | 0.8817   | 0.8813   | 0.3353 | 1.4776      | 3.33e-05  | 22            | 1527.63           | 25.46               |
| Dual-Phase Loss (0.8)   | Dual-Phase     | 0.8              | Yes        | 0.8806   | 0.8795   | 0.3304 | 1.4380      | 3.33e-05  | 20            | 1429.87           | 23.83               |

### Node 1
| Experiment              | DP Type        | Noise Multiplier | DP Enabled | Accuracy | F1 Score | Loss   | Epsilon (ε) | Delta (δ) | DP Drop Epoch | Training Time (s) | Training Time (min) |
| ----------------------- | -------------- | ---------------- | ---------- | -------- | -------- | ------ | ----------- | --------- | ------------- | ----------------- | ------------------- |
| No DP                   | None           | __               | No         | 0.8885   | 0.8883   | 0.3217 | __          | __        | __            | 780.23            | 13.00               |
| Cascaded DP (0.5)       | Cascaded       | 0.5              | Yes        | 0.8839   | 0.8824   | 0.3249 | 5.8038      | 3.33e-05  | 9             | 1163.78           | 19.40               |
| Cascaded DP + SNR (0.5) | Cascaded + SNR | 0.5              | Yes        | 0.8838   | 0.8839   | 0.3323 | 7.7335      | 3.33e-05  | 22            | 1441.91           | 24.03               |
| Cascaded DP + SNR (0.8) | Cascaded + SNR | 0.8              | Yes        | 0.8817   | 0.8813   | 0.3353 | 1.4776      | 3.33e-05  | 22            | 1531.79           | 25.53               |
| Dual-Phase Loss (0.8)   | Dual-Phase     | 0.8              | Yes        | 0.8806   | 0.8795   | 0.3304 | 1.4380      | 3.33e-05  | 20            | 1432.95           | 23.88               |


---
