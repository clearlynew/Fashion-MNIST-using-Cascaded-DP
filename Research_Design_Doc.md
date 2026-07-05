# CascadedDP: Research Design Document

**Project:** Cascaded Differential Privacy for Federated Learning via HPE Swarm Learning
**Status:** Design finalized; experimental phase pending
**Last updated:** 2026-07-05

---

## 1. Scope and Contribution Statement

### 1.1 Claim Boundaries

This work does not claim that CascadedDP provides a formal end-to-end
(ε, δ)-differential-privacy guarantee. Once DP-SGD is dropped in Stage 2 of
the cascade, no accounting exists for subsequent training unless a Stage 2
noise floor is introduced (Section 5, Open Item 1). This limitation is
treated as a first-class constraint on the paper's claims, not a caveat to
be minimized.

### 1.2 Contributions

1. **A decentralized, coordinator-free consensus mechanism** for triggering
   phase transitions in federated DP training. A non-trainable weight
   (`ConvergenceFlagLayer`) is attached to the model so that it is included
   in Swarm Learning's existing cross-host weight-merge operation, removing
   the need for a separate coordination channel or central vote-counting
   service.
2. **A local, self-scaling convergence signal**, combining gradient-norm
   slope and validation-accuracy plateau, each measured relative to an
   empirically estimated burn-in noise floor rather than fixed constants.
3. **An empirical characterization of the privacy-utility tradeoff**,
   evaluated across data heterogeneity (Dirichlet α) and DP configuration
   (No DP / Full DP / Cascaded DP), using LiRA as the membership-inference
   evidence. This is presented as an empirical study, not a formal proof of
   privacy.

This is a systems and empirical contribution. The paper's framing should
reflect this explicitly rather than implying a stronger formal privacy
result than is supported by the method.

---

## 2. Fixed Experimental Configuration

The following parameters are fixed across all experiments and are not
subject to per-experiment retuning.

| Parameter | Value | Justification |
|---|---|---|
| Architecture | CNN: Conv(16, k8, s2) → Pool → Conv(32, k4, s2) → Pool → Flatten → Dense(32) → Dense(10) | Matches the canonical DP-SGD benchmark architecture used in TensorFlow Privacy and Opacus reference implementations for MNIST/Fashion-MNIST |
| Dataset | Fashion-MNIST | Standard benchmark in the DP-FL literature. Known limitation: this dataset is a comparatively weak testbed for membership-inference evaluation; attack sensitivity must be validated on the undefended baseline before results are interpreted (Section 5) |
| Optimizer | Adam (lr = 0.001), with SGD + momentum (lr = 0.01) as a secondary comparison | Standard defaults for this architecture/dataset pairing. DP-SGD is more sensitive to learning rate than standard training; if Full DP accuracy falls substantially below the ~84-86% range typically reported for DP-SGD on Fashion-MNIST, learning rate should be investigated before the method is questioned |
| Batch size | 32 | Standard |
| Training epochs (`maxEpoch`) | 50 | To be verified empirically that the slowest-converging condition (expected: Full DP, α = 0.01) reaches a plateau within this budget prior to use in comparative experiments |
| Burn-in length (`min_dp_epochs`) | 5 | Used both as the minimum epoch count before convergence checks activate and as the window for estimating the burn-in noise floor. Requires empirical verification that the resulting floor estimate is stable (Section 4) |
| `syncFrequency` | 1024 (batches) | Infrastructure/communication parameter, orthogonal to the privacy mechanism; not swept. Requires verification that this value results in at least one synchronization per epoch at the most heterogeneous partition (α = 0.01), where per-node shard sizes are smallest |
| `useAdaptiveSync` | False | Not required; the consensus mechanism operates entirely through the standard weight-merge channel. Enabling this parameter would introduce an additional, undocumented dependency without functional benefit |
| `microbatches` | Equal to batch size | Standard, maximally privacy-precise configuration; may be relaxed if computational cost becomes prohibitive, at the cost of coarser per-step privacy accounting |
| `minPeers` | Equal to node count | Deployment parameter, not an experimental variable |
| Node count (primary results) | 2-3 | Consistent with current validated deployment; higher node counts are out of scope for this work |
| Rolling window size (grad-norm / accuracy buffer) | Fixed once selected (e.g., 5) | The window's effect on trigger sensitivity is absorbed into the self-scaling threshold formulation (Section 3.3), removing the need for a joint sweep with the threshold parameters |
| `noise_multiplier`, `l2_norm_clip` | Not fixed; selected as deliberate pairs (Section 2.1) | These parameters jointly determine ε and must be chosen to produce comparable, reportable privacy levels rather than left at arbitrary defaults |

### 2.1 Selection of Noise Multiplier and Clipping Norm

`noise_multiplier` and `l2_norm_clip` are selected deliberately rather than
left at default values, so that the Full DP and Cascaded DP conditions
correspond to recognized operating points in the literature (commonly
ε ≈ 1 for a strict regime and ε ≈ 8 for a moderate regime).

Procedure:
1. Fix `l2_norm_clip = 1.0`.
2. Using `compute_dp_sgd_privacy` (already available in the training
   script), search over `noise_multiplier` values for the chosen
   `num_train_samples`, `batch_size`, and `maxEpoch` until ε approaches each
   target.
3. Adopt the resulting `noise_multiplier` values as fixed configurations for
   the Full DP condition. If time permits, both ε ≈ 1 and ε ≈ 8 are reported
   as separate DP tiers; otherwise ε ≈ 8 is used as the primary Full DP
   condition.
4. Apply the same `noise_multiplier` to Stage 1 of Cascaded DP, so that
   Full DP and Cascaded DP differ only in duration/mechanism of noise
   application, not in noise magnitude.

---

## 3. Experimental Design

### 3.1 Axis A - Data Heterogeneity

Seven-point Dirichlet α sweep: `{inf (IID), 1.0, 0.5, 0.3, 0.1, 0.05, 0.01}`.

### 3.2 Axis B - Privacy Condition

Three conditions, with ε reported for each:
1. No DP
2. Full DP (DP-SGD for the full training run)
3. Cascaded DP (Stage 1 DP-SGD, consensus-triggered transition, Stage 2)

**Open item:** Stage 2 currently applies no noise, so the reported ε for
Cascaded DP reflects Stage 1 only (Section 5, Open Item 1). This is
acceptable for sensitivity and comparison experiments but must be resolved
before final reporting.

### 3.3 Axis C - Convergence-Trigger Sensitivity

A grid sweep over `k_slope x k_acc`, run at one fixed α (proposed: 0.3) and
one fixed noise configuration, characterizing the sensitivity of trigger
timing, final accuracy, final ε, and LiRA AUC to these two dimensionless
multipliers.

Proposed grid: `k_slope in {0.5, 1.0, 1.5, 2.0}`, `k_acc in {0.5, 1.0, 1.5, 2.0}`
(16 configurations). Results to be reported as a heatmap per metric.

This experiment directly substantiates the choice of trigger threshold
values with empirical evidence, rather than presenting them as unjustified
constants.

### 3.4 Multi-Seed Replication

The current Dirichlet partitioning uses a fixed seed, meaning every run at a
given α draws an identical partition. This is insufficient to distinguish a
genuine heterogeneity effect from an artifact of a single partition
instance, particularly at extreme α values where at least one unresolved
anomaly (α = 0.005 underperformance) requires confirmation.

| Experiment | Seed count | Rationale |
|---|---|---|
| Main table (3 conditions x 7 α = 21 configurations) | Minimum 3 seeds across all configurations; 5 seeds at the extremes (IID, α = 0.01) and at any configuration where Cascaded DP closely matches or exceeds baseline performance | Confidence intervals are required for headline results; uniform 5-seed replication across all configurations is not computationally justified |
| Threshold sensitivity ablation (16 configurations) | 1 seed (2-3 if computational budget allows) | This experiment characterizes mechanism sensitivity rather than reporting a headline result |
| LiRA shadow-model evaluation | Underlying trained models must originate from multiple seeds | LiRA success rate is itself noisy per trained model; single-seed underlying models would compound this with partition-seed variance |

The Dirichlet partition seed should be varied; varying model initialization
seed as well is preferable if computational budget allows. All reported
results should include mean and standard deviation (or 95% confidence
interval), not single-point estimates.

---

## 4. Pre-Experiment Verification Checklist

The following must be verified before committing to the full experimental
matrix.

- [x] CNN trains without shape or broadcast errors (verified via 2-3 epoch, No DP, IID smoke test)
- [x] No-DP baseline accuracy exceeds prior MLP-based results (verified: 88.9% at α = 1.0, consistent with expected performance for this architecture/dataset combination)
- [ ] Slowest-converging condition (expected: Full DP, α = 0.01) reaches a plateau within `maxEpoch = 50`
- [ ] Partitioning (per-node class counts) and ε at fixed configuration are unchanged relative to prior MLP-based runs at identical seed/α (these quantities are architecture-independent; any discrepancy indicates a defect, not an expected consequence of the architecture change)
- [ ] Burn-in noise floor (`burn_in_std_grad`, logged at runtime) is non-degenerate; a near-zero value indicates `min_dp_epochs` is too short for a stable estimate and should be increased before trusting `k_slope`/`k_acc` sweep results
- [ ] `ConvergenceFlagLayer` merge reaches quorum correctly under the current Docker deployment (post weight-enumeration crash fix); this has not yet been independently reverified end-to-end
- [ ] True cross-host merge behavior is confirmed on physically separate hosts, not only across containers on a single host, prior to any claim regarding coordinator-free cross-host consensus

---

## 5. Open Items

| Item | Status | Impact |
|---|---|---|
| 1. Stage 2 noise floor (required for a defined Cascaded DP ε covering the full training run) | Not implemented | Blocks final reported results; does not block sensitivity/comparison experiments |
| 2. LiRA implementation (replacing the TF Privacy built-in MIA as primary evidence) | Not implemented | Blocks the privacy-evidence section of the paper |
| 3. LiRA x heterogeneity cross-indexing | Depends on Item 2 | Anticipated to be the strongest unique empirical result of this work |
| 4. Multi-host validation of `ConvergenceFlagLayer` merge | Pending; single-host multi-container test previously failed on a weight-enumeration assertion, since resolved but not reverified | Blocks the coordinator-free, cross-host systems claim |
| 5. MIA/LiRA sensitivity check on the undefended baseline | Not yet performed | Required to establish that the chosen attack is sufficiently sensitive before any "no leakage detected" result is reported |

---

## 6. Estimated Experimental Volume

- Main results: 21 base configurations (3 conditions x 7 α values), scaled by seed replication per Section 3.4 - approximately 75-90 total runs
- Threshold sensitivity ablation: 16 configurations (single seed)
- LiRA shadow-model training: computationally dominant component; representative α subsampling (e.g., IID, 0.3, 0.01) recommended if full-matrix LiRA evaluation is not feasible; underlying models must be drawn from multiple seeds per Section 3.4
- Multi-host merge validation: separate, small-scale correctness check, not part of the main experimental matrix

---

## 7. Infrastructure

The current Docker-based Swarm Learning deployment supports multi-container
testing on a single host but cannot validate genuine cross-host behavior,
which is required to substantiate the coordinator-free, cross-host
consensus claim (Section 1.2, Contribution 1). Recommended options for
multi-host testing:

- Azure for Students (subject to institutional email verification) - no
  payment method required
- Google Cloud Platform trial tier - $300 credit, 90-day validity

Either option is sufficient for the computational requirements of this
workload. A browser-based notebook environment (e.g., Google Colab) is not
viable, as it does not provide the containerization support required by the
Swarm Learning deployment model.
