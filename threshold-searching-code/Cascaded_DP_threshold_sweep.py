"""
CascadedDP threshold sweep tool.

Simulates, on already-collected per-epoch metrics (grad_slope, acc_variance,
val_acc per node/run/noise level), what epoch quorum would fire at for
different GRAD_THRESHOLD / ACC_VAR_THRESHOLD percentile choices -- without
needing to rerun training.

Usage:
    python cascaded_DP_threshold_sweep.py

Edit the CONFIG section below to point at your CSVs and choose which
percentile combinations to sweep.
"""

import pandas as pd
import itertools

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CSV_PATHS = [
    "metrics_ml1.csv",
    "metrics_ml2.csv",
]

# Percentiles to sweep for each signal. The script will try every combination
# of (grad_pctile, var_pctile). Set both lists to the same single value if
# you just want one threshold pair.
GRAD_PERCENTILES = list(range(10, 51, 5))       # 10,15,20,...,50
ACC_VAR_PERCENTILES = list(range(10, 51, 5))    # 10,15,20,...,50

# Quorum rule: "and" = both nodes must locally converge (current design,
# one-way latch -> quorum epoch = max of node trigger epochs).
# "or" = first node to converge fires quorum (sensitivity check only).
QUORUM_RULE = "and"  # "and" or "or"

NODE_COL = "node"
RUN_COL = "run"
NOISE_COL = "noise_multiplier"
EPOCH_COL = "epoch"
GRAD_COL = "grad_slope"
VAR_COL = "acc_variance"
ACC_COL = "val_acc"


# ---------------------------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------------------------

def load_data(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df


def compute_thresholds(df, grad_pctile, var_pctile):
    """Per-noise-level thresholds from the pooled distribution at a given percentile."""
    grad_thr = df.groupby(NOISE_COL)[GRAD_COL].quantile(grad_pctile / 100).to_dict()
    var_thr = df.groupby(NOISE_COL)[VAR_COL].quantile(var_pctile / 100).to_dict()
    return grad_thr, var_thr


def node_trigger_epoch(node_df, grad_thr, var_thr, noise):
    """First epoch where a single node satisfies the AND-condition locally."""
    node_df = node_df.sort_values(EPOCH_COL)
    hit = node_df[
        (node_df[GRAD_COL] < grad_thr[noise]) & (node_df[VAR_COL] < var_thr[noise])
    ]
    return hit[EPOCH_COL].iloc[0] if len(hit) else None


def simulate_run(run_df, grad_thr, var_thr, noise, quorum_rule):
    nodes = run_df[NODE_COL].unique()
    triggers = {
        n: node_trigger_epoch(run_df[run_df[NODE_COL] == n], grad_thr, var_thr, noise)
        for n in nodes
    }

    fired = [e for e in triggers.values() if e is not None]

    if quorum_rule == "and":
        if len(fired) < len(nodes):
            quorum_epoch = None  # not all nodes converged -> no quorum
        else:
            quorum_epoch = max(fired)
    elif quorum_rule == "or":
        quorum_epoch = min(fired) if fired else None
    else:
        raise ValueError("QUORUM_RULE must be 'and' or 'or'")

    val_acc_at_quorum = None
    if quorum_epoch is not None:
        accs = []
        for n in nodes:
            row = run_df[(run_df[NODE_COL] == n) & (run_df[EPOCH_COL] == quorum_epoch)]
            if len(row):
                accs.append(row[ACC_COL].iloc[0])
        if accs:
            val_acc_at_quorum = sum(accs) / len(accs)

    final_accs = [
        run_df[run_df[NODE_COL] == n].sort_values(EPOCH_COL).iloc[-1][ACC_COL]
        for n in nodes
    ]
    final_val_acc = sum(final_accs) / len(final_accs)

    return {
        **{f"{n}_trigger": e for n, e in triggers.items()},
        "quorum_epoch": quorum_epoch,
        "val_acc_at_quorum": val_acc_at_quorum,
        "final_val_acc": final_val_acc,
        "quorum_reached": quorum_epoch is not None,
    }


def run_sweep(df, grad_percentiles, acc_var_percentiles, quorum_rule):
    all_rows = []
    for grad_p, var_p in itertools.product(grad_percentiles, acc_var_percentiles):
        grad_thr, var_thr = compute_thresholds(df, grad_p, var_p)
        for noise, g in df.groupby(NOISE_COL):
            for run, run_df in g.groupby(RUN_COL):
                res = simulate_run(run_df, grad_thr, var_thr, noise, quorum_rule)
                all_rows.append({
                    "grad_percentile": grad_p,
                    "var_percentile": var_p,
                    "noise": noise,
                    "run": run,
                    "grad_thr": round(grad_thr[noise], 4),
                    "var_thr": var_thr[noise],
                    **res,
                })
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = load_data(CSV_PATHS)
    sweep = run_sweep(df, GRAD_PERCENTILES, ACC_VAR_PERCENTILES, QUORUM_RULE)
    sweep.to_csv("cascaded_dp_sweep_detail.csv", index=False)
    print("Saved: cascaded_dp_sweep_detail.csv")