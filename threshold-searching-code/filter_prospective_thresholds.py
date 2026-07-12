"""
Filter cascaded_dp_sweep_detail.csv down to a ranked table of prospective
GRAD_THRESHOLD / ACC_VAR_THRESHOLD percentile combos.

Reads the per-run detail file produced by cascaded_dp_threshold_sweep.py
(one row per grad_percentile, var_percentile, noise, run) and aggregates
it directly -- so you don't need the separate summary CSV.

Usage:
    python filter_prospective_thresholds.py
"""

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DETAIL_CSV = "cascaded_dp_sweep_detail.csv"

# A combo is only "reliable" if quorum fires in ALL runs, at ALL noise levels.
# Set to False to allow partial reliability (not recommended for final pick).
REQUIRE_FULL_RELIABILITY = True

# How many top candidates to show, ranked by smallest accuracy gap at quorum.
TOP_N = 15


# ---------------------------------------------------------------------------
# LOGIC
# ---------------------------------------------------------------------------

def load_detail(path):
    return pd.read_csv(path)


def aggregate_per_combo_noise(detail):
    """Collapse runs -> one row per (grad_pctile, var_pctile, noise)."""
    g = (
        detail.groupby(["grad_percentile", "var_percentile", "noise"])
        .agg(
            n_runs=("run", "count"),
            n_quorum_reached=("quorum_reached", "sum"),
            avg_quorum_epoch=("quorum_epoch", "mean"),
            avg_val_acc_at_quorum=("val_acc_at_quorum", "mean"),
            avg_final_val_acc=("final_val_acc", "mean"),
        )
        .reset_index()
    )
    g["acc_gap_at_quorum"] = g["avg_final_val_acc"] - g["avg_val_acc_at_quorum"]
    g["fully_reached"] = g["n_quorum_reached"] == g["n_runs"]
    return g


def aggregate_across_noise(per_combo_noise, require_full_reliability):
    """Collapse noise levels -> one row per (grad_pctile, var_pctile)."""
    rows = []
    for (gp, vp), sub in per_combo_noise.groupby(["grad_percentile", "var_percentile"]):
        n_noise_levels = sub.shape[0]
        n_noise_fully_reached = sub["fully_reached"].sum()
        reliable_everywhere = n_noise_fully_reached == n_noise_levels

        if require_full_reliability and not reliable_everywhere:
            continue

        rows.append({
            "grad_percentile": gp,
            "var_percentile": vp,
            "noise_levels_covered": n_noise_levels,
            "noise_levels_fully_reliable": n_noise_fully_reached,
            "avg_quorum_epoch": sub["avg_quorum_epoch"].mean(),
            "avg_acc_gap": sub["acc_gap_at_quorum"].mean(),
            "max_acc_gap": sub["acc_gap_at_quorum"].max(),
            "min_quorum_epoch": sub["avg_quorum_epoch"].min(),
            "max_quorum_epoch": sub["avg_quorum_epoch"].max(),
        })

    return pd.DataFrame(rows)


# Accuracy-gap budgets (in accuracy fraction, e.g. 0.02 = 2 percentage points)
# to test. For each budget, we find the combo(s) meeting avg_acc_gap <= budget
# that fire quorum EARLIEST (i.e. maximize time-savings subject to the
# accuracy constraint).
GAP_BUDGETS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]


def best_combo_for_budget(combo_table, budget):
    """Among combos with avg_acc_gap <= budget, return the one with the
    earliest (smallest) avg_quorum_epoch. Returns None if nothing qualifies."""
    eligible = combo_table[combo_table["avg_acc_gap"] <= budget]
    if eligible.empty:
        return None
    return eligible.sort_values("avg_quorum_epoch").iloc[0]


def build_budget_table(combo_table, budgets):
    rows = []
    for b in budgets:
        best = best_combo_for_budget(combo_table, b)
        if best is None:
            rows.append({
                "gap_budget": b, "grad_percentile": None, "var_percentile": None,
                "avg_quorum_epoch": None, "avg_acc_gap": None,
                "note": "no combo meets this budget",
            })
        else:
            rows.append({
                "gap_budget": b,
                "grad_percentile": best["grad_percentile"],
                "var_percentile": best["var_percentile"],
                "avg_quorum_epoch": best["avg_quorum_epoch"],
                "avg_acc_gap": best["avg_acc_gap"],
                "note": "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    detail = load_detail(DETAIL_CSV)

    per_combo_noise = aggregate_per_combo_noise(detail)
    combo_table = aggregate_across_noise(per_combo_noise, REQUIRE_FULL_RELIABILITY)

    if combo_table.empty:
        print("No combos met the reliability requirement. "
              "Try REQUIRE_FULL_RELIABILITY = False to see all combos.")
    else:
        print("\n=== All reliable combos (sorted by accuracy gap, for reference) ===\n")
        print(combo_table.sort_values("avg_acc_gap").head(TOP_N).to_string(index=False))

        budget_table = build_budget_table(combo_table, GAP_BUDGETS)
        print("\n=== Earliest-firing combo per accuracy-gap budget "
              "(gap budget -> fastest combo meeting it) ===\n")
        print(budget_table.to_string(index=False))

        combo_table.sort_values("avg_acc_gap").to_csv("prospective_thresholds.csv", index=False)
        budget_table.to_csv("threshold_by_budget.csv", index=False)
        print("\nSaved: prospective_thresholds.csv, threshold_by_budget.csv")