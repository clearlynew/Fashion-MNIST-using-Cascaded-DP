"""
Pulls the actual threshold/accuracy/gradient/variance numbers for a chosen
(grad_percentile, var_percentile) combo from the Cascaded DP sweep, plus the
raw per-epoch grad_slope / acc_variance rows from ml1 and ml2 around each
node's trigger epoch.

Edit GRAD_PCT / VAR_PCT below to match whichever row you picked from
thresholds_by_metrics.csv.
"""

import csv

# ---- CONFIG: set to your chosen combo ----
GRAD_PCT = "35"
VAR_PCT = "40"

SWEEP_FILE = "Cascaded_DP_sweep.csv"
ML1_FILE = "metrics_ml1.csv"
ML2_FILE = "metrics_ml2.csv"
OUT_FILE = "chosen_run_values.csv"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_sweep_rows(grad_pct, var_pct):
    rows = load_csv(SWEEP_FILE)
    return [r for r in rows if r["grad_percentile"] == grad_pct and r["var_percentile"] == var_pct]


def get_metric_row(metrics, node, noise, run, epoch):
    for r in metrics:
        if (
            r["node"] == node
            and float(r["noise_multiplier"]) == float(noise)
            and r["run"] == run
            and float(r["epoch"]) == float(epoch)
        ):
            return r
    return None


def main():
    sweep_rows = get_sweep_rows(GRAD_PCT, VAR_PCT)
    if not sweep_rows:
        print(f"No rows found for grad_percentile={GRAD_PCT}, var_percentile={VAR_PCT}")
        return

    ml1 = load_csv(ML1_FILE)
    ml2 = load_csv(ML2_FILE)

    print(f"=== Combo grad_percentile={GRAD_PCT}, var_percentile={VAR_PCT} ===\n")

    out_rows = []

    for r in sweep_rows:
        noise, run = r["noise"], r["run"]
        print(f"--- noise={noise}  run={run} ---")
        print(f"  grad_thr={r['grad_thr']}  var_thr={r['var_thr']}")
        print(f"  ml1_trigger_epoch={r['ml1_trigger']}  ml2_trigger_epoch={r['ml2_trigger']}")
        print(f"  quorum_epoch={r['quorum_epoch']}  quorum_reached={r['quorum_reached']}")
        print(f"  val_acc_at_quorum={r['val_acc_at_quorum']}  final_val_acc={r['final_val_acc']}")

        row1 = row2 = None
        if r["ml1_trigger"]:
            row1 = get_metric_row(ml1, "ml1", noise, run, r["ml1_trigger"])
            if row1:
                print(f"  [ml1 @ epoch {r['ml1_trigger']}] grad_slope={row1['grad_slope']}  "
                      f"acc_variance={row1['acc_variance']}  val_acc={row1['val_acc']}")

        if r["ml2_trigger"]:
            row2 = get_metric_row(ml2, "ml2", noise, run, r["ml2_trigger"])
            if row2:
                print(f"  [ml2 @ epoch {r['ml2_trigger']}] grad_slope={row2['grad_slope']}  "
                      f"acc_variance={row2['acc_variance']}  val_acc={row2['val_acc']}")
        print()

        out_rows.append({
            "grad_percentile": GRAD_PCT,
            "var_percentile": VAR_PCT,
            "noise": noise,
            "run": run,
            "grad_thr": r["grad_thr"],
            "var_thr": r["var_thr"],
            "ml1_trigger_epoch": r["ml1_trigger"],
            "ml2_trigger_epoch": r["ml2_trigger"],
            "quorum_epoch": r["quorum_epoch"],
            "quorum_reached": r["quorum_reached"],
            "val_acc_at_quorum": r["val_acc_at_quorum"],
            "final_val_acc": r["final_val_acc"],
            "ml1_grad_slope": row1["grad_slope"] if row1 else "",
            "ml1_acc_variance": row1["acc_variance"] if row1 else "",
            "ml1_val_acc": row1["val_acc"] if row1 else "",
            "ml2_grad_slope": row2["grad_slope"] if row2 else "",
            "ml2_acc_variance": row2["acc_variance"] if row2 else "",
            "ml2_val_acc": row2["val_acc"] if row2 else "",
        })

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved {len(out_rows)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
