"""
Parses swarm-learning training logs (e.g. exp_cascaded_dp_*.log) and extracts
one row per [CascadedDP-Trigger] event, giving a clean CSV to analyze
convergence timing, threshold crossings, and accuracy over epochs.

Usage:
    python3 parse_cascaded_logs.py log1.log log2.log ... -o out.csv

Or just drop all .log files in the same folder and run with no args --
it will glob *.log automatically.
"""

import re
import csv
import sys
import glob
import argparse
import os

# --- Regex patterns for the lines we care about ---

# [CascadedDP-Trigger] Node=0 | epoch=9 | grad_slope=0.664855 (thresh 0.494300, fail) | acc_variance=0.00023989 (thresh 0.00002240, fail) | local_converged=False
# NOTE: logs print "PASS" (uppercase) when a threshold passes and "fail"
# (lowercase) when it doesn't -- must match case-insensitively or every
# passing epoch gets silently dropped.
TRIGGER_RE = re.compile(
    r"\[CascadedDP-Trigger\]\s*Node=(?P<log_node>\d+)\s*\|\s*epoch=(?P<epoch>\d+)\s*\|\s*"
    r"grad_slope=(?P<grad_slope>[\d.eE+-]+)\s*\(thresh\s*(?P<grad_thresh>[\d.eE+-]+),\s*(?P<grad_pass>pass|fail)\)\s*\|\s*"
    r"acc_variance=(?P<acc_variance>[\d.eE+-]+)\s*\(thresh\s*(?P<var_thresh>[\d.eE+-]+),\s*(?P<var_pass>pass|fail)\)\s*\|\s*"
    r"local_converged=(?P<local_converged>True|False)",
    re.IGNORECASE,
)

# [CascadedDP-Quorum] Node 0 tracking global weighted flag value: 0.0000 (...)
QUORUM_RE = re.compile(
    r"\[CascadedDP-Quorum\]\s*Node\s*(?P<node>\d+)\s*tracking global weighted flag value:\s*(?P<flag>[\d.]+)"
)

# val_accuracy line from Keras, e.g.:
# ... - val_loss: 1.1688 - val_accuracy: 0.6812
VAL_ACC_RE = re.compile(r"val_accuracy:\s*([\d.]+)")

# Epoch marker Keras prints between epochs, e.g. "Epoch 9/50"
EPOCH_MARKER_RE = re.compile(r"Epoch\s+(\d+)/\d+")

# Filename pattern to pull condition/noise/run/node, e.g.:
# exp_baseline_ml1_run1.log            -> condition=baseline,     noise="",  node=ml1, run=1
# exp_cascaded_dp_0.5_ml1_run1.log     -> condition=cascaded_dp,  noise=0.5, node=ml1, run=1
# exp_full_dp_0.5_ml1_run1.log         -> condition=full_dp,      noise=0.5, node=ml1, run=1
# Only ml1/ml2 (.log) files are parsed here -- sl1/sl2 .json files are a
# different format (swarm-learning session metadata, not epoch logs).
FNAME_RE = re.compile(
    r"exp_(?P<condition>baseline|cascaded_dp|full_dp)_"
    r"(?:(?P<noise>[\d.]+)_)?"
    r"(?P<node_name>ml\d+)_run(?P<run>\d+)"
)


def parse_filename(path):
    base = os.path.basename(path)
    m = FNAME_RE.search(base)
    if not m:
        return {"condition": "", "noise": "", "node_name": "", "run": ""}
    return {
        "condition": m.group("condition"),
        "noise": m.group("noise") or "",
        "node_name": m.group("node_name"),
        "run": m.group("run"),
    }


EMPTY_TRIGGER_FIELDS = {
    "grad_slope": "", "grad_thresh": "", "grad_pass": "",
    "acc_variance": "", "var_thresh": "", "var_pass": "",
    "local_converged": "", "global_flag_value": "", "log_reported_node": "",
}


def parse_log(path):
    """
    Builds one row per epoch (1..N), for every condition. val_accuracy is
    always filled in from the Keras epoch-end line. For cascaded_dp logs,
    trigger fields (grad_slope, acc_variance, thresholds, local_converged,
    etc.) are ALSO filled in for any epoch where a [CascadedDP-Trigger] line
    appeared -- but epochs before a check happens (e.g. before
    MIN_DP_EPOCHS) still get a row with just val_accuracy and blank trigger
    fields, rather than being dropped.

    If a log reports trigger checks for more than one node (log_reported_node
    differs from the file's own ml1/ml2), the LAST trigger line seen for that
    epoch wins, since we want one row per epoch. If you need every individual
    node's trigger line broken out, ask and I can switch to multiple rows per
    epoch instead.
    """
    meta = parse_filename(path)

    with open(path, "r", errors="ignore") as f:
        content = f.read()

    # Split on \r\n or \n since these logs mix both
    lines = re.split(r"\r?\n", content)

    epoch_rows = {}  # epoch (str) -> row dict, in first-seen order
    current_epoch = ""
    last_val_acc = ""
    pending_quorum_flag = ""

    def get_or_create_row(epoch):
        if epoch not in epoch_rows:
            row = {
                "noise": meta["noise"],
                "run": meta["run"],
                "node": meta["node_name"],
                "condition": meta["condition"],
                "epoch": epoch,
                "val_accuracy": "",
                "source_file": os.path.basename(path),
            }
            row.update(EMPTY_TRIGGER_FIELDS)
            epoch_rows[epoch] = row
        return epoch_rows[epoch]

    for line in lines:
        epoch_match = EPOCH_MARKER_RE.search(line)
        if epoch_match:
            # "Epoch N/50" marks the START of epoch N (1-indexed). Finalize
            # the val_accuracy for the epoch that just ended (current_epoch)
            # before switching to the new one.
            if current_epoch != "" and last_val_acc != "":
                get_or_create_row(current_epoch)["val_accuracy"] = last_val_acc
            current_epoch = epoch_match.group(1)
            continue

        val_acc_match = VAL_ACC_RE.search(line)
        if val_acc_match:
            last_val_acc = val_acc_match.group(1)

        quorum_match = QUORUM_RE.search(line)
        if quorum_match:
            pending_quorum_flag = quorum_match.group("flag")

        trig_match = TRIGGER_RE.search(line)
        if trig_match:
            d = trig_match.groupdict()
            row = get_or_create_row(d["epoch"])
            row.update({
                "grad_slope": d["grad_slope"],
                "grad_thresh": d["grad_thresh"],
                "grad_pass": d["grad_pass"].lower() == "pass",
                "acc_variance": d["acc_variance"],
                "var_thresh": d["var_thresh"],
                "var_pass": d["var_pass"].lower() == "pass",
                "local_converged": d["local_converged"].lower() == "true",
                "global_flag_value": pending_quorum_flag,
                "log_reported_node": d["log_node"],
            })
            if last_val_acc:
                row["val_accuracy"] = last_val_acc

    # Flush the final epoch (no trailing "Epoch N/50" marker follows it)
    if current_epoch != "" and last_val_acc != "":
        row = get_or_create_row(current_epoch)
        if not row["val_accuracy"]:
            row["val_accuracy"] = last_val_acc

    return list(epoch_rows.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="Log file paths (defaults to *.log in cwd)")
    ap.add_argument("-o", "--out-prefix", default="results",
                     help="Prefix for output files -- writes <prefix>_baseline.csv, "
                          "<prefix>_full_dp.csv, <prefix>_cascaded_dp.csv")
    args = ap.parse_args()

    prefix = args.out_prefix
    if prefix.lower().endswith(".csv"):
        prefix = prefix[:-4]

    log_paths = args.logs if args.logs else glob.glob("*.log")
    if not log_paths:
        print("No log files found. Pass paths explicitly or run in a folder with .log files.")
        sys.exit(1)

    all_rows = []
    for path in log_paths:
        if not path.endswith(".log"):
            print(f"{path}: skipping (not a .log file)")
            continue
        rows = parse_log(path)
        n_with_trigger = sum(1 for r in rows if r["local_converged"] != "")
        print(f"{path}: {len(rows)} epoch rows ({n_with_trigger} with trigger-check data)")
        all_rows.extend(rows)

    if not all_rows:
        print("No matching rows found in any file. Check the regexes against your log format.")
        sys.exit(1)

    fieldnames = [
        "condition", "noise", "run", "node", "epoch",
        "grad_slope", "grad_thresh", "grad_pass",
        "acc_variance", "var_thresh", "var_pass",
        "local_converged", "global_flag_value", "log_reported_node",
        "val_accuracy", "source_file",
    ]

    by_condition = {"baseline": [], "full_dp": [], "cascaded_dp": []}
    unrecognized = []
    for row in all_rows:
        cond = row.get("condition", "")
        if cond in by_condition:
            by_condition[cond].append(row)
        else:
            unrecognized.append(row)

    for cond, rows in by_condition.items():
        out_path = f"{prefix}_{cond}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda r: (r["noise"], r["run"], r["node"], int(r["epoch"]))):
                writer.writerow(row)
        print(f"Wrote {len(rows)} rows to {out_path}")

    if unrecognized:
        out_path = f"{prefix}_unrecognized.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unrecognized)
        print(f"Wrote {len(unrecognized)} rows with unrecognized condition to {out_path} -- "
              f"check filename pattern")


if __name__ == "__main__":
    main()
