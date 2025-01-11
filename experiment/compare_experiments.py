#!/usr/bin/env python3
"""
search_k_top_for_same_bw.py

We want to match the bandwidth of diloco_int8 with DeMo (demo_int8).
Steps:
 1) Run "exp2.py" once with mode=diloco_int8 => parse baseline BW, baseline loss.
 2) Use binary search on k_top for DeMo to find a k_top that yields ~the same total BW.
 3) Print final results.
"""

import subprocess
import re
import os
import sys

###############################################################################
# Parsing Helpers
###############################################################################
def parse_loss_from_output(output_str):
    """
    Looks for: "[Mode=XYZ] rank=0 done. Final avg loss=X.XXXX"
    Returns float or None.
    """
    m = re.search(r"Final avg loss=([0-9\.]+)", output_str)
    if m:
        return float(m.group(1))
    return None

def parse_bw_from_output(output_str):
    """
    Looks for: "[Mode=XYZ] rank=0 total cluster bandwidth=Y.YYY MB"
    Returns float or None.
    """
    m = re.search(r"total cluster bandwidth=([0-9\.]+)\s*MB", output_str)
    if m:
        return float(m.group(1))
    return None

###############################################################################
# Runner for exp2.py
###############################################################################
def run_exp2(mode, k_top=None):
    """
    Runs exp2.py with fixed hyperparams, streaming logs in real time.
    Returns (final_loss, final_bw, raw_output) or (None, None, raw_output) if error.
    """
    cmd = [
        "python3", "exp2.py",
        "--world_size=2",
        f"--mode={mode}",
        "--num_outer_steps=10",
        "--local_steps=5",
        "--outer_lr=0.1",
        "--outer_momentum=0.9",
        "--batch_size=8",
        "--num_samples=200",
        # You can tweak other args if needed (seq_len, vocab_size, etc.)
    ]
    if mode == "demo_int8" and k_top is not None:
        cmd.append(f"--k_top={k_top}")

    print(f"\n[search_k_top_for_same_bw] Running: {' '.join(cmd)}\n")
    sys.stdout.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
        bufsize=1
    )

    output_lines = []
    with proc:
        try:
            for line in proc.stdout:
                print(line, end='')
                sys.stdout.flush()
                output_lines.append(line)
        except KeyboardInterrupt:
            print("KeyboardInterrupt... terminating.")
            proc.terminate()
            proc.wait()

    retcode = proc.wait()
    raw_out = "".join(output_lines)
    if retcode != 0:
        return (None, None, raw_out)

    final_loss = parse_loss_from_output(raw_out)
    final_bw   = parse_bw_from_output(raw_out)
    return (final_loss, final_bw, raw_out)

###############################################################################
# Main
###############################################################################
def main():
    # 1) Run the baseline DiLoCo
    print("=== Baseline: DiLoCo int8 ===")
    base_loss, base_bw, base_log = run_exp2(mode="diloco_int8", k_top=None)
    if base_loss is None or base_bw is None:
        print("Error: baseline run failed. Exiting.")
        return
    print(f"Baseline => Loss={base_loss:.4f}, BW={base_bw:.3f} MB")

    # 2) We'll do a binary search on k_top for demo_int8
    #    We want to get final_bw ~ base_bw
    #    We'll keep it in [1..some_upper_bound]. Let's guess 30000 as upper bound.
    #    You can refine if your param count is smaller or bigger.
    left, right = 1, 30000
    best_k_top = None
    best_diff  = float('inf')
    target_bw  = base_bw
    tolerance  = 0.05  # MB tolerance for "close enough" (~50 KB)

    # We'll do ~10-15 iterations max
    for _ in range(15):
        mid = (left + right) // 2
        print(f"\nBinarySearch: Trying k_top={mid} in DeMo (demo_int8) ...")
        mid_loss, mid_bw, mid_log = run_exp2(mode="demo_int8", k_top=mid)
        if mid_loss is None or mid_bw is None:
            print("Something went wrong. We'll just reduce 'right' and continue.")
            right = mid - 1
            continue

        diff = abs(mid_bw - target_bw)
        print(f"  -> got Loss={mid_loss:.4f}, BW={mid_bw:.3f} MB, diff={diff:.3f} MB")

        # if best so far, store
        if diff < best_diff:
            best_diff  = diff
            best_k_top = mid
            if diff <= tolerance:
                # close enough, let's break early
                break

        # if mid_bw < target_bw => we want more freq => increase k_top
        # else if mid_bw > target_bw => we want fewer freq => decrease k_top
        if mid_bw < target_bw:
            left = mid + 1
        else:
            right = mid - 1

    if best_k_top is None:
        print("No valid k_top found via binary search. :(")
        return

    print(f"\n=== Found best_k_top={best_k_top} with diff={best_diff:.3f} MB to baseline BW={target_bw:.3f} MB ===")

    # 3) Final: run a fresh experiment with best_k_top
    final_loss, final_bw, final_log = run_exp2(mode="demo_int8", k_top=best_k_top)
    if final_loss is None or final_bw is None:
        print("Error: final run with best_k_top failed.")
        return

    rel_loss = final_loss / base_loss if base_loss > 1e-9 else 999.0

    print("\n=========================")
    print(f"Baseline DiLoCo => Loss={base_loss:.4f}, BW={base_bw:.3f} MB")
    print(f"DeMo k_top={best_k_top} => Loss={final_loss:.4f}, RelLoss={rel_loss:.3f}, BW={final_bw:.3f} MB")

if __name__ == "__main__":
    main()
