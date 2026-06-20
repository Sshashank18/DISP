"""
run_forced_lambda_chain.py -- lock in the inference-recomputed T1 lambda, then
run the full DISP + CAST chain with it.

Motivation
----------
check_lambda_infer.py recomputes the T1 lambda choice by REAL inference (alpha=1,
tau=0.90) and may land on a different argmin than the analytical (linear-response)
one the normal pipeline uses. This driver says: whatever lambda inference picked,
just use THAT lambda end-to-end. For each (model, method) it

  1. reads  lambda_check/<model>_<method>.json  -> "inferred_argmin"  (the lambda),
  2. forces that lambda into run_disp_infer.run_model_method (by pinning
     run_disp.LAMBDAS to a single value, so its internal argmin == that lambda)
     and regenerates T1/T2/T3 with real inference, OVERWRITING
     real_sweep_<model>.json,
  3. runs the CAST + DISP baseline (run_baselines.py T6), which rebuilds the
     DISP-best (T3) vectors from the just-written real_sweep_<model>.json -- so it
     automatically inherits the forced lambda.

Nothing in run_disp_infer.py / run_baselines.py is modified; the lambda is pinned
at runtime and the overwritten sweep is what carries it downstream.

Usage
-----
  python run_forced_lambda_chain.py --model llama-3.1-8b --methods CAA DIM
  python run_forced_lambda_chain.py --model qwen3-4b          # all methods with
                                                              # a lambda_check json
  python run_forced_lambda_chain.py --model gemma-3-4b --methods CAA --skip-cast
"""
import argparse
import json
import os
import subprocess
import sys

import run_disp as disp               # constants + analytical helpers
import run_disp_infer as ri           # run_model_method (real T1/T2/T3 inference)

HERE = os.path.dirname(os.path.abspath(__file__))
LAMCHECK_DIR = os.path.join(HERE, "lambda_check")
BASELINES = os.path.join(HERE, "Baselines")


def discover_methods(model):
    """Methods that have a lambda_check/<model>_<method>.json available."""
    found = []
    for me in disp.METHODS:
        if os.path.exists(os.path.join(LAMCHECK_DIR, f"{model}_{me}.json")):
            found.append(me)
    return found


def read_inferred_lambda(model, method):
    """Return (inferred_lambda, analytical_lambda, matched) from lambda_check."""
    path = os.path.join(LAMCHECK_DIR, f"{model}_{method}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no lambda_check result at {path}; run check_lambda_infer.py "
            f"--model {model} --methods {method} first")
    with open(path) as f:
        rec = json.load(f)
    inf = float(rec["inferred_argmin"])
    ana = float(rec["analytical_argmin"])
    if inf not in disp.LAMBDAS:
        raise ValueError(
            f"inferred_argmin {inf} for {model}/{method} is not in "
            f"disp.LAMBDAS {disp.LAMBDAS}; refusing to force an off-grid lambda")
    return inf, ana, bool(rec.get("match", inf == ana))


def run_disp_with_lambda(model, method, lam, results, out_path):
    """Pin run_disp.LAMBDAS to [lam] so run_model_method's internal argmin is
    forced to lam, run real T1/T2/T3, then restore the grid."""
    saved = disp.LAMBDAS
    disp.LAMBDAS = [lam]                # the only candidate -> min() returns it
    try:
        ri.run_model_method(model, method, results, out_path)
    finally:
        disp.LAMBDAS = saved
    # sanity: the written record must carry the forced lambda
    got = results.get(model, {}).get(method, {}).get("lambda")
    if got is not None and abs(float(got) - lam) > 1e-9:
        raise RuntimeError(
            f"forced lambda {lam} but real_sweep recorded {got} for "
            f"{model}/{method}")


def run_cast_disp_baseline(model, method):
    """Run run_baselines.py T6 (CAST + DISP-best T3) for (model, method)."""
    cmd = [sys.executable, "run_baselines.py",
           "--model", model, "--method", method, "--task", "6"]
    print(f"\n>>> CAST+DISP baseline (T6): {' '.join(cmd)}  (cwd={BASELINES})",
          flush=True)
    subprocess.run(cmd, cwd=BASELINES, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=disp.MODELS)
    ap.add_argument("--methods", nargs="+", choices=disp.METHODS, default=None,
                    help="default: every method with a lambda_check json")
    ap.add_argument("--skip-cast", action="store_true",
                    help="only force lambda + regenerate T1/T2/T3; skip T6")
    args = ap.parse_args()

    methods = args.methods or discover_methods(args.model)
    if not methods:
        sys.exit(f"no lambda_check results found for {args.model} in "
                 f"{LAMCHECK_DIR}; run check_lambda_infer.py first")

    out_path = os.path.join(ri.OUT_DIR, f"real_sweep_{args.model}.json")
    # resume/merge so methods we don't touch this run are preserved
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    print(f"=== forced-lambda chain: {args.model}  methods={methods} ===")
    print(f"    real_sweep -> {out_path}  (overwritten in place)\n")

    plan = {}
    for method in methods:
        lam, ana, matched = read_inferred_lambda(args.model, method)
        plan[method] = lam
        flag = "MATCH (no change)" if matched else "DIFFERS -> using inferred"
        print(f"  {method}: analytical lambda={ana}  inferred lambda={lam}  "
              f"[{flag}]")
    print()

    # Stage 1: regenerate T1/T2/T3 with the forced lambda (loads model per method)
    for method in methods:
        print(f"\n########## DISP T1/T2/T3 @ forced lambda={plan[method]}  "
              f"{args.model}/{method} ##########")
        run_disp_with_lambda(args.model, method, plan[method], results, out_path)

    # Stage 2: CAST + DISP baseline (reads the overwritten real_sweep -> same lambda)
    if args.skip_cast:
        print("\n[--skip-cast] not running T6 CAST+DISP baseline.")
    else:
        for method in methods:
            run_cast_disp_baseline(args.model, method)

    print(f"\n=== done: {args.model} methods={methods} ===")
    print(f"    DISP sweep:     {out_path}")
    if not args.skip_cast:
        print(f"    CAST+DISP (T6): {BASELINES}/outputs/{args.model}/"
              f"<method>/T6_CAST_DISP_T3_<method>.json")


if __name__ == "__main__":
    main()
