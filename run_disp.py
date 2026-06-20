"""
DISP ablation pipeline (analytical, no inference).

Methods covered: CAA, DIM, ACE
Models covered:  llama-3.1-8b, gemma-3-4b, gemma-3-12b, qwen3-4b, qwen3-8b

Tasks (each feeds best hyperparam forward):
  T1  Grid \lambda \in {0.5,...,1.0} for bidirectional weighting.
       SVD energy threshold = 0.90, alpha_k = 1.0 (hard projection).
  T2  At T1 best \lambda, grid alpha_k per SVD direction over {0.0,0.1,...,1.5}.
       Threshold = 0.90.  Subject to |self_after| \geq 0.9 |self_CAA|; among
       valid combos pick the one minimising side-effect.
  T3  At T1 best \lambda and T2 alpha strategy, grid SVD energy threshold
       over {0.50,0.55,...,0.99}.

Per-target weight (j \neq i, target i):
  out_norm = |pct[i,j]| / max_j' |pct[i,j']|
  in_norm  = |pct[j,i]| / max_i' |pct[i',i]|
  w_j      = \lambda * out_norm + (1-\lambda) * in_norm

Side-effect metric (single scalar per matrix):
  SE% = \sum_{i \neq j} |pct[i,j]|   (pct already in %)

Implementation notes
  * The estimated post-DISP pct matrix is analytical: corrected vector v_i' is
    expressed as a linear combination of original method vectors via dot/norm^2,
    then est_row_i = sum_k c_k * pct[k,:].
  * Per-direction alpha grid is fully vectorised over the Cartesian product
    when k <= ALPHA_GRID_K_CAP (default 4 -> at most 16^4 = 65536 combos).
    For k > cap we fall back to greedy per-direction search (still vectorised
    over the 16-value 1-D grid for each direction).
"""
import argparse
import json
import os
import numpy as np
import torch

BASE = "/home/gpuuser1/gpuuser1_a/Shashank/STEERING/Final"
OUT_DIR = os.path.join(BASE, "DISP")
MODELS = ["llama-3.1-8b", "gemma-3-4b", "gemma-3-12b", "qwen3-4b", "qwen3-8b"]
METHODS = ["CAA", "DIM", "ACE"]

LAMBDAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
              0.85, 0.90, 0.95, 0.99]
ALPHA_GRID = np.round(np.arange(0.0, 1.51, 0.05), 3).tolist()  # 31 values
DEFAULT_THRESHOLD = 0.90
SELF_FLOOR = 0.85           # |self_after| must be >= SELF_FLOOR * |self_baseline|


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_data(model, method):
    rpt_path = os.path.join(BASE, "outputs", model, method,
                            "baseline_delta_report.json")
    with open(rpt_path) as f:
        rpt = json.load(f)
    behaviors = rpt["behaviors"]
    pct = np.array(rpt["pct_change_matrix"], dtype=np.float64)
    # desired direction per behavior (+1 = increase, -1 = decrease)
    # CAA/ACE/DIM all inject sign_k * alpha * v_k at inference time, so
    # pct[k, :] reflects that signed injection. For the analytical
    # linear-response estimator to combine multiple basis vectors
    # correctly we must multiply each cross term by sign_i * sign_k.
    desired = rpt["desired_directions"]
    sign = np.array(
        [+1.0 if desired[b] == "increase" else -1.0 for b in behaviors],
        dtype=np.float64,
    )
    vec_dir = os.path.join(BASE, "vectors", model, method)
    vectors = []
    for b in behaviors:
        d = torch.load(os.path.join(vec_dir, f"{b}.pt"),
                       weights_only=False, map_location="cpu")
        vectors.append(d["vector"].float().numpy().astype(np.float64))
    V = np.stack(vectors, axis=0)        # (N, d)
    return behaviors, pct, V, sign


# ---------------------------------------------------------------------------
# Core DISP per target
# ---------------------------------------------------------------------------

def bidir_weights(pct, target_idx, lambda_val):
    row = np.abs(pct[target_idx, :])
    col = np.abs(pct[:, target_idx])
    rmax = row.max() if row.max() > 0 else 1.0
    cmax = col.max() if col.max() > 0 else 1.0
    w = np.zeros(pct.shape[0])
    for j in range(pct.shape[0]):
        if j == target_idx:
            continue
        w[j] = (lambda_val * abs(pct[target_idx, j]) / rmax
                + (1 - lambda_val) * abs(pct[j, target_idx]) / cmax)
    return w


def disp_one(target_idx, pct, V, sign, lambda_val, energy_threshold,
             alpha_strategy):
    """
    Returns: dict with keys
      'est_row'    estimated pct row (length N)
      'k'          number of SVD directions used
      'alphas'     final alpha vector (length k)
      'corrected'  bool indicating whether projection was applied
    """
    v = V[target_idx].copy()
    orig_norm = np.linalg.norm(v)
    orig_self = abs(pct[target_idx, target_idx])

    w = bidir_weights(pct, target_idx, lambda_val)
    mask = w > 1e-10
    if not mask.any():
        return {"est_row": pct[target_idx].copy(), "k": 0,
                "alphas": [], "corrected": False}

    I_mat = (w[mask, None] * V[mask])               # (m, d)
    _, S, Vh = np.linalg.svd(I_mat, full_matrices=False)
    energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    k = int((energy < energy_threshold).sum()) + 1
    k = max(1, min(k, Vh.shape[0]))

    # unit right-singular vectors
    top = Vh[:k]                                    # (k, d)
    top = top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-12)

    proj_v = top @ v                                # (k,)
    proj_v_sq = proj_v ** 2
    vec_norms_sq = (V * V).sum(axis=1)              # (N,)
    base_overlap = (V @ v) / (vec_norms_sq + 1e-12) # (N,)  <v_i, v_k>/||v_k||^2
    dir_overlap = (top @ V.T) / (vec_norms_sq + 1e-12)  # (k, N)  <d_d, v_k>/||v_k||^2
    pct_target_row = pct[target_idx].copy()         # (N,)
    # sign_i * sign_k for each k in the basis: applied to every
    # cross-vector contribution so the estimator matches the signed
    # injection used by steering.apply() at inference time.
    sign_prod = sign[target_idx] * sign                              # (N,)

    def estimated_rows(alpha_arr):
        """
        Linear-response estimator with rescaling.

        v_i' (after subtract) = v_i + Dv0,   Dv0 = -sum_d a_d * proj_v_d * d_d
        v_i_final = scale * v_i',            scale = ||v_i|| / ||v_i'||
        Dv = v_i_final - v_i = (scale-1) * v_i + scale * Dv0
        c_k = <Dv, v_k> / ||v_k||^2
            = (scale-1) * base_overlap[k] + scale * Dv0_overlap[k]
        Dv0_overlap[k] = -sum_d a_d * proj_v_d * dir_overlap[d, k]

        effect ~= pct[i, :] + sum_k c_k * pct[k, :]
        At a=0: scale=1, Dv0=0, c=0  =>  effect = pct[i, :]  (exact)

        alpha_arr shape: (G, k)
        Returns est shape (G, N). Invalid combos (norm collapse) -> NaN row.
        """
        norm_sq = (orig_norm ** 2
                   - ((2 * alpha_arr - alpha_arr ** 2)
                      * proj_v_sq[None, :]).sum(axis=1))
        bad = norm_sq < 1e-12
        norm_sq = np.where(bad, 1.0, norm_sq)
        scale = orig_norm / (np.sqrt(norm_sq) + 1e-12)          # (G,)

        # Dv0_overlap[g, k] = -sum_d alpha[g,d] * proj_v[d] * dir_overlap[d, k]
        weighted_alpha = alpha_arr * proj_v[None, :]            # (G, k)
        dv0_overlap = -weighted_alpha @ dir_overlap             # (G, N)

        # c_k per grid combo
        c = ((scale[:, None] - 1.0) * base_overlap[None, :]
             + scale[:, None] * dv0_overlap)                    # (G, N)
        # Sign-correct each cross-term: contribution of basis vector k to
        # est_row is sign_i * sign_k * c_k * pct[k, :], because
        # pct[k, :] was measured under signed injection sign_k * v_k.
        c = c * sign_prod[None, :]                              # (G, N)

        est = pct_target_row[None, :] + c @ pct                 # (G, N)
        if bad.any():
            est[bad] = np.nan
        return est

    if alpha_strategy == "fixed":
        alphas = np.ones(k)
        est = estimated_rows(alphas[None, :])[0]
        if np.isnan(est).any():
            est = pct[target_idx].copy()
            return {"est_row": est, "k": k, "alphas": alphas.tolist(),
                    "corrected": False}
        return {"est_row": est, "k": k, "alphas": alphas.tolist(),
                "corrected": True}

    # alpha_strategy == "grid": greedy per-direction at every k.
    # For each direction d, scan ALPHA_GRID while holding the other
    # directions fixed at their current best, accept the value that
    # minimises side-effect subject to |self_after| >= SELF_FLOOR * orig_self.
    # alpha=0 is always feasible (retention = 1 under linear response),
    # so we are guaranteed at least the baseline row.
    g = np.array(ALPHA_GRID, dtype=np.float64)
    G = len(g)
    alphas = np.zeros(k)
    for d_idx in range(k):
        candidates = np.tile(alphas[None, :], (G, 1))
        candidates[:, d_idx] = g
        est = estimated_rows(candidates)                  # (G, N)
        valid = ~np.isnan(est).any(axis=1)
        if orig_self > 1e-12:
            self_after = np.abs(est[:, target_idx])
            valid &= (self_after >= SELF_FLOOR * orig_self)
        sides = np.abs(est).sum(axis=1) - np.abs(est[:, target_idx])
        sides_valid = np.where(valid, sides, np.inf)
        alphas[d_idx] = float(g[int(np.argmin(sides_valid))])

    est_final = estimated_rows(alphas[None, :])[0]
    if np.isnan(est_final).any():
        return {"est_row": pct[target_idx].copy(), "k": k,
                "alphas": alphas.tolist(), "corrected": False}
    return {"est_row": est_final, "k": k, "alphas": alphas.tolist(),
            "corrected": True}


def run_config(behaviors, pct, V, sign, lambda_val, threshold,
               alpha_strategy):
    N = len(behaviors)
    est_matrix = np.zeros((N, N))
    per_target = []
    for i in range(N):
        info = disp_one(i, pct, V, sign, lambda_val, threshold,
                        alpha_strategy)
        est_matrix[i] = info["est_row"]
        per_target.append({
            "behavior": behaviors[i],
            "k": info["k"],
            "alphas": info["alphas"],
            "corrected": info["corrected"],
        })
    return total_side_effect(est_matrix), est_matrix, per_target


def total_side_effect(M):
    return float(np.abs(M - np.diag(np.diag(M))).sum())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_model_method(model, method):
    print(f"\n  [{model}/{method}]")
    behaviors, pct, V, sign = load_data(model, method)
    caa_se = total_side_effect(pct)
    print(f"    baseline SE% = {caa_se:.2f}")

    # Task 1: lambda grid, threshold 0.90, alpha=1
    t1_grid = {}
    for lam in LAMBDAS:
        se, _, _ = run_config(behaviors, pct, V, sign, lam,
                              DEFAULT_THRESHOLD, "fixed")
        t1_grid[lam] = se
    best_lambda = min(t1_grid, key=t1_grid.get)
    t1_se = t1_grid[best_lambda]
    print(f"    T1 best lambda={best_lambda} SE%={t1_se:.2f} "
          f"ratio={caa_se/t1_se:.3f}x")

    # Task 2: grid alpha at best lambda, threshold 0.90
    t2_se, _, t2_targets = run_config(behaviors, pct, V, sign, best_lambda,
                                      DEFAULT_THRESHOLD, "grid")
    print(f"    T2 (grid alpha) SE%={t2_se:.2f} "
          f"ratio={caa_se/t2_se:.3f}x")

    # Task 3: threshold grid, alpha=grid, lambda=best
    t3_grid = {}
    t3_alpha_per_thr = {}
    for th in THRESHOLDS:
        se, _, targets = run_config(behaviors, pct, V, sign,
                                    best_lambda, th, "grid")
        t3_grid[th] = se
        t3_alpha_per_thr[th] = targets
    best_th = min(t3_grid, key=t3_grid.get)
    t3_se = t3_grid[best_th]
    print(f"    T3 best thr={best_th:.2f} SE%={t3_se:.2f} "
          f"ratio={caa_se/t3_se:.3f}x")

    return {
        "baseline_se_pct": caa_se,
        "task1": {
            "grid_lambda_to_se": t1_grid,
            "best_lambda": best_lambda,
            "se_pct": t1_se,
            "ratio": caa_se / t1_se,
            "fixed": {"threshold": DEFAULT_THRESHOLD, "alpha_k": 1.0},
        },
        "task2": {
            "lambda": best_lambda,
            "threshold": DEFAULT_THRESHOLD,
            "se_pct": t2_se,
            "ratio": caa_se / t2_se,
            "per_target_alphas": t2_targets,
        },
        "task3": {
            "lambda": best_lambda,
            "grid_threshold_to_se": t3_grid,
            "best_threshold": best_th,
            "se_pct": t3_se,
            "ratio": caa_se / t3_se,
            "per_target_alphas_at_best": t3_alpha_per_thr[best_th],
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="DISP ablation pipeline (analytical).")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=["CAA"],
        help="Steering methods to process (default: CAA).")
    parser.add_argument(
        "--models", nargs="+", choices=MODELS, default=MODELS,
        help="Models to process (default: all 5).")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON filename (default: results_<methods>.json).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Configuration:")
    print(f"  models     : {args.models}")
    print(f"  methods    : {args.methods}")
    print(f"  self floor : {SELF_FLOOR}")
    print(f"  alpha grid : 0.0 -> 1.5 step 0.05 ({len(ALPHA_GRID)} values)")
    print(f"  threshold sweep : {THRESHOLDS}")
    print(f"  lambda sweep    : {LAMBDAS}")
    print(f"  alpha search    : greedy per-direction (any k)\n")

    all_results = {}
    for model in args.models:
        print(f"\n{'='*72}\n{model}\n{'='*72}")
        all_results[model] = {}
        for method in args.methods:
            all_results[model][method] = run_model_method(model, method)

    out_name = args.out or f"results_{'_'.join(args.methods)}.json"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Console summary
    print("\n\n" + "=" * 110)
    print(" MAIN TABLE  (SE% = off-diag |pct| sum; ratio = baseline/DISP)")
    print("=" * 110)
    print(f"{'Model':<14} {'Method':<5} {'Base':>8} | "
          f"{'T1':>8} {'r':>6} | {'T2':>8} {'r':>6} | {'T3':>8} {'r':>6}")
    print("-" * 110)
    for m in args.models:
        for me in args.methods:
            r = all_results[m][me]
            print(f"{m:<14} {me:<5} {r['baseline_se_pct']:>8.2f} | "
                  f"{r['task1']['se_pct']:>8.2f} "
                  f"{r['task1']['ratio']:>5.2f}x | "
                  f"{r['task2']['se_pct']:>8.2f} "
                  f"{r['task2']['ratio']:>5.2f}x | "
                  f"{r['task3']['se_pct']:>8.2f} "
                  f"{r['task3']['ratio']:>5.2f}x")
    print("=" * 110)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
