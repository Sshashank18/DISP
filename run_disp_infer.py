"""
DISP inference pipeline.

Computes DISP-corrected steering vectors analytically, saves them, then
runs REAL forward-pass evaluation through the existing Final/src/ pipeline
to obtain a true post-DISP steered matrix per (model, task).

Tasks (each picks best hyperparam from grid):
  T1  lambda grid by REAL inference  (alpha_k = 1.0 fixed, threshold = 0.90):
      every lambda is scored by an actual steered forward pass; the lambda with
      the least off-diagonal side-effect wins (no analytical surrogate).
  T2  + per-direction alpha   (lambda* from T1, threshold = 0.90)
  T3  + threshold grid        (lambda* from T1, alpha grid, threshold*)

Usage:
  python run_disp_infer.py --methods CAA --models llama-3.1-8b
  python run_disp_infer.py --methods CAA          # all 5 models
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch as t
from tqdm import tqdm

# Make sibling Final/ importable
HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.dirname(HERE)
sys.path.insert(0, FINAL)

import config as fcfg                 # noqa: E402  (Final/config.py)
from src import data_utils as du      # noqa: E402
from src import steering              # noqa: E402
from src.model_wrapper import ModelWrapper  # noqa: E402

# Reuse analytical functions
sys.path.insert(0, HERE)
import run_disp as disp               # noqa: E402

OUT_DIR = HERE
VEC_OUT = os.path.join(OUT_DIR, "vectors")
RES_OUT = os.path.join(OUT_DIR, "inference_results.json")
REAL_OUT = os.path.join(OUT_DIR, "real_sweep_results.json")

# --- Real-inference per-direction greedy config ---
# Every alpha here is evaluated by an actual forward pass (no surrogate).
REAL_ALPHA_GRID = np.round(np.arange(0.0, 1.51, 0.1), 3).tolist()  # 16 values
REAL_SELF_FLOOR = 0.85   # keep >= 85% of each behavior's baseline self-effect


# ---------------------------------------------------------------------------
# Build corrected vector for one (target, lambda, threshold, alpha_strategy)
# ---------------------------------------------------------------------------

def build_corrected_vector(target_idx, pct, V, sign, lambda_val,
                           energy_threshold, alpha_strategy):
    """
    Return (corrected_vec as np.ndarray, alphas used, k used).
    Mirrors disp.disp_one but constructs the actual corrected vector
    (rescaled to original norm) rather than only the estimated pct row.
    Includes sign-correction in the linear-response estimator so the
    hyperparameter selection matches what real-inference will measure.
    """
    v = V[target_idx].copy()
    orig_norm = np.linalg.norm(v)
    orig_self = abs(pct[target_idx, target_idx])

    w = disp.bidir_weights(pct, target_idx, lambda_val)
    mask = w > 1e-10
    if not mask.any():
        return v, [], 0

    I_mat = (w[mask, None] * V[mask])
    _, S, Vh = np.linalg.svd(I_mat, full_matrices=False)
    energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    k = int((energy < energy_threshold).sum()) + 1
    k = max(1, min(k, Vh.shape[0]))

    top = Vh[:k]
    top = top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-12)
    proj_v = top @ v
    proj_v_sq = proj_v ** 2
    vec_norms_sq = (V * V).sum(axis=1)
    base_overlap = (V @ v) / (vec_norms_sq + 1e-12)
    dir_overlap = (top @ V.T) / (vec_norms_sq + 1e-12)
    pct_target = pct[target_idx]
    sign_prod = sign[target_idx] * sign                  # (N,)

    def estimated_rows(alpha_arr):
        norm_sq = (orig_norm ** 2
                   - ((2 * alpha_arr - alpha_arr ** 2)
                      * proj_v_sq[None, :]).sum(axis=1))
        bad = norm_sq < 1e-12
        norm_sq = np.where(bad, 1.0, norm_sq)
        scale_ = orig_norm / (np.sqrt(norm_sq) + 1e-12)
        weighted_alpha = alpha_arr * proj_v[None, :]
        dv0_overlap = -weighted_alpha @ dir_overlap
        c = ((scale_[:, None] - 1.0) * base_overlap[None, :]
             + scale_[:, None] * dv0_overlap)
        c = c * sign_prod[None, :]                      # sign correction
        est = pct_target[None, :] + c @ pct
        if bad.any():
            est[bad] = np.nan
        return est

    if alpha_strategy == "fixed":
        alphas = np.ones(k)
    else:
        # greedy per-direction
        g = np.array(disp.ALPHA_GRID, dtype=np.float64)
        G = len(g)
        alphas = np.zeros(k)
        for d_idx in range(k):
            cand = np.tile(alphas[None, :], (G, 1))
            cand[:, d_idx] = g
            est = estimated_rows(cand)
            valid = ~np.isnan(est).any(axis=1)
            if orig_self > 1e-12:
                self_after = np.abs(est[:, target_idx])
                valid &= (self_after >= disp.SELF_FLOOR * orig_self)
            sides = np.abs(est).sum(axis=1) - np.abs(est[:, target_idx])
            sides_valid = np.where(valid, sides, np.inf)
            alphas[d_idx] = float(g[int(np.argmin(sides_valid))])

    # Build the actual corrected vector (then rescale to original norm)
    v_corr = v.copy()
    for d_idx in range(k):
        v_corr -= alphas[d_idx] * proj_v[d_idx] * top[d_idx]
    n_corr = np.linalg.norm(v_corr)
    if n_corr > 1e-12:
        v_corr = v_corr * (orig_norm / n_corr)
    return v_corr, alphas.tolist(), int(k)


# ---------------------------------------------------------------------------
# Select hyperparameters via the analytical pipeline + build/save vectors
# ---------------------------------------------------------------------------

def prepare_task_vectors(model_key, method):
    """
    Returns dict: { 'T1': {...}, 'T2': {...}, 'T3': {...} } each with
      'best_lambda', 'threshold', 'vectors' (list of np arrays in behavior order),
      'alphas_per_target', 'k_per_target'
    Vectors are also written to disk under VEC_OUT/<model>/<method>/<task>/.
    """
    behaviors, pct, V, sign = disp.load_data(model_key, method)
    N = len(behaviors)

    # ------- T1: lambda grid, alpha=1 fixed, threshold=0.90 -------
    t1_grid = {}
    for lam in disp.LAMBDAS:
        se, _, _ = disp.run_config(behaviors, pct, V, sign, lam,
                                   disp.DEFAULT_THRESHOLD, "fixed")
        t1_grid[lam] = se
    best_lam = min(t1_grid, key=t1_grid.get)

    def build_for(lambda_val, threshold, alpha_strategy):
        vecs, alphas_list, k_list = [], [], []
        for i in range(N):
            v_corr, a, k = build_corrected_vector(
                i, pct, V, sign, lambda_val, threshold, alpha_strategy)
            vecs.append(v_corr)
            alphas_list.append(a)
            k_list.append(k)
        return vecs, alphas_list, k_list

    t1_vecs, t1_alpha, t1_k = build_for(best_lam,
                                        disp.DEFAULT_THRESHOLD, "fixed")

    # ------- T2: best lambda, alpha grid, threshold=0.90 -------
    t2_vecs, t2_alpha, t2_k = build_for(best_lam,
                                        disp.DEFAULT_THRESHOLD, "grid")

    # ------- T3: best lambda, alpha grid, threshold grid -------
    t3_grid = {}
    for th in disp.THRESHOLDS:
        se, _, _ = disp.run_config(behaviors, pct, V, sign,
                                   best_lam, th, "grid")
        t3_grid[th] = se
    best_th = min(t3_grid, key=t3_grid.get)
    t3_vecs, t3_alpha, t3_k = build_for(best_lam, best_th, "grid")

    out = {
        "behaviors": behaviors,
        "T1": {"best_lambda": best_lam,
               "threshold": disp.DEFAULT_THRESHOLD,
               "vectors": t1_vecs,
               "alphas_per_target": t1_alpha,
               "k_per_target": t1_k},
        "T2": {"best_lambda": best_lam,
               "threshold": disp.DEFAULT_THRESHOLD,
               "vectors": t2_vecs,
               "alphas_per_target": t2_alpha,
               "k_per_target": t2_k},
        "T3": {"best_lambda": best_lam,
               "threshold": best_th,
               "vectors": t3_vecs,
               "alphas_per_target": t3_alpha,
               "k_per_target": t3_k},
    }

    # Persist vectors (one .pt per (model, method, task, behavior))
    for task in ("T1", "T2", "T3"):
        out_dir = os.path.join(VEC_OUT, model_key, method, task)
        os.makedirs(out_dir, exist_ok=True)
        for b, vec in zip(behaviors, out[task]["vectors"]):
            path = os.path.join(out_dir, f"{b}.pt")
            t.save({"vector": t.from_numpy(vec.astype(np.float32)),
                    "task": task,
                    "lambda": out[task]["best_lambda"],
                    "threshold": out[task]["threshold"]}, path)
    return out


# ---------------------------------------------------------------------------
# Real inference with corrected vectors
# ---------------------------------------------------------------------------

def _select_test_items(method, behavior):
    # All methods evaluate on the identical 50-item slice (eval[50:100]) to
    # match the corrected Final/ baselines (CAA/ACE no longer use the full
    # 100-item caa_test). Keeps DISP-infer consistent with the 50-pt
    # baseline_delta_report it reads.
    return du.dim_test(behavior)


def real_steered_matrix(model, model_key, method, behaviors, task_vectors):
    """Run baseline (if needed) + 16x16 steered using the supplied vectors."""
    info = fcfg.MODELS[model_key]
    target_layer = info["target_layer"]
    a_id, b_id = du.get_ab_token_ids(model.tokenizer)

    # Baseline: reuse from existing CAA results if available
    base_path = os.path.join(FINAL, "outputs", model_key, method,
                             "baseline_delta_report.json")
    with open(base_path) as f:
        base_rpt = json.load(f)
    baseline = base_rpt["baseline"]
    print(f"  loaded existing baseline ({method}) for {model_key}")

    # ACE refs (only used for ACE; left None for CAA)
    n = len(behaviors)
    steered = np.zeros((n, n), dtype=float)
    for i, steer_b in enumerate(behaviors):
        vec = task_vectors[i]
        if not isinstance(vec, t.Tensor):
            vec = t.from_numpy(np.asarray(vec)).float()
        ace_ref = None
        if method == "ACE":
            ace_ref = t.load(fcfg.ace_ref_path(model_key, steer_b),
                             map_location="cpu")
        layer = target_layer  # for CAA / ACE (DIM uses its own per-vector
                              # layer; not used in this CAA-only iteration)
        for j, eval_b in enumerate(tqdm(
            behaviors, desc=f"  steer={steer_b}", leave=False
        )):
            model.reset()
            steering.apply(model, method, vec, layer, steer_b, ace_ref=ace_ref)
            items = _select_test_items(method, eval_b)
            steered[i, j] = du.mean_matching_prob(model, items, a_id, b_id)
            model.reset()

    base_arr = np.array([baseline[b] for b in behaviors])
    delta = steered - base_arr[None, :]
    # percentage change: ((steered - baseline) / baseline) * 100
    safe = np.where(np.abs(base_arr) < 1e-12, 1.0, base_arr)
    pct = (delta / safe[None, :]) * 100.0
    return baseline, steered, delta, pct


def total_se_pct(pct_matrix):
    n = pct_matrix.shape[0]
    return float(np.abs(pct_matrix - np.diag(np.diag(pct_matrix))).sum())


def self_pct(pct_matrix):
    """On-target effect = sum of |diagonal| of the pct matrix."""
    return float(np.abs(np.diag(np.asarray(pct_matrix))).sum())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Real-inference per-direction greedy alpha selection
# ---------------------------------------------------------------------------
# Replaces the surrogate (linear-response) alpha selection used by T1/T2/T3.
# Every alpha is judged by an ACTUAL forward pass. For each target behavior we
# walk the SVD interference directions one at a time (strongest first); within
# a direction we raise alpha over REAL_ALPHA_GRID and accept while the target's
# real self-effect stays >= REAL_SELF_FLOOR of its baseline AND does not drop
# vs the previous accepted alpha. At the first violation we stop and keep the
# previous alpha, then move to the next direction. The SVD energy threshold is
# itself searched by inference: we run the greedy at every threshold and keep,
# per target, the corrected vector that minimises real side-effect subject to
# the self floor. Only the [i,i] self cell is evaluated during the sweep (the
# stop-rule reads nothing else); the full 16-wide row is measured once when a
# target's alphas are locked, and the rows are assembled into the final matrix.


def _svd_dirs(target_idx, V, pct, lambda_val, threshold):
    """SVD interference directions for one target at a given energy threshold."""
    v = V[target_idx].copy()
    orig_norm = float(np.linalg.norm(v))
    w = disp.bidir_weights(pct, target_idx, lambda_val)
    mask = w > 1e-10
    if not mask.any():
        return v, None, None, 0, orig_norm
    I_mat = w[mask, None] * V[mask]
    _, S, Vh = np.linalg.svd(I_mat, full_matrices=False)
    energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    k = int((energy < threshold).sum()) + 1
    k = max(1, min(k, Vh.shape[0]))
    top = Vh[:k]
    top = top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-12)
    proj_v = top @ v
    return v, top, proj_v, int(k), orig_norm


def _build_vec(v, top, proj_v, alphas, orig_norm):
    """v - alpha * sum_d (v.top_d) top_d  (no norm rescaling)."""
    if top is None or len(alphas) == 0:
        return v.copy()
    a = np.asarray(alphas, dtype=np.float64)
    return v - (a[:, None] * proj_v[:, None] * top).sum(axis=0)


def _cell_prob(model, method, vec, layer, steer_b, eval_b, a_id, b_id):
    """One steered evaluation: steer with `vec` for steer_b, score eval_b."""
    vt = (vec if isinstance(vec, t.Tensor)
          else t.from_numpy(np.asarray(vec, dtype=np.float32)))
    model.reset()
    steering.apply(model, method, vt, layer, steer_b)
    prob = du.mean_matching_prob(
        model, _select_test_items(method, eval_b), a_id, b_id)
    model.reset()
    return prob


def _single_alpha_search(model, method, behaviors, target_idx, v, top, proj_v, k,
                         orig_norm, baseline, base_self_pct, layer, a_id, b_id):
    """Single scalar alpha applied uniformly to all k SVD directions.

    Sweeps REAL_ALPHA_GRID; for each alpha builds v' = v - alpha*sum_d(proj_v[d]*top[d])
    rescaled to original norm, then checks the real self-effect via a forward pass.
    Picks the largest alpha where self >= REAL_SELF_FLOOR * |base_self_pct|.
    Returns (alphas_array, n_evals, trace).
    """
    steer_b = behaviors[target_idx]
    base_prob = baseline[steer_b]
    ref_self = abs(base_self_pct) + 1e-12
    n_evals = 0
    trace = []
    feasible = []   # (alpha, self_pct, retention) where floor is satisfied

    for alpha in REAL_ALPHA_GRID:
        if alpha == 0.0:
            continue
        vec = _build_vec(v, top, proj_v, np.full(k, alpha), orig_norm)
        prob = _cell_prob(model, method, vec, layer, steer_b, steer_b,
                          a_id, b_id)
        n_evals += 1
        self_p = (prob - base_prob) / base_prob * 100.0 if base_prob else 0.0
        ret = abs(self_p) / ref_self
        event = "accept" if ret >= REAL_SELF_FLOOR else "floor_fail"
        trace.append({"alpha": float(alpha), "self_pct": float(self_p),
                      "retention": float(ret), "event": event})
        if ret >= REAL_SELF_FLOOR:
            feasible.append(alpha)

    # Largest feasible alpha = most aggressive correction that keeps self alive.
    # Fall back to 0.0 (no correction) if nothing satisfies the floor.
    best_alpha = max(feasible) if feasible else 0.0
    return np.full(k, best_alpha), n_evals, trace


def _full_row(model, method, vec, layer, steer_b, behaviors, baseline,
              a_id, b_id):
    """16-wide pct-change row for steering `steer_b` with `vec`."""
    out = []
    for eb in behaviors:
        p = _cell_prob(model, method, vec, layer, steer_b, eb, a_id, b_id)
        bp = baseline[eb]
        out.append((p - bp) / (bp if bp else 1.0) * 100.0)
    return np.array(out)


def _mat_summary(mat, base_se, base_self_total, per_target):
    side = total_se_pct(mat)
    slf = self_pct(mat)
    return {
        "side_pct": side,
        "self_pct": slf,
        "ratio": base_se / side if side > 0 else float("inf"),
        "self_retained": slf / base_self_total if base_self_total else 0.0,
        "self_over_side": slf / side if side > 0 else 0.0,
        "steered_pct_matrix": mat.tolist(),
        "per_target": per_target,
    }


def _infer_t1_lambda(model, method, behaviors, pct, V, baseline, layer,
                     a_id, b_id, DEF):
    """T1 lambda selection by REAL inference (replaces the analytical surrogate).

    For each lambda in disp.LAMBDAS, build the alpha=1 / threshold=DEF corrected
    vectors for every behavior, measure the full steered pct-change matrix, and
    score it by total off-diagonal side-effect. Returns (best_lambda,
    {lambda: side_pct}, best_matrix) -- the winning matrix IS the T1 matrix, so
    no T1 forward passes are repeated downstream.
    """
    N = len(behaviors)
    side_by_lam, mat_by_lam = {}, {}
    for lam in disp.LAMBDAS:
        mat = np.zeros((N, N))
        for i, steer_b in enumerate(behaviors):
            v, top, proj_v, k0, onorm = _svd_dirs(i, V, pct, lam, DEF)
            vec1 = _build_vec(v, top, proj_v, np.ones(k0), onorm)
            mat[i] = _full_row(model, method, vec1, layer, steer_b, behaviors,
                               baseline, a_id, b_id)
        side_by_lam[lam] = total_se_pct(mat)
        mat_by_lam[lam] = mat
        print(f"    [T1 infer] lambda={lam}: SIDE={side_by_lam[lam]:.1f}",
              flush=True)
    best_lam = min(side_by_lam, key=side_by_lam.get)
    print(f"  lambda (T1 inference) = {best_lam}  "
          f"(side={side_by_lam[best_lam]:.1f})")
    return best_lam, side_by_lam, mat_by_lam[best_lam]


def run_model_method(model_key, method, results, out_path):
    print(f"\n========== {model_key} / {method}  "
          f"[real inference; self-floor {REAL_SELF_FLOOR}] ==========")
    behaviors, pct, V, sign = disp.load_data(model_key, method)
    N = len(behaviors)
    DEF = disp.DEFAULT_THRESHOLD   # 0.90 -- the fixed threshold for T1 and T2

    base_path = os.path.join(FINAL, "outputs", model_key, method,
                             "baseline_delta_report.json")
    with open(base_path) as f:
        base_rpt = json.load(f)
    baseline = base_rpt["baseline"]
    base_pct = np.array(base_rpt["pct_change_matrix"], dtype=np.float64)
    base_self_diag = np.abs(np.diag(base_pct))          # per-target baseline self
    base_se = total_se_pct(base_pct)
    base_self_total = self_pct(base_pct)
    print(f"  baseline: SIDE={base_se:.2f}  SELF={base_self_total:.2f}")

    # Model is loaded BEFORE lambda is fixed: T1's lambda is now chosen by real
    # inference (forward passes), not by the analytical linear-response surrogate.
    info = fcfg.MODELS[model_key]
    layer = info["target_layer"]
    print(f"  Loading model {info['hf_id']} ...")
    model = ModelWrapper(
        hf_id=info["hf_id"],
        family=info["family"],
        hf_token=fcfg.hf_token(),
        dtype=t.bfloat16,
        device_map="auto",
    )
    a_id, b_id = du.get_ab_token_ids(model.tokenizer)

    # ---- T1: pick lambda by REAL inference; the winning matrix becomes T1 ----
    best_lam, lam_side, matT1 = _infer_t1_lambda(
        model, method, behaviors, pct, V, baseline, layer, a_id, b_id, DEF)

    matT2 = np.zeros((N, N))
    matT3 = np.zeros((N, N))
    ptT1, ptT2, ptT3 = [], [], []
    sweep = {}          # full trajectory: per behavior, every alpha/threshold tried
    total_evals = len(disp.LAMBDAS) * N * N   # forward passes spent on T1 selection

    def _record(row, i, tau, alphas, k):
        self_p = abs(row[i])
        side_i = float(np.abs(row).sum() - abs(row[i]))
        ret = self_p / (base_self_diag[i] + 1e-12)
        return {"behavior": behaviors[i], "threshold": tau,
                "alphas": list(alphas), "k": int(k), "self_pct": self_p,
                "side_pct": side_i, "self_retained": ret,
                "baseline_self_pct": float(base_pct[i, i]),
                "baseline_side_pct": float(np.abs(base_pct[i]).sum()
                                           - abs(base_pct[i, i]))}

    for i in range(N):
        steer_b = behaviors[i]

        # ---- T1: full projection (alpha = 1) at threshold 0.90 ----
        # Row already measured during the inference lambda sweep (matT1); only
        # the SVD rank k0 is recovered here (analytical, no forward pass).
        _, _, _, k0, _ = _svd_dirs(i, V, pct, best_lam, DEF)
        row1 = matT1[i]
        r1 = _record(row1, i, DEF, [1.0] * k0, k0)
        ptT1.append(r1)

        # ---- T2/T3: single uniform alpha; T2 = tau 0.90, T3 = best tau ----
        cands = []
        for tau in disp.THRESHOLDS:
            v, top, proj_v, k, onorm = _svd_dirs(i, V, pct, best_lam, tau)
            alphas, ne, trace = _single_alpha_search(
                model, method, behaviors, i, v, top, proj_v, k, onorm,
                baseline, base_pct[i, i], layer, a_id, b_id)
            total_evals += ne
            vec = _build_vec(v, top, proj_v, alphas, onorm)
            row = _full_row(model, method, vec, layer, steer_b, behaviors,
                            baseline, a_id, b_id)
            total_evals += N
            rec = _record(row, i, tau, alphas.tolist(), k)
            rec["row"] = row
            rec["greedy_trace"] = trace
            cands.append(rec)

        t2 = next(c for c in cands if abs(c["threshold"] - DEF) < 1e-9)
        feasible = [c for c in cands
                    if c["self_retained"] >= REAL_SELF_FLOOR] or cands
        t3 = min(feasible, key=lambda c: c["side_pct"])
        matT2[i] = t2["row"]
        matT3[i] = t3["row"]
        drop = ("row",)   # keep greedy_trace in per_target; drop only the raw row
        ptT2.append({kk: vv for kk, vv in t2.items() if kk not in drop})
        ptT3.append({kk: vv for kk, vv in t3.items() if kk not in drop})

        # full trajectory for this behavior: T1 + every threshold candidate
        sweep[steer_b] = {
            "T1": {"threshold": DEF, "alphas": [1.0] * k0, "k": k0,
                   "self_pct": r1["self_pct"], "side_pct": r1["side_pct"],
                   "self_retained": r1["self_retained"]},
            "threshold_candidates": [
                {**{kk: vv for kk, vv in c.items() if kk != "row"},
                 "is_T2": abs(c["threshold"] - DEF) < 1e-9,
                 "is_T3": c is t3}
                for c in cands],
        }

        print(f"  [{i+1:2d}/{N}] {steer_b:<26} "
              f"T1 self{r1['self_pct']:.0f}({100*r1['self_retained']:.0f}%)"
              f"/side{r1['side_pct']:.0f} | "
              f"T2 self{t2['self_pct']:.0f}({100*t2['self_retained']:.0f}%)"
              f"/side{t2['side_pct']:.0f} | "
              f"T3 tau{t3['threshold']:.2f} "
              f"self{t3['self_pct']:.0f}({100*t3['self_retained']:.0f}%)"
              f"/side{t3['side_pct']:.0f}")

    results.setdefault(model_key, {})[method] = {
        "lambda": best_lam,
        "lambda_selection": "inference",
        "lambda_side_by_inference": {str(k): v for k, v in lam_side.items()},
        "self_floor": REAL_SELF_FLOOR,
        "alpha_grid": REAL_ALPHA_GRID,
        "default_threshold": DEF,
        "baseline_side_pct": base_se,
        "baseline_self_pct": base_self_total,
        "T1": _mat_summary(matT1, base_se, base_self_total, ptT1),
        "T2": _mat_summary(matT2, base_se, base_self_total, ptT2),
        "T3": _mat_summary(matT3, base_se, base_self_total, ptT3),
        "sweep": sweep,
        "total_evals": total_evals,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for tag in ("T1", "T2", "T3"):
        s = results[model_key][method][tag]
        print(f"  {tag}: SIDE {base_se:.1f}->{s['side_pct']:.1f} "
              f"({s['ratio']:.2f}x)  SELF {base_self_total:.1f}->"
              f"{s['self_pct']:.1f} ({100*s['self_retained']:.0f}%)  "
              f"self/side={s['self_over_side']:.3f}")
    print(f"  evals={total_evals}")

    del model
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        choices=disp.METHODS, default=["CAA"])
    parser.add_argument("--models", nargs="+",
                        choices=disp.MODELS, default=disp.MODELS)
    args = parser.parse_args()

    # Per-run output file keyed by the model(s) so parallel runs of different
    # models never clobber a shared results file.
    out_path = os.path.join(
        OUT_DIR, f"real_sweep_{'_'.join(args.models)}.json")

    # load existing results if any (resume)
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
    else:
        results = {}

    for model_key in args.models:
        for method in args.methods:
            try:
                run_model_method(model_key, method, results, out_path)
            except Exception as e:
                print(f"  !! FAILED {model_key}/{method}: {e!r}")
                import traceback
                traceback.print_exc()
                results.setdefault(model_key, {})[method] = {"error": str(e)}
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    # Summary
    print("\n\n" + "=" * 110)
    print(" REAL-INFERENCE SUMMARY  (self-floor "
          f"{REAL_SELF_FLOOR}; ratio = baseSIDE/SIDE; self% = self retained)")
    print(" T1 = full proj a=1 @tau0.90 | T2 = greedy a @tau0.90 | "
          "T3 = greedy a, best tau")
    print("=" * 110)
    print(f"{'Model':<14}{'Meth':<5}{'baseSIDE':>9} | "
          f"{'T1 SIDE':>9}{'r':>6}{'self%':>6} | "
          f"{'T2 SIDE':>9}{'r':>6}{'self%':>6} | "
          f"{'T3 SIDE':>9}{'r':>6}{'self%':>6}")
    print("-" * 110)
    for m in args.models:
        for me in args.methods:
            r = results.get(m, {}).get(me)
            if not r or "error" in r:
                continue
            cells = ""
            for tag in ("T1", "T2", "T3"):
                s = r[tag]
                cells += (f"{s['side_pct']:>9.1f}{s['ratio']:>5.2f}x"
                          f"{100*s['self_retained']:>5.0f}% | ")
            print(f"{m:<14}{me:<5}{r['baseline_side_pct']:>9.1f} | {cells}")
    print("=" * 110)
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
