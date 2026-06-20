"""
Baselines for steering-vector correction & conditional steering  (six tasks)
============================================================================

Generalised over MODEL and METHOD via the CLI (--model, --method, --task).
Methods are applied *method-natively*:
    CAA  : ActAdd  h += sign*alpha*v               at the target layer
    DIM  : suppress(decrease) -> ablate  h -= alpha*(h.d)d   at ALL layers
           boost  (increase) -> ActAdd  h += alpha*v          at target layer
    ACE  : ablate+affine  h -= (h.d)d + (ref.d)d  and nudge h += sign*alpha*v
           at the target layer (loads *_ref.pt)

Every task produces a 16x16 *steering-effect* matrix. Entry =
    pct[i, j] = (steered[i, j] - baseline[j]) / baseline[j] * 100
(row = steer behaviour i, col = eval behaviour j). Saved JSON also includes the
signed steering_effect_matrix (pct * desired-direction sign), baseline, and
SIDE/SELF totals. Output: Baselines/outputs/<model>/<method>/<task>.json

Tasks
-----
  T1  Method strength sweep at 5 strengths in [0.5, 1.5]   -> 5 matrices.
        (alpha scales: ActAdd magnitude for CAA/DIM-boost/ACE-nudge,
         ablation fraction for DIM-suppress.)
  T2  Correction by PCA-projection  (centered SVD of the 15 other vectors).
  T3  Correction by Gram-Schmidt    (orthogonalise vs the top-k other vectors).
  T4  Correction by unscaled SVD    (plain, uncentered, unweighted SVD).
  T5  CAST: method gated by a learned condition vector + F1-tuned threshold.
  T6  CAST gating applied to the DISP-best (T3 real-inference) corrected
      vectors, rebuilt from ../real_sweep_<model>.json for THIS method.
      Skipped with a warning when no sweep is available for (model, method).

Decisions baked in (per user):
  * T2-T4 correct each target against ALL 15 other vectors, unweighted.
  * rank for PCA/SVD/GS = top components up to 0.90 cumulative energy.
  * corrected vectors (T2-T4) and CAST vectors applied at alpha = 1.0.
  * "unscaled SVD" = plain SVD, no interference weighting, no norm rescaling.
  * CAST condition vectors from data/generate/*; threshold by F1 grid search.
  * CAST is single-layer conditional steering at the target layer for every
    method (so DIM-suppress under CAST is single-layer gated ablation, a
    documented departure from its native all-layer ablation).
  * output: JSON only.

Usage
-----
  python run_baselines.py --model llama-3.1-8b --method CAA --task all
  python run_baselines.py --model qwen3-4b    --method DIM --task 1,2,5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch as t

# --- make Final/ importable (config, src.*) ---------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))           # .../Final/DISP/Baselines
DISP_DIR = os.path.dirname(HERE)                            # .../Final/DISP
FINAL = os.path.dirname(DISP_DIR)                           # .../Final
STEERING_ROOT = os.path.dirname(FINAL)                     # .../STEERING
for p in (FINAL, DISP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config as cfg                                         # noqa: E402
import src.data_utils as du                                 # noqa: E402
from src.model_wrapper import ModelWrapper                  # noqa: E402

ENERGY_THRESHOLD = 0.90        # cumulative-energy cutoff for PCA / SVD / GS rank
T1_STRENGTHS = [0.5, 0.75, 1.0, 1.25, 1.5]
CAST_N_POS = 100               # condition-positive samples per behavior
CAST_N_NEG = 100               # condition-negative samples (pooled from others)


# ===========================================================================
# Data loading (per model, method)
# ===========================================================================
def load_method_data(model_key, method):
    """Return behaviors, pct matrix, V (N,d numpy), and ace_refs (dict|None)."""
    rpt_path = os.path.join(FINAL, "outputs", model_key, method,
                            "baseline_delta_report.json")
    with open(rpt_path) as f:
        rpt = json.load(f)
    behaviors = rpt["behaviors"]
    pct = np.array(rpt["pct_change_matrix"], dtype=np.float64)
    vecs = []
    for b in behaviors:
        d = t.load(cfg.vector_path(model_key, method, b),
                   weights_only=False, map_location="cpu")
        vecs.append(d["vector"].float().numpy().astype(np.float64))
    V = np.stack(vecs, axis=0)
    ace_refs = None
    if method == "ACE":
        ace_refs = {}
        for b in behaviors:
            r = t.load(cfg.ace_ref_path(model_key, b),
                       weights_only=False, map_location="cpu")
            rv = r["vector"] if isinstance(r, dict) and "vector" in r else r
            ace_refs[b] = rv.float().numpy().astype(np.float64)
    return behaviors, pct, V, ace_refs


def _sign(behavior):
    return +1.0 if cfg.DESIRED_DIRECTION[behavior] == "increase" else -1.0


# ===========================================================================
# Correction methods (T2, T3, T4) -- method-agnostic vector corrections.
# Each corrects target i against the 15 OTHER vectors, unweighted, no rescale.
# ===========================================================================
def _energy_rank(singular_values, threshold=ENERGY_THRESHOLD):
    s2 = singular_values ** 2
    energy = np.cumsum(s2) / (s2.sum() + 1e-12)
    return max(1, min(int((energy < threshold).sum()) + 1, len(singular_values)))


def correct_pca(behaviors, V):
    """PCA-projection: SVD of the *centered* 15-other-vector stack; remove the
    target's projection onto the top-k principal components (0.90 energy)."""
    out, info = {}, {}
    for i, b in enumerate(behaviors):
        others = np.delete(V, i, axis=0)
        centered = others - others.mean(axis=0, keepdims=True)
        _, S, Vh = np.linalg.svd(centered, full_matrices=False)
        k = _energy_rank(S)
        pcs = Vh[:k]
        pcs = pcs / (np.linalg.norm(pcs, axis=1, keepdims=True) + 1e-12)
        v = V[i].copy()
        out[b] = v - ((pcs @ v)[:, None] * pcs).sum(axis=0)
        info[b] = int(k)
    return out, info


def correct_unscaled_svd(behaviors, V):
    """Plain SVD of the uncentered, unweighted 15-other-vector stack; remove the
    target's projection onto the top-k right-singular directions (0.90 energy)."""
    out, info = {}, {}
    for i, b in enumerate(behaviors):
        others = np.delete(V, i, axis=0)
        _, S, Vh = np.linalg.svd(others, full_matrices=False)
        k = _energy_rank(S)
        dirs = Vh[:k]
        dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        v = V[i].copy()
        out[b] = v - ((dirs @ v)[:, None] * dirs).sum(axis=0)
        info[b] = int(k)
    return out, info


def correct_gram_schmidt(behaviors, V):
    """Gram-Schmidt: pick the top-k OTHER vectors most aligned with the target
    (k = same energy-0.90 count as the SVD task), orthonormalise them with
    Gram-Schmidt, and remove the target's projection onto that basis."""
    out, info = {}, {}
    for i, b in enumerate(behaviors):
        others = np.delete(V, i, axis=0)
        _, S, _ = np.linalg.svd(others, full_matrices=False)
        k = _energy_rank(S)
        v = V[i].copy()
        onorm = np.linalg.norm(others, axis=1) + 1e-12
        cos = np.abs(others @ v) / (onorm * (np.linalg.norm(v) + 1e-12))
        order = np.argsort(-cos)[:k]
        basis = []
        for idx in order:
            u = others[idx].copy()
            for q in basis:
                u = u - (u @ q) * q
            nrm = np.linalg.norm(u)
            if nrm > 1e-8:
                basis.append(u / nrm)
        v_corr = v.copy()
        for q in basis:
            v_corr = v_corr - (v_corr @ q) * q
        out[b] = v_corr
        info[b] = len(basis)
    return out, info


# ===========================================================================
# DISP T3 vector reconstruction (for T6) from ../real_sweep_<model>.json
# ===========================================================================
def _bidir_weights(pct, target_idx, lam):
    row = np.abs(pct[target_idx, :]); col = np.abs(pct[:, target_idx])
    rmax = row.max() if row.max() > 0 else 1.0
    cmax = col.max() if col.max() > 0 else 1.0
    w = np.zeros(pct.shape[0])
    for j in range(pct.shape[0]):
        if j == target_idx:
            continue
        w[j] = (lam * abs(pct[target_idx, j]) / rmax
                + (1 - lam) * abs(pct[j, target_idx]) / cmax)
    return w


def _svd_dirs(target_idx, V, pct, lam, threshold):
    v = V[target_idx].copy()
    w = _bidir_weights(pct, target_idx, lam)
    mask = w > 1e-10
    if not mask.any():
        return v, None, None, 0
    I_mat = w[mask, None] * V[mask]
    _, S, Vh = np.linalg.svd(I_mat, full_matrices=False)
    energy = np.cumsum(S ** 2) / (np.sum(S ** 2) + 1e-12)
    k = max(1, min(int((energy < threshold).sum()) + 1, Vh.shape[0]))
    top = Vh[:k]
    top = top / (np.linalg.norm(top, axis=1, keepdims=True) + 1e-12)
    return v, top, top @ v, int(k)


def build_disp_t3_vectors(behaviors, pct, V, model_key, method):
    """Rebuild T3 (best-tau, real-inference) corrected vectors for (model,
    method) from the stored lambda + per-target (tau, alphas). Returns
    (vectors, lambda, info) or (None, None, None) if no sweep is available."""
    sweep_path = os.path.join(DISP_DIR, f"real_sweep_{model_key}.json")
    if not os.path.exists(sweep_path):
        return None, None, None
    with open(sweep_path) as f:
        sweep = json.load(f)
    if model_key not in sweep or method not in sweep[model_key]:
        return None, None, None
    rec = sweep[model_key][method]
    lam = rec["lambda"]
    t3 = {e["behavior"]: e for e in rec["T3"]["per_target"]}
    out, info = {}, {}
    for i, b in enumerate(behaviors):
        tau = t3[b]["threshold"]
        alphas = np.array(t3[b]["alphas"], dtype=np.float64)
        v, top, proj_v, k = _svd_dirs(i, V, pct, lam, tau)
        if top is None or len(alphas) == 0:
            out[b] = v
        else:
            a = alphas[:top.shape[0]]
            out[b] = v - (a[:, None] * proj_v[:, None] * top).sum(axis=0)
        info[b] = {"tau": tau, "k": int(k), "alphas": alphas.tolist()}
    return out, lam, info


# ===========================================================================
# Method-native steering via forward hooks (supports alpha + optional CAST gate)
# ===========================================================================
def _gate(hidden, cond_dir, tau):
    """Per-position 0/1 mask: 1 where cos(hidden, cond_dir) >= tau."""
    if cond_dir is None:
        return 1.0
    cd = cond_dir.to(hidden.dtype).to(hidden.device)
    hn = hidden.norm(dim=-1, keepdim=True) + 1e-8
    cos = (hidden * cd).sum(-1, keepdim=True) / (hn * (cd.norm() + 1e-8))
    return (cos >= tau).to(hidden.dtype)


def _add_hook(add_vec, cond_dir, tau):
    def fn(module, inp, out):
        is_t = isinstance(out, tuple); h = out[0] if is_t else out
        v = add_vec.to(h.dtype).to(h.device)
        h = h + _gate(h, cond_dir, tau) * v
        return (h,) + out[1:] if is_t else h
    return fn


def _ablate_hook(direction, alpha, cond_dir, tau):
    def fn(module, inp, out):
        is_t = isinstance(out, tuple); h = out[0] if is_t else out
        d = direction.to(h.dtype).to(h.device)
        dot = (h * d).sum(-1, keepdim=True)
        h = h + _gate(h, cond_dir, tau) * (-alpha * dot * d)
        return (h,) + out[1:] if is_t else h
    return fn


def _ace_hook(direction, ace_ref, alpha, sign, cond_dir, tau):
    def fn(module, inp, out):
        is_t = isinstance(out, tuple); h = out[0] if is_t else out
        d = direction.to(h.dtype).to(h.device)
        r = ace_ref.to(h.dtype).to(h.device)
        dot = (h * d).sum(-1, keepdim=True)
        affine = (r * d).sum()
        delta = -dot * d + affine * d + alpha * sign * d
        h = h + _gate(h, cond_dir, tau) * delta
        return (h,) + out[1:] if is_t else h
    return fn


def apply_steer(model, method, vec_np, layer, behavior, alpha,
                ace_ref_np=None, cond=None):
    """Register method-native forward hooks; return list of handles to remove.

    cond = (cond_dir_np, threshold) enables single-layer CAST gating at `layer`.
    """
    sign = _sign(behavior)
    vec = t.from_numpy(np.asarray(vec_np, dtype=np.float32))
    cond_dir = (t.from_numpy(np.asarray(cond[0], dtype=np.float32))
                if cond is not None else None)
    tau = cond[1] if cond is not None else None
    desired = cfg.DESIRED_DIRECTION[behavior]
    handles = []

    if method == "CAA":
        handles.append(model._layers[layer].register_forward_hook(
            _add_hook(sign * alpha * vec, cond_dir, tau)))

    elif method == "DIM":
        if desired == "decrease":               # suppress -> ablation
            # native = all layers; CAST = single (target) layer only.
            layers = [layer] if cond is not None else range(model.n_layers)
            for L in layers:
                cd = cond_dir if (cond is not None and L == layer) else None
                tt = tau if (cond is not None and L == layer) else None
                handles.append(model._layers[L].register_forward_hook(
                    _ablate_hook(vec, alpha, cd, tt)))
        else:                                    # boost -> ActAdd (no sign)
            handles.append(model._layers[layer].register_forward_hook(
                _add_hook(alpha * vec, cond_dir, tau)))

    elif method == "ACE":
        ref = t.from_numpy(np.asarray(ace_ref_np, dtype=np.float32))
        handles.append(model._layers[layer].register_forward_hook(
            _ace_hook(vec, ref, alpha, sign, cond_dir, tau)))

    else:
        raise ValueError(f"Unknown method {method}")
    return handles


# ===========================================================================
# Inference core
# ===========================================================================
# All tasks evaluate on the 50-item slice eval[50:100] (du.dim_test) so these
# baselines match the corrected Final/ pipeline (CAA/ACE no longer use the full
# 100-item caa_test). Baseline and steered matrices use the same 50 items.
def _baseline_probs(model, behaviors, a_id, b_id):
    base = {}
    model.reset()
    for b in behaviors:
        base[b] = du.mean_matching_prob(model, du.dim_test(b), a_id, b_id)
    return base


def _row_pct(model, behaviors, baseline, a_id, b_id):
    row = np.zeros(len(behaviors))
    for j, eb in enumerate(behaviors):
        p = du.mean_matching_prob(model, du.dim_test(eb), a_id, b_id)
        bp = baseline[eb]
        row[j] = (p - bp) / bp * 100.0 if bp else 0.0
    return row


def steered_matrix(model, behaviors, method, vectors, layer, baseline,
                   a_id, b_id, alpha=1.0, ace_refs=None, conditions=None):
    """16x16 pct matrix; steer each behaviour method-natively (optionally
    CAST-gated when `conditions` is supplied)."""
    N = len(behaviors)
    mat = np.zeros((N, N))
    for i, sb in enumerate(behaviors):
        cond = None
        if conditions is not None:
            cond = (conditions[sb]["direction"], conditions[sb]["threshold"])
        ace_ref = ace_refs[sb] if ace_refs else None
        model.reset()
        handles = apply_steer(model, method, vectors[sb], layer, sb, alpha,
                              ace_ref, cond)
        try:
            mat[i] = _row_pct(model, behaviors, baseline, a_id, b_id)
        finally:
            for h in handles:
                h.remove()
        model.reset()
    return mat


# ===========================================================================
# CAST: condition vectors + F1-tuned thresholds (method-agnostic)
# ===========================================================================
def _layer_last_token_acts(model, items, layer, n):
    acts = []
    model.set_save_activations(True)
    for it in items[:n]:
        ids = model.build_chat_tokens(user_input=it["question"],
                                      assistant_prefix="(")
        _ = model.forward_logits(ids)
        h = model.get_activations(layer)
        acts.append(h[0, -1, :].float().cpu().numpy())
    model.set_save_activations(False)
    return np.stack(acts, axis=0)


def build_cast_conditions(model, behaviors, layer):
    """Per behaviour: (condition direction, F1-optimal cosine threshold).
    Positives = own generate prompts; negatives = pooled others."""
    gen = {}
    for b in behaviors:
        with open(os.path.join(STEERING_ROOT, "data", "generate",
                               f"{b}.json")) as f:
            gen[b] = json.load(f)
    conditions = {}
    for b in behaviors:
        pos = _layer_last_token_acts(model, gen[b], layer, CAST_N_POS)
        others = [x for x in behaviors if x != b]
        per = max(1, CAST_N_NEG // len(others))
        neg_items = []
        for ob in others:
            neg_items.extend(gen[ob][:per])
        neg = _layer_last_token_acts(model, neg_items, layer, CAST_N_NEG)

        direction = pos.mean(0) - neg.mean(0)
        direction = direction / (np.linalg.norm(direction) + 1e-12)

        def cos(a):
            return (a @ direction) / (np.linalg.norm(a, axis=1) + 1e-12)
        pcos, ncos = cos(pos), cos(neg)
        cands = np.unique(np.concatenate([pcos, ncos]))
        best_tau, best_f1 = float(cands.min()), -1.0
        for tau in cands:
            tp = (pcos >= tau).sum(); fp = (ncos >= tau).sum()
            fn = (pcos < tau).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
        conditions[b] = {"direction": direction, "threshold": best_tau,
                         "f1": best_f1}
        print(f"    {b:<26} tau={best_tau:+.3f} F1={best_f1:.3f} "
              f"pos={pcos.mean():.3f} neg={ncos.mean():.3f}")
    return conditions


# ===========================================================================
# Save
# ===========================================================================
def save_matrix(out_dir, name, model_key, method, behaviors, baseline, mat,
                config):
    os.makedirs(out_dir, exist_ok=True)
    side = float(np.abs(mat - np.diag(np.diag(mat))).sum())
    self_ = float(np.abs(np.diag(mat)).sum())
    dir_sign = np.array([_sign(b) for b in behaviors])
    steering_effect = mat * dir_sign[np.newaxis, :]
    payload = {
        "model": model_key, "method": method, "task": name,
        "behaviors": behaviors,
        "baseline": [baseline[b] for b in behaviors],
        "pct_change_matrix": mat.tolist(),
        "steering_effect_matrix": steering_effect.tolist(),
        "side_pct_total": side, "self_pct_total": self_,
        "config": config,
    }
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved {path}   SIDE={side:.1f}  SELF={self_:.1f}")


# ===========================================================================
# Driver
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(cfg.MODELS.keys()))
    ap.add_argument("--method", required=True, choices=["CAA", "DIM", "ACE"])
    ap.add_argument("--task", required=True, help="comma list of 1..6 or 'all'")
    args = ap.parse_args()
    model_key, method = args.model, args.method
    tasks = ([str(i) for i in range(1, 7)] if args.task == "all"
             else [s.strip() for s in args.task.split(",")])
    out_dir = os.path.join(HERE, "outputs", model_key, method)

    behaviors, pct, V, ace_refs = load_method_data(model_key, method)
    layer = cfg.MODELS[model_key]["target_layer"]

    print(f"Loading {model_key} (method={method}) ...")
    model = ModelWrapper(
        hf_id=cfg.MODELS[model_key]["hf_id"],
        family=cfg.MODELS[model_key]["family"],
        hf_token=cfg.hf_token(), dtype=t.bfloat16, device_map="auto",
    )
    a_id, b_id = du.get_ab_token_ids(model.tokenizer)
    print("Computing unsteered baseline ...")
    baseline = _baseline_probs(model, behaviors, a_id, b_id)
    vecs = {b: V[i] for i, b in enumerate(behaviors)}

    def run_matrix(name, vectors, cfgd, alpha=1.0, conditions=None):
        mat = steered_matrix(model, behaviors, method, vectors, layer, baseline,
                             a_id, b_id, alpha=alpha, ace_refs=ace_refs,
                             conditions=conditions)
        save_matrix(out_dir, name, model_key, method, behaviors, baseline, mat,
                    cfgd)

    if "1" in tasks:
        print(f"\n== T1: {method} strength sweep ==")
        for s in T1_STRENGTHS:
            run_matrix(f"T1_{method}_alpha_{s:.2f}", vecs, {"alpha": s}, alpha=s)

    if "2" in tasks:
        print("\n== T2: PCA-projection correction ==")
        cv, info = correct_pca(behaviors, V)
        run_matrix(f"T2_{method}_PCA_projection", cv,
                   {"alpha": 1.0, "energy_threshold": ENERGY_THRESHOLD,
                    "k_per_target": info,
                    "interference_set": "all_15_unweighted_centered"})

    if "3" in tasks:
        print("\n== T3: Gram-Schmidt correction ==")
        cv, info = correct_gram_schmidt(behaviors, V)
        run_matrix(f"T3_{method}_gram_schmidt", cv,
                   {"alpha": 1.0, "energy_threshold": ENERGY_THRESHOLD,
                    "k_per_target": info,
                    "interference_set": "top_k_aligned_of_15"})

    if "4" in tasks:
        print("\n== T4: unscaled SVD correction ==")
        cv, info = correct_unscaled_svd(behaviors, V)
        run_matrix(f"T4_{method}_unscaled_svd", cv,
                   {"alpha": 1.0, "energy_threshold": ENERGY_THRESHOLD,
                    "k_per_target": info,
                    "interference_set": "all_15_unweighted_uncentered"})

    if "5" in tasks:
        print(f"\n== T5: CAST ({method} + F1 threshold) ==")
        conditions = build_cast_conditions(model, behaviors, layer)
        run_matrix(f"T5_CAST_{method}", vecs,
                   {"alpha": 1.0,
                    "thresholds": {b: conditions[b]["threshold"] for b in behaviors},
                    "f1": {b: conditions[b]["f1"] for b in behaviors}},
                   conditions=conditions)

    if "6" in tasks:
        print(f"\n== T6: CAST gating on DISP-best (T3) {method} vectors ==")
        disp_vecs, lam, dinfo = build_disp_t3_vectors(
            behaviors, pct, V, model_key, method)
        if disp_vecs is None:
            print(f"  [skip] no real_sweep_{model_key}.json with method "
                  f"{method}; run the DISP sweep first to enable T6.")
        else:
            conditions = build_cast_conditions(model, behaviors, layer)
            run_matrix(f"T6_CAST_DISP_T3_{method}", disp_vecs,
                       {"alpha": 1.0, "disp_lambda": lam, "disp_t3": dinfo,
                        "thresholds": {b: conditions[b]["threshold"]
                                       for b in behaviors}},
                       conditions=conditions)

    del model
    if t.cuda.is_available():
        t.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
