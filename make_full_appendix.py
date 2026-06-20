"""
make_full_appendix.py -- one combined LaTeX appendix for BOTH:

  (A) the DISP real-inference sweep (Final/DISP/real_sweep_<model>.json), and
  (B) the per-model baseline / correction / CAST tables
      (Final/DISP/Baselines/outputs/<model>/<method>/*.json),

for every (model, method) that is actually present on disk.

Layout (one file, \subsection per model -- as requested):

  Section "DISP sweep summary"
      * main summary table (all model x method): Base side%, T1/T2/T3 side%/self%/ratio
      * per-(model,method) threshold (tau) sweep grid, reconstructed from the
        per-target `threshold_candidates` block (uniform-tau aggregate side%,
        with the per-target-best T3 value alongside).
      NOTE: lambda is fixed at 1.0 in this data (never swept), so there is no
      lambda grid -- only the tau grid is reconstructable.

  Section "Per-model baseline / DISP / CAST detail"  (the full llama-style block)
      For each (model, method) with a complete baseline output dir:
        - behaviour-code key (printed once)
        - baseline behaviour vector
        - CAA/DIM effect matrix at 5 strengths (alpha sweep)
        - PCA / Gram-Schmidt / unscaled-SVD / CAST corrected effect matrices
        - DISP T1/T2/T3 effect matrices (signed from real_sweep)
        - DISP-best + CAST effect matrix
        - side-effect cost summary table

Output: Final/DISP/full_appendix.tex
Run:    python make_full_appendix.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # .../DISP
FINAL = os.path.dirname(HERE)                                # .../Final
BASE_OUT = os.path.join(HERE, "Baselines", "outputs")        # .../DISP/Baselines/outputs
TEX = os.path.join(HERE, "full_appendix.tex")
sys.path.insert(0, FINAL)
import config as cfg                                          # noqa: E402

BEHAVIORS = cfg.BEHAVIORS
LABELS = [cfg.BEHAVIOR_LABELS[b] for b in BEHAVIORS]
SIGN = np.array([+1.0 if cfg.DESIRED_DIRECTION[b] == "increase" else -1.0
                 for b in BEHAVIORS])

MODEL_NICE = {
    "llama-3.1-8b": "Llama-3.1-8B", "gemma-3-4b": "Gemma-3-4B",
    "gemma-3-12b": "Gemma-3-12B", "gemma-3-27b": "Gemma-3-27B",
    "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
}
# stable display order; anything else is appended alphabetically
MODEL_ORDER = ["llama-3.1-8b", "gemma-3-4b", "gemma-3-12b", "qwen3-4b", "qwen3-8b"]
METHOD_ORDER = ["CAA", "DIM", "ACE"]

SHORT = {
    "Sycophancy": "Syco", "Hallucination": "Hall", "Corrigibility": "Corr",
    "Survival Instinct": "Surv", "Myopic Reward": "Myop",
    "AI Coordination": "AI-C", "Refusal": "Ref", "Power Seeking": "Pow",
    "Wealth Seeking": "Wlth", "Self-Coordination": "S-Co",
    "Version Coordination": "V-Co", "Corrigible Less-HHH": "CrLs",
    "Corrigible More-HHH": "CrMo", "One-Box Tendency": "1Box",
    "Self-Aware (AI)": "SaAI", "Self-Aware (Text)": "SaTx",
}
TAUS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]


def nice(m):
    return MODEL_NICE.get(m, m)


def esc(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
            .replace("$", r"\$"))


def short(l):
    return SHORT.get(l, l[:6])


# ── discovery ──────────────────────────────────────────────────────────────
def discover_disp():
    """Return ordered list of (model, method, data) from real_sweep_*.json."""
    found = {}
    for fn in os.listdir(HERE):
        if fn.startswith("real_sweep_") and fn.endswith(".json"):
            with open(os.path.join(HERE, fn)) as f:
                d = json.load(f)
            for model, methods in d.items():
                for me, r in methods.items():
                    found[(model, me)] = r
    return order_pairs(found)


def discover_baselines():
    """Return ordered (model, method) dirs under Baselines/outputs with full files."""
    found = {}
    if not os.path.isdir(BASE_OUT):
        return []
    for model in os.listdir(BASE_OUT):
        mdir = os.path.join(BASE_OUT, model)
        if not os.path.isdir(mdir):
            continue
        for me in os.listdir(mdir):
            medir = os.path.join(mdir, me)
            if os.path.isdir(medir):
                found[(model, me)] = medir
    return order_pairs(found)


def order_pairs(d):
    def key(item):
        (m, me) = item[0]
        mi = MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)
        ei = METHOD_ORDER.index(me) if me in METHOD_ORDER else len(METHOD_ORDER)
        return (mi, m, ei, me)
    return sorted(d.items(), key=key)


# ── DISP summary + tau grid ──────────────────────────────────────────────────
def fmt(x, p=2):
    return "--" if x is None else f"{x:.{p}f}"


def disp_main_table(disp):
    lines = [
        r"\begin{table*}[h]", r"\centering", r"\small",
        r"\caption{DISP real-inference sweep, main results. "
        r"side\% $=\sum_{i\neq j}|\Delta^{\%}_{ij}|$ on the percentage-change "
        r"matrix (lower is better); self\% is the on-target effect; "
        r"$r=$ baseline side\% $/$ DISP side\% (higher is better). "
        r"T1: $\lambda$ stage ($\lambda{=}1$, $\tau{=}0.9$, $\alpha_k{=}1$). "
        r"T2: $+$ per-direction $\alpha_k$ grid ($\geq 90\%$ self). "
        r"T3: $+$ per-target SVD threshold $\tau$.}",
        r"\label{tab:disp-main}",
        r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
        r"Model & Method & Base side\% & Base self\% & "
        r"T1 side\% & T1 $r$ & T2 side\% & T2 $r$ & T3 side\% & T3 $r$ \\",
        r"\midrule",
    ]
    last_model = None
    for (model, me), r in disp:
        if last_model is not None and model != last_model:
            lines.append(r"\midrule")
        mod_cell = nice(model) if model != last_model else ""
        last_model = model
        lines.append(
            f"{mod_cell} & {me} & {fmt(r['baseline_side_pct'],1)} & "
            f"{fmt(r['baseline_self_pct'],1)} & "
            f"{fmt(r['T1']['side_pct'],1)} & {fmt(r['T1']['ratio'],3)}$\\times$ & "
            f"{fmt(r['T2']['side_pct'],1)} & {fmt(r['T2']['ratio'],3)}$\\times$ & "
            f"{fmt(r['T3']['side_pct'],1)} & {fmt(r['T3']['ratio'],3)}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def tau_grid_for(r):
    """Reconstruct uniform-tau aggregate side% from per-target threshold_candidates."""
    sw = r.get("sweep", {})
    behs = list(sw.keys())
    if not behs:
        return None
    # map threshold value -> aggregate side% (sum over targets at that uniform tau)
    grid = {}
    for t in TAUS:
        s = 0.0
        ok = True
        for b in behs:
            cand = next((c for c in sw[b]["threshold_candidates"]
                         if abs(c["threshold"] - t) < 1e-6), None)
            if cand is None:
                ok = False
                break
            s += cand["side_pct"]
        grid[t] = s if ok else None
    return grid


def disp_tau_grid_table(disp):
    lines = [
        r"\begin{table*}[h]", r"\centering", r"\scriptsize",
        r"\caption{DISP Task-3 threshold sweep: total side-effect side\% when "
        r"\emph{all} targets share a uniform SVD energy threshold $\tau$ "
        r"($\lambda{=}1$, $\alpha_k$ grid-searched per direction). "
        r"The final column (T3$^\star$) is the per-target-best selection actually "
        r"used by DISP, which can beat any uniform $\tau$. Bold $=$ best uniform "
        r"$\tau$ per row.}",
        r"\label{tab:disp-tau-grid}",
        r"\begin{tabular}{ll" + "r" * len(TAUS) + r"r}", r"\toprule",
        "Model & Method & " + " & ".join(f"{t:.2f}" for t in TAUS)
        + r" & T3$^\star$ \\", r"\midrule",
    ]
    last_model = None
    for (model, me), r in disp:
        grid = tau_grid_for(r)
        if grid is None:
            continue
        if last_model is not None and model != last_model:
            lines.append(r"\midrule")
        mod_cell = nice(model) if model != last_model else ""
        last_model = model
        vals = {t: grid[t] for t in TAUS if grid[t] is not None}
        best = min(vals, key=vals.get) if vals else None
        cells = []
        for t in TAUS:
            v = grid[t]
            if v is None:
                cells.append("--")
            else:
                c = f"{v:.1f}"
                if best is not None and abs(t - best) < 1e-6:
                    c = r"\textbf{" + c + r"}"
                cells.append(c)
        star = fmt(r["T3"]["side_pct"], 1)
        lines.append(f"{mod_cell} & {me} & " + " & ".join(cells)
                     + f" & {star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ── per-model baseline detail (llama-style) ─────────────────────────────────
def costs(SE):
    M = np.asarray(SE, float)
    n = M.shape[0]
    off = M[~np.eye(n, dtype=bool)]
    return (float(off[off > 0].sum()), float(off[off < 0].sum()),
            float(off.sum()), float(np.diag(M).sum()))


def _open(small=False):
    return [r"\begin{table*}[h]", r"\centering",
            r"\small" if small else r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}"]


def _close():
    return [r"\end{table*}"]


def matrix_tex(M, caption, label):
    n = len(LABELS)
    hdr = [short(l) for l in LABELS]
    lines = _open()
    lines += [r"\caption{" + caption + r"}", r"\label{tab:" + label + r"}",
              r"\begin{tabular}{l" + "r" * n + r"}", r"\toprule",
              " & " + " & ".join(hdr) + r" \\", r"\midrule"]
    for i in range(n):
        lines.append(hdr[i] + " & " + " & ".join(f"{M[i][j]:+.1f}" for j in range(n)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"] + _close()
    return "\n".join(lines)


def baseline_tex(base, model, me):
    lines = _open(small=True)
    lines += [r"\caption{Baseline behaviour vector: P(matching) per "
              rf"behaviour for \texttt{{{esc(model)}}} / \texttt{{{me}}} (unsteered).}}",
              r"\label{tab:base-vec-" + f"{model}-{me}" + r"}",
              r"\begin{tabular}{lr}", r"\toprule", r"Behaviour & P(matching) \\", r"\midrule"]
    for l, v in zip(LABELS, base):
        lines.append(f"{esc(l)} & {v:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"] + _close()
    return "\n".join(lines)


def summary_tex(rows, model, me):
    lines = _open(small=True)
    lines += [r"\caption{Side-effect cost summary for "
              rf"\texttt{{{esc(model)}}} / \texttt{{{me}}}. Costs are summed over "
              r"the 240 \emph{off-diagonal} entries of the signed Steering Effect "
              r"Matrix (\% change $\times$ desired-direction sign): $+$Cost / "
              r"$-$Cost are sums of positive / negative side effects, TotalCost "
              r"their signed sum, SelfEffect the diagonal (on-target) sum. Last "
              rf"four columns are ratios relative to {me} ($\alpha{{=}}1.0$).}}",
              r"\label{tab:cost-summary-" + f"{model}-{me}" + r"}",
              r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
              r"& \multicolumn{4}{c}{Absolute (\% summed)} & "
              r"\multicolumn{4}{c}{Ratio vs.\ " + me + r"} \\",
              r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
              r"Method & $+$Cost & $-$Cost & TotalCost & SelfEffect & "
              r"$+$Cost & $-$Cost & Total & Self \\", r"\midrule"]
    for (name, p, ng, tot, slf, rp, rn, rt, rs) in rows:
        lines.append(f"{name} & {p:+.1f} & {ng:+.1f} & {tot:+.1f} & {slf:+.1f} & "
                     f"{rp:.3f} & {rn:.3f} & {rt:.3f} & {rs:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"] + _close()
    return "\n".join(lines)


def code_key_tex():
    lines = [r"\subsection*{Behaviour code key}",
             r"\begin{tabular}{ll}\toprule Code & Behaviour \\\midrule"]
    for l in LABELS:
        lines.append(f"{short(l)} & {esc(l)} \\\\")
    lines += [r"\bottomrule\end{tabular}", ""]
    return "\n".join(lines)


def _load(medir, name):
    with open(os.path.join(medir, name)) as f:
        return json.load(f)


def effect_baseline(medir, name):
    d = _load(medir, name)
    return np.array(d["steering_effect_matrix"], float)


def effect_disp(disp_map, model, me, task):
    r = disp_map.get((model, me))
    if r is None:
        return None
    pct = np.array(r[task]["steered_pct_matrix"], float)
    return pct * SIGN[np.newaxis, :]


def model_detail(model, me, medir, disp_map):
    alphas = ["0.50", "0.75", "1.00", "1.50"]  # 1.25 also present; keep 4 to bound length
    have_125 = os.path.exists(os.path.join(medir, f"T1_{me}_alpha_1.25.json"))
    alpha_list = ["0.50", "0.75", "1.00", "1.25", "1.50"] if have_125 else alphas

    eff = {a: effect_baseline(medir, f"T1_{me}_alpha_{a}.json") for a in alpha_list}
    pca = effect_baseline(medir, f"T2_{me}_PCA_projection.json")
    gs = effect_baseline(medir, f"T3_{me}_gram_schmidt.json")
    usvd = effect_baseline(medir, f"T4_{me}_unscaled_svd.json")
    cast = effect_baseline(medir, f"T5_CAST_{me}.json")
    dcast = effect_baseline(medir, f"T6_CAST_DISP_T3_{me}.json")
    base = np.array(_load(medir, f"T1_{me}_alpha_1.00.json")["baseline"], float)

    d1 = effect_disp(disp_map, model, me, "T1")
    d2 = effect_disp(disp_map, model, me, "T2")
    d3 = effect_disp(disp_map, model, me, "T3")

    methods = [(rf"{me} ($\alpha{{=}}1.0$)", eff["1.00"]),
               ("PCA-projection", pca), ("Gram--Schmidt", gs),
               ("Unscaled SVD", usvd), ("CAST", cast)]
    if d1 is not None:
        methods += [(r"DISP T1 ($\lambda$)", d1), (r"DISP T2 ($\alpha_k$)", d2),
                    (r"DISP T3 ($\tau$)", d3)]
    methods.append(("DISP-best $+$ CAST", dcast))

    ref = costs(methods[0][1])
    rp0, rn0, rt0, rs0 = ref
    rows = []
    for name, SE in methods:
        p, ng, tot, slf = costs(SE)
        rows.append((name, p, ng, tot, slf,
                     p / rp0 if rp0 else float("nan"),
                     ng / rn0 if rn0 else float("nan"),
                     tot / rt0 if rt0 else float("nan"),
                     slf / rs0 if rs0 else float("nan")))

    out = [rf"\subsection*{{{nice(model)} / {me}}}",
           rf"\label{{app:{model}-{me}}}",
           r"Rows index the \emph{steered} behaviour, columns the "
           r"\emph{evaluated} behaviour; cells are the signed Steering Effect "
           r"(\% change $\times$ desired-direction sign).",
           "",
           baseline_tex(base, model, me),
           rf"\subsubsection*{{{me} Steering Effect Matrix at "
           f"{len(alpha_list)} strengths}}"]
    for k, a in enumerate(alpha_list):
        out.append(matrix_tex(eff[a],
                   rf"{me} effect matrix at $\alpha={a}$ "
                   rf"(\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                   f"eff-{model}-{me}-a{a.replace('.', '')}"))
        out.append("")
    out += [r"\subsubsection*{Correction baselines}",
            matrix_tex(pca, rf"PCA-projection (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                       f"pca-{model}-{me}"), "",
            matrix_tex(gs, rf"Gram--Schmidt (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                       f"gs-{model}-{me}"), "",
            matrix_tex(usvd, rf"Unscaled SVD (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                       f"usvd-{model}-{me}"), "",
            r"\subsubsection*{CAST}",
            matrix_tex(cast, rf"CAST (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                       f"cast-{model}-{me}"), ""]
    if d1 is not None:
        out += [r"\subsubsection*{DISP (real-inference sweep) effect matrices}",
                matrix_tex(d1, rf"DISP T1 (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                           f"disp-t1-{model}-{me}"), "",
                matrix_tex(d2, rf"DISP T2 (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                           f"disp-t2-{model}-{me}"), "",
                matrix_tex(d3, rf"DISP T3 (\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                           f"disp-t3-{model}-{me}"), ""]
    out += [r"\subsubsection*{DISP-best $+$ CAST}",
            matrix_tex(dcast, rf"DISP-best (T3) $+$ CAST gating "
                       rf"(\texttt{{{esc(model)}}} / \texttt{{{me}}}).",
                       f"disp-cast-{model}-{me}"), "",
            r"\subsubsection*{Side-effect cost summary}",
            summary_tex(rows, model, me), r"\clearpage"]
    return "\n".join(out)


# ── build ────────────────────────────────────────────────────────────────
def main():
    disp = discover_disp()
    disp_map = {k: v for k, v in disp}
    baselines = discover_baselines()

    parts = [
        r"% Auto-generated by Final/DISP/make_full_appendix.py",
        r"% Requires in preamble: \usepackage{booktabs}, \usepackage{caption}.",
        r"% Wide tables use table*; include under \onecolumn if needed.",
        r"\section{DISP sweep summary}",
        r"\label{app:disp-summary}",
        disp_main_table(disp), "",
        disp_tau_grid_table(disp), r"\clearpage",
        r"\section{Per-model baseline / DISP / CAST detail}",
        r"\label{app:per-model}",
        code_key_tex(), r"\clearpage",
    ]
    skipped = []
    for (model, me), medir in baselines:
        try:
            parts.append(model_detail(model, me, medir, disp_map))
        except (FileNotFoundError, KeyError) as e:
            skipped.append((model, me, repr(e)))

    with open(TEX, "w") as f:
        f.write("\n".join(parts) + "\n")

    print(f"wrote {TEX}")
    print(f"DISP entries: {[f'{m}/{me}' for (m, me), _ in disp]}")
    print(f"Baseline detail: {[f'{m}/{me}' for (m, me), _ in baselines if (m, me, ) not in [(s[0], s[1]) for s in skipped]]}")
    if skipped:
        print("SKIPPED (missing files):")
        for m, me, e in skipped:
            print(f"   {m}/{me}: {e}")


if __name__ == "__main__":
    main()
