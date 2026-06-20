"""
make_llama_caa_tables.py  --  appendix-style LaTeX tables for the
llama-3.1-8b / CAA baseline + DISP + CAST results.

Sources (all on the 100-item CAA test set, internally consistent):
    DISP/Baselines/outputs/llama-3.1-8b/T1_CAA_alpha_*.json   (CAA sweep)
    DISP/Baselines/outputs/llama-3.1-8b/T2_PCA_projection.json
    DISP/Baselines/outputs/llama-3.1-8b/T3_gram_schmidt.json
    DISP/Baselines/outputs/llama-3.1-8b/T4_unscaled_svd.json
    DISP/Baselines/outputs/llama-3.1-8b/T5_CAST_CAA.json
    DISP/Baselines/outputs/llama-3.1-8b/T6_CAST_DISP_T3.json
    DISP/real_sweep_llama-3.1-8b.json -> [llama-3.1-8b][CAA][T1|T2|T3]

Conventions (per user):
  * "effect matrix" = signed Steering Effect Matrix = pct_change * direction
    sign of the *evaluated* (column) behaviour. Baseline files already store
    this as steering_effect_matrix; for DISP T1/T2/T3 we sign the stored
    steered_pct_matrix ourselves.
  * Costs are computed over OFF-DIAGONAL entries only (side effects):
        +Cost = sum of positive off-diagonal entries
        -Cost = sum of negative off-diagonal entries
        TotalCost = sum of all off-diagonal entries (= +Cost + -Cost)
        SelfEffect = sum of the diagonal (on-target effect)
  * Reduction columns are ratios method/CAA(alpha=1.0); the CAA row is 1.000.

Output: DISP/Baselines/llama_caa_appendix.tex
Run:   python make_llama_caa_tables.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../DISP/Baselines
DISP_DIR = os.path.dirname(HERE)                           # .../DISP
FINAL = os.path.dirname(DISP_DIR)                          # .../Final
sys.path.insert(0, FINAL)
import config as cfg                                        # noqa: E402

MODEL = "llama-3.1-8b"
OUT_DIR = os.path.join(HERE, "outputs", MODEL)
TEX = os.path.join(HERE, "llama_caa_appendix.tex")

SHORT = {
    "Sycophancy": "Syco", "Hallucination": "Hall", "Corrigibility": "Corr",
    "Survival Instinct": "Surv", "Myopic Reward": "Myop",
    "AI Coordination": "AI-C", "Refusal": "Ref", "Power Seeking": "Pow",
    "Wealth Seeking": "Wlth", "Self-Coordination": "S-Co",
    "Version Coordination": "V-Co", "Corrigible Less-HHH": "CrLs",
    "Corrigible More-HHH": "CrMo", "One-Box Tendency": "1Box",
    "Self-Aware (AI)": "SaAI", "Self-Aware (Text)": "SaTx",
}

BEHAVIORS = cfg.BEHAVIORS
LABELS = [cfg.BEHAVIOR_LABELS[b] for b in BEHAVIORS]
SIGN = np.array([+1.0 if cfg.DESIRED_DIRECTION[b] == "increase" else -1.0
                 for b in BEHAVIORS])                       # per column j


def esc(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")
            .replace("$", r"\$"))


# --------------------------------------------------------------------------
# Loading the 9 signed effect matrices
# --------------------------------------------------------------------------
def _load_baseline_json(name):
    with open(os.path.join(OUT_DIR, name)) as f:
        return json.load(f)


def effect_from_baseline(name):
    """Baseline files already store the signed steering_effect_matrix."""
    d = _load_baseline_json(name)
    assert d["behaviors"] == BEHAVIORS, f"behaviour order mismatch in {name}"
    return np.array(d["steering_effect_matrix"], dtype=float)


def effect_from_disp(task):
    """real_sweep stores the unsigned pct matrix; sign it by column direction."""
    with open(os.path.join(DISP_DIR, f"real_sweep_{MODEL}.json")) as f:
        rs = json.load(f)
    pct = np.array(rs[MODEL]["CAA"][task]["steered_pct_matrix"], dtype=float)
    return pct * SIGN[np.newaxis, :]


def baseline_vector():
    d = _load_baseline_json("T1_CAA_alpha_1.00.json")
    return np.array(d["baseline"], dtype=float)


# --------------------------------------------------------------------------
# Cost computation (off-diagonal side effects + diagonal self-effect)
# --------------------------------------------------------------------------
def costs(SE):
    M = np.asarray(SE, dtype=float)
    n = M.shape[0]
    off = M[~np.eye(n, dtype=bool)]
    pos = float(off[off > 0].sum())
    neg = float(off[off < 0].sum())
    total = float(off.sum())
    self_eff = float(np.diag(M).sum())
    return pos, neg, total, self_eff


# --------------------------------------------------------------------------
# LaTeX helpers (mirroring Final/make_appendix.py styling)
# --------------------------------------------------------------------------
def _open(small=False):
    return [r"\par\bigskip\noindent\begingroup", r"\begin{center}",
            r"\small" if small else r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}"]


def _close():
    return [r"\end{center}\endgroup\par\vspace{1.5em}"]


def matrix_tex(M, caption, label):
    n = len(LABELS)
    hdr = [SHORT.get(l, l[:6]) for l in LABELS]
    lines = _open(small=False)
    lines += [
        r"\captionof{table}{" + caption + r"}",
        r"\label{tab:" + label + r"}",
        r"\begin{tabular}{l" + "r" * n + r"}", r"\toprule",
        " & " + " & ".join(hdr) + r" \\", r"\midrule",
    ]
    for i in range(n):
        cells = [f"{M[i][j]:+.1f}" for j in range(n)]
        lines.append(hdr[i] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += _close()
    return "\n".join(lines)


def baseline_tex(base):
    lines = _open(small=True)
    lines += [
        r"\captionof{table}{Baseline behaviour vector: P(matching) per behaviour "
        r"for \texttt{llama-3.1-8b} / \texttt{CAA} (100-item test set, unsteered).}",
        r"\label{tab:base-vec}",
        r"\begin{tabular}{lr}", r"\toprule",
        r"Behaviour & P(matching) \\", r"\midrule",
    ]
    for l, v in zip(LABELS, base):
        lines.append(f"{esc(l)} & {v:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += _close()
    return "\n".join(lines)


def summary_tex(rows):
    """rows = list of (name, pos, neg, total, self, rp, rn, rt, rs)."""
    lines = _open(small=True)
    lines += [
        r"\captionof{table}{Side-effect cost summary for \texttt{llama-3.1-8b} / "
        r"\texttt{CAA}. Costs are summed over the 240 \emph{off-diagonal} entries "
        r"of the signed Steering Effect Matrix (\% change $\times$ desired-direction "
        r"sign): $+$Cost / $-$Cost are the sums of positive / negative side effects, "
        r"TotalCost their signed sum, and SelfEffect the diagonal (on-target) sum. "
        r"The last four columns are ratios relative to CAA ($\alpha{=}1.0$), so the "
        r"CAA row is $1.000$; values below $1$ indicate a reduction.}",
        r"\label{tab:cost-summary}",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"& \multicolumn{4}{c}{Absolute (\% summed)} & "
        r"\multicolumn{4}{c}{Ratio vs.\ CAA} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Method & $+$Cost & $-$Cost & TotalCost & SelfEffect & "
        r"$+$Cost & $-$Cost & Total & Self \\", r"\midrule",
    ]
    for (name, p, ng, tot, slf, rp, rn, rt, rs) in rows:
        lines.append(
            f"{name} & {p:+.1f} & {ng:+.1f} & {tot:+.1f} & {slf:+.1f} & "
            f"{rp:.3f} & {rn:.3f} & {rt:.3f} & {rs:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    lines += _close()
    return "\n".join(lines)


def code_key_tex():
    lines = [r"\subsection*{Behaviour code key}",
             r"\begin{tabular}{ll}\toprule Code & Behaviour \\\midrule"]
    for l in LABELS:
        lines.append(f"{SHORT.get(l, l)} & {esc(l)} \\\\")
    lines += [r"\bottomrule\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def main():
    alphas = ["0.50", "0.75", "1.00", "1.25", "1.50"]
    caa_eff = {a: effect_from_baseline(f"T1_CAA_alpha_{a}.json") for a in alphas}
    pca = effect_from_baseline("T2_PCA_projection.json")
    gs = effect_from_baseline("T3_gram_schmidt.json")
    usvd = effect_from_baseline("T4_unscaled_svd.json")
    cast = effect_from_baseline("T5_CAST_CAA.json")
    disp_t1 = effect_from_disp("T1")
    disp_t2 = effect_from_disp("T2")
    disp_t3 = effect_from_disp("T3")
    disp_cast = effect_from_baseline("T6_CAST_DISP_T3.json")
    base = baseline_vector()

    # ---- summary rows (9 methods) ----
    methods = [
        (r"CAA ($\alpha{=}1.0$)", caa_eff["1.00"]),
        ("PCA-projection",        pca),
        ("Gram--Schmidt",         gs),
        ("Unscaled SVD",          usvd),
        ("CAST",                  cast),
        (r"DISP T1 ($\lambda$)",  disp_t1),
        (r"DISP T2 ($\alpha_k$)", disp_t2),
        (r"DISP T3 ($\tau$)",     disp_t3),
        (r"DISP-best $+$ CAST",   disp_cast),
    ]
    ref = costs(methods[0][1])                       # CAA(alpha=1.0)
    rp0, rn0, rt0, rs0 = ref
    rows = []
    for name, SE in methods:
        p, ng, tot, slf = costs(SE)
        rows.append((name, p, ng, tot, slf,
                     p / rp0 if rp0 else float("nan"),
                     ng / rn0 if rn0 else float("nan"),
                     tot / rt0 if rt0 else float("nan"),
                     slf / rs0 if rs0 else float("nan")))

    # ---- assemble .tex ----
    out = [
        r"% Auto-generated by DISP/Baselines/make_llama_caa_tables.py.",
        r"% Requires in the main preamble: \usepackage{booktabs} and \usepackage{caption}.",
        r"% Tables are non-floating; wrap the include in \onecolumn if two-column.",
        r"\section*{Appendix: llama-3.1-8b / CAA --- baseline, corrections, DISP \& CAST}",
        r"\label{app:llama-caa}",
        r"In every matrix, rows index the \emph{steered} behaviour and columns the "
        r"\emph{evaluated} behaviour; cells are the signed Steering Effect "
        r"(\% change $\times$ desired-direction sign). Behaviour names are abbreviated "
        r"per the key below.",
        "",
        code_key_tex(),
        r"\subsection*{Baseline behaviour vector}",
        baseline_tex(base),
        r"\clearpage",
        r"\subsection*{CAA Steering Effect Matrix at five strengths}",
    ]
    for k, a in enumerate(alphas):
        out.append(matrix_tex(
            caa_eff[a],
            rf"CAA Steering Effect Matrix at $\alpha={a}$ "
            r"(\texttt{llama-3.1-8b} / \texttt{CAA}).",
            f"caa-alpha-{a.replace('.', '')}"))
        out.append("")
        if (k + 1) % 2 == 0:
            out.append(r"\clearpage")

    out += [
        r"\clearpage",
        r"\subsection*{Correction baselines}",
        matrix_tex(pca, r"PCA-projection corrected Steering Effect Matrix "
                        r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "pca"),
        "",
        matrix_tex(gs, r"Gram--Schmidt corrected Steering Effect Matrix "
                       r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "gs"),
        r"\clearpage",
        matrix_tex(usvd, r"Unscaled-SVD corrected Steering Effect Matrix "
                         r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "usvd"),
        "",
        r"\subsection*{CAST}",
        matrix_tex(cast, r"CAST (condition-gated CAA) Steering Effect Matrix "
                         r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "cast"),
        r"\clearpage",
        r"\subsection*{DISP (real-inference sweep) Steering Effect Matrices}",
        matrix_tex(disp_t1, r"DISP T1 ($\lambda$ grid) Steering Effect Matrix "
                            r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "disp-t1"),
        "",
        matrix_tex(disp_t2, r"DISP T2 ($\alpha_k$ grid) Steering Effect Matrix "
                            r"(\texttt{llama-3.1-8b} / \texttt{CAA}).", "disp-t2"),
        r"\clearpage",
        matrix_tex(disp_t3, r"DISP T3 ($\tau$ grid, DISP-best) Steering Effect "
                            r"Matrix (\texttt{llama-3.1-8b} / \texttt{CAA}).", "disp-t3"),
        "",
        r"\subsection*{DISP-best $+$ CAST}",
        matrix_tex(disp_cast, r"DISP-best (T3) vectors with CAST gating: Steering "
                              r"Effect Matrix (\texttt{llama-3.1-8b} / \texttt{CAA}).",
                   "disp-cast"),
        r"\clearpage",
        r"\subsection*{Side-effect cost summary}",
        summary_tex(rows),
    ]

    with open(TEX, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {TEX}")

    # console sanity check
    print(f"\n{'method':22s} {'+Cost':>9} {'-Cost':>9} {'Total':>9} "
          f"{'Self':>9}  {'r+':>6} {'r-':>6} {'rTot':>6} {'rSelf':>6}")
    for (name, p, ng, tot, slf, rp, rn, rt, rs) in rows:
        clean = (name.replace(r"$\alpha{=}1.0$", "a=1").replace(r"$\lambda$", "lam")
                 .replace(r"$\alpha_k$", "a_k").replace(r"$\tau$", "tau")
                 .replace(r"$+$", "+").replace("--", "-"))
        print(f"{clean:22s} {p:9.1f} {ng:9.1f} {tot:9.1f} {slf:9.1f}  "
              f"{rp:6.3f} {rn:6.3f} {rt:6.3f} {rs:6.3f}")


if __name__ == "__main__":
    main()
