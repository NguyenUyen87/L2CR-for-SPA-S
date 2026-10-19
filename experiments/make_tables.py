"""Generate the LaTeX tables and figure data of the paper from results/.

    python experiments/make_tables.py [OUTPUT_DIR]      # default: tables/

Inputs: results/small.jsonl, results/large.jsonl, results/tradeoff.jsonl,
results/ablation.txt, weights/rl.json. Also writes results/summary_*.json.
"""
import json
import os
import re
import statistics as st
import sys

from _common import ROOT, RESULTS as RES
from spa_lq.stats import wilcoxon, holm

OUT = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tables")
os.makedirs(OUT, exist_ok=True)

NICE = {"HUN": r"\HUN", "HUN-16": r"{\sc Hun-16}", "ABM+HUN": r"{\sc Abm+Hun}", "CASCADE": r"{\sc Cascade}",
        "CEM": r"{\sc Cem}", "RL": r"{\sc RL}", "ABM+RL": r"{\sc Abm+RL}", "RL-16": r"{\sc RL-16}",
        "HUN-MS": r"{\sc Hun-MS}", "SAC-HUN": r"{\sc Sac-Hun}", "L2CR": r"\LCR", "ABM+L2CR": r"{\sc Abm+}\LCR", "L2CR-free": r"\LCR$_{\infty}$"}


def write(name, s):
    for a in ("\\toprule", "\\midrule", "\\bottomrule"):
        s = s.replace(a, "\\hline")
    open(os.path.join(OUT, name), "w").write(s)
    print("wrote", name)


# ------------------------------------------------------------------ methods table (static)
write("tab_methods.tex", r"""\begin{table}[h]
\caption{Methods compared. C: constructive; B: time-budgeted. ``Ours'' marks methods introduced or first evaluated as heuristics here.}
\label{tab:methods}
\centering
\small
\begin{tabular}{llll}
\hline
Method & Type & Description & Source \\\hline
\HUN & C & close deficient project with minimum $|M(p_j)|/\ell_j$ & \citet{BiroFleinerIrvingManlove2010} \\
{\sc Hun-16} & C & best of \HUN{} and 15 runs with random tie-breaking & baseline \\
{\sc Abm+Hun} & C & {\sc Abm} preprocessing, then \HUN & \citet{AgostonBiroMcBride2016}; ours \\
{\sc Cascade} & C & hand-crafted linear rule on 3 features & ours \\
{\sc Cem} & C & linear rule learned by cross-entropy method & ours \\
{\sc RL} & C & linear rule learned by REINFORCE & ours \\
{\sc Abm+RL} & C & {\sc Abm} preprocessing, then {\sc RL} & ours \\
{\sc RL-16} & C & best of greedy and 15 sampled {\sc RL} trajectories & ours \\
{\sc Hun-MS} & B & random-restart \HUN{} until the budget is spent & baseline \\
{\sc Sac-Hun} & B & sacrifice search completed by \HUN & ours \\
\LCR & B & sacrifice search completed by {\sc RL} & ours \\
{\sc Abm+}\LCR & B & {\sc Abm} preprocessing, then \LCR & ours \\
IP & exact & Appendix~\ref{app:ip}, HiGHS, small instances & \citet{Olaosebikan2020}, \citet{AgostonBiroSzanto2018} \\
\hline
\end{tabular}
\end{table}
""")

# ------------------------------------------------------------------ small instances (E12)
R = [json.loads(l) for l in open(RES / "small.jsonl")]
opt = [r for r in R if r["status"] == "Optimal"]
M = ["HUN", "HUN-16", "ABM+HUN", "CASCADE", "CEM", "RL", "RL-16", "HUN-MS", "SAC-HUN", "L2CR", "L2CR-free"]
cells = [(n, lq) for lq in (0.6, 0.8) for n in (40, 60, 80)]
cells = [c for c in cells if any(r["n"] == c[0] and r["lq"] == c[1] for r in opt)]
hg = st.mean(r["opt"] - r["HUN"] for r in opt)
lines = []
for k in M:
    vals = []
    for n, lq in cells:
        C = [r for r in opt if r["n"] == n and r["lq"] == lq]
        vals.append(f"{st.mean(100 * (r['opt'] - r[k]) / r['opt'] for r in C):.2f}")
    closed = 100 * (hg - st.mean(r["opt"] - r[k] for r in opt)) / hg
    nopt = sum(r[k] == r["opt"] for r in opt)
    coal = st.mean(r[k + "_coal"] for r in opt)
    lines.append(f"{NICE[k]} & " + " & ".join(vals) + f" & {nopt} & {closed:.0f}\\% & {coal:.2f} \\\\")
head = " & ".join(f"{n}/{lq}" for n, lq in cells)
cnts = " & ".join(str(sum(1 for r in opt if r["n"] == n and r["lq"] == lq)) for n, lq in cells)
write("tab_gap2.tex", r"""\begin{table}[h]
\caption{Small instances: mean gap (\%) to the IP optimum per cell $n/\rho$ (instances solved to optimality within 120\,s); \#opt: instances where the method is optimal; closed: share of the mean gap of \HUN{} closed by the method; coal.: mean number of blocking coalitions. Budgeted methods use 5\,s. \LCR$_{\infty}$ is \LCR{} without coalition cap.}
\label{tab:gap2}
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l""" + "r" * len(cells) + r"""rrr}
\hline
$n/\rho$ & """ + head + r""" & \#opt & closed & coal. \\
instances & """ + cnts + f" & {len(opt)} & & " + r"""\\\hline
""" + "\n".join(lines) + r"""
\hline
\end{tabular}
\end{table}
""")
small = {"n_opt": len(opt), "n_all": len(R),
         "closed": {k: 100 * (hg - st.mean(r["opt"] - r[k] for r in opt)) / hg for k in M},
         "gap": {k: st.mean(100 * (r["opt"] - r[k]) / r["opt"] for r in opt) for k in M},
         "nopt": {k: sum(r[k] == r["opt"] for r in opt) for k in M},
         "coal": {k: st.mean(r[k + "_coal"] for r in opt) for k in M}}
json.dump(small, open(RES / "summary_small.json", "w"), indent=1)

# ------------------------------------------------------------------ large instances (E10)
E = [json.loads(l) for l in open(RES / "large.jsonl")]
NS = sorted({r["n"] for r in E})
CELLS = [(n, lq) for lq in (0.6, 0.8) for n in NS]


def fmt_p(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def block(methods, ref="HUN", vs=None):
    """Per-cell mean improvement over ref; pooled Wilcoxon vs ref (Holm over the
    methods of the table); optional pooled comparison against method `vs`."""
    pooled = {}
    for k in methods:
        if k == ref:
            continue
        C = [r for r in E if k in r]
        pooled[k] = wilcoxon([r[k] for r in C], [r[ref] for r in C])
    keys = list(pooled)
    adj = dict(zip(keys, holm([pooled[k] for k in keys])))
    rows = []
    for k in methods:
        vals = []
        for n, lq in CELLS:
            C = [r for r in E if r["n"] == n and r["lq"] == lq and k in r]
            vals.append("--" if not C else ("0" if k == ref else f"{st.mean(100 * (r[k] - r[ref]) / r[ref] for r in C):+.2f}"))
        C = [r for r in E if k in r]
        wl = "--" if k == ref else f"{sum(r[k] > r[ref] for r in C)}/{sum(r[k] < r[ref] for r in C)}"
        p = "--" if k == ref else fmt_p(adj[k])
        Ck = [r for r in C if r["HUN_coal"] > 0]
        coal = 100 * st.mean(r[k + "_coal"] / r["HUN_coal"] for r in Ck)
        extra = ""
        if vs is not None:
            if k == vs:
                extra = " & -- "
            else:
                Cv = [r for r in C if vs in r]
                d = st.mean(100 * (r[k] - r[vs]) / r[vs] for r in Cv)
                extra = f" & {d:+.2f} ({sum(r[k] > r[vs] for r in Cv)}/{sum(r[k] < r[vs] for r in Cv)})"
        rows.append(f"{NICE[k]} & " + " & ".join(vals) + f" & {wl} & {p} & {coal:.0f}{extra} \\\\")
    return rows


hdr_n = " & ".join(f"{n // 1000}k" for n in NS)
cnt_line = " & ".join(str(len([r for r in E if r["n"] == n and r["lq"] == lq])) for n, lq in CELLS)


def tabhead(extra_col=""):
    ncol = len(CELLS) + 4 + (1 if extra_col else 0)
    return (r"\begin{tabular}{l" + "r" * (ncol - 1) + "}" + "\n\\hline\n" +
            r" & \multicolumn{" + str(len(NS)) + r"}{c}{$\rho=0.6$} & \multicolumn{" + str(len(NS)) +
            r"}{c}{$\rho=0.8$} & W/L & $p$ & coal." + (" & " + extra_col if extra_col else "") + r"\\" + "\n" +
            r"$n$ & " + hdr_n + " & " + hdr_n + r" & & & (\%)" + (" & " if extra_col else "") + r" \\" + "\n" +
            r"instances & " + cnt_line + r" & & & " + (" & " if extra_col else "") + r" \\\hline" + "\n")


cons = ["HUN", "HUN-16", "ABM+HUN", "CASCADE", "CEM", "RL", "ABM+RL", "RL-16"]
write("tab_constructive.tex", r"""\begin{table}[h]
\caption{Constructive rules on large instances: mean improvement over \HUN{} in matched students (\%) per size and tightness. W/L: instances (out of all sizes) on which the method matches more/fewer students than \HUN; $p$: pooled two-sided Wilcoxon test against \HUN{} with Holm's correction over the methods of the table; coal.: mean number of blocking coalitions relative to \HUN{} (\%). Learned rules were trained on $n=200$.}
\label{tab:constructive}
\centering
\small
\setlength{\tabcolsep}{3pt}
""" + tabhead() + "\n".join(block(cons)) + r"""
\hline
\end{tabular}
\end{table}
""")
bud = ["HUN-MS", "SAC-HUN", "L2CR", "ABM+L2CR"]
write("tab_budget.tex", r"""\begin{table}[h]
\caption{Time-budgeted methods (budget 30, 60, 120 and 240\,s for $n=1$k, 2k, 5k and 10k): columns as in Table~\ref{tab:constructive}; the last column gives the pooled mean difference to {\sc Abm+Hun} (\%) with wins/losses. The coalition cap is the number of coalitions of \HUN{} for {\sc Sac-Hun} and \LCR, and that of {\sc Abm+Hun} for {\sc Abm+}\LCR.}
\label{tab:budget}
\centering
\small
\setlength{\tabcolsep}{3pt}
""" + tabhead("vs {\\sc Abm+Hun}") + "\n".join(block(bud, vs="ABM+HUN")) + r"""
\hline
\end{tabular}
\end{table}
""")

# head-to-head summaries for the text
def h2h(a, b):
    C = [r for r in E if a in r and b in r]
    if not C:
        return None
    d = st.mean(100 * (r[a] - r[b]) / r[b] for r in C)
    return {"delta": d, "win": sum(r[a] > r[b] for r in C), "tie": sum(r[a] == r[b] for r in C),
            "loss": sum(r[a] < r[b] for r in C), "p": wilcoxon([r[a] for r in C], [r[b] for r in C]), "N": len(C)}


summary = {"N": len(E), "by_cell": {}, "h2h": {}}
for a, b in [("ABM+L2CR", "ABM+HUN"), ("ABM+L2CR", "L2CR"), ("L2CR", "SAC-HUN"), ("L2CR", "ABM+HUN"), ("L2CR", "HUN-MS"), ("L2CR", "HUN"), ("SAC-HUN", "HUN"),
             ("ABM+HUN", "HUN"), ("ABM+HUN", "RL"), ("ABM+RL", "ABM+HUN"), ("RL", "HUN"), ("RL", "CASCADE"),
             ("RL", "CEM"), ("RL-16", "HUN-16"), ("HUN-MS", "HUN"), ("SAC-HUN", "ABM+HUN"), ("HUN-16", "HUN")]:
    summary["h2h"][f"{a} vs {b}"] = h2h(a, b)
for n, lq in CELLS:
    C = [r for r in E if r["n"] == n and r["lq"] == lq]
    if not C:
        continue
    summary["by_cell"][f"{n}/{lq}"] = {k: st.mean(100 * (r[k] - r["HUN"]) / r["HUN"] for r in C)
                                       for k in cons + bud if all(k in r for r in C)}
    summary["by_cell"][f"{n}/{lq}"]["coal"] = {k: st.mean(r[k + "_coal"] for r in C) for k in cons + bud if all(k in r for r in C)}
    summary["by_cell"][f"{n}/{lq}"]["time"] = {k: st.mean(r[k + "_t"] for r in C) for k in cons + bud if all(k in r for r in C)}
    summary["by_cell"][f"{n}/{lq}"]["L2CR_sacrificed"] = st.mean(r["L2CR_sacrificed"] for r in C)
    summary["by_cell"][f"{n}/{lq}"]["L2CR_evals"] = st.mean(r["L2CR_evals"] for r in C)
json.dump(summary, open(RES / "summary_large.json", "w"), indent=1)

# ------------------------------------------------------------------ trade-off (E13)
P = [json.loads(l) for l in open(RES / "tradeoff.jsonl")]
for n in sorted({r["n"] for r in P}):
    Q = [r for r in P if r["n"] == n]
    with open(os.path.join(OUT, f"pareto_points_{n}.dat"), "w") as fh:
        fh.write("method coal gain\n")
        for k in ("HUN", "ABM+HUN", "RL"):
            fh.write(f"{k} {st.mean(r[k][1] for r in Q):.3f} {st.mean(100 * (r[k][0] - r['HUN'][0]) / r['HUN'][0] for r in Q):.3f}\n")
    for fam in ("L2CR", "ABM+L2CR"):
        with open(os.path.join(OUT, f"pareto_{fam.replace('+', '')}_{n}.dat"), "w") as fh:
            fh.write("label coal gain\n")
            # caps below the coalition count of the starting solution cannot be met:
            # L2CR starts from RL (caps >= that of HUN), ABM+L2CR from ABM+HUN
            labels = ("hun", "2hun", "inf") if fam == "L2CR" else ("abm", "hun", "inf")
            for lab in labels:
                pts = []
                for r in Q:
                    ca, ch = r["ABM+HUN"][1], r["HUN"][1]
                    cap = {"abm": ca, "mid": (ca + ch) // 2, "hun": ch, "2hun": 2 * ch, "inf": 10 ** 6}[lab]
                    if str(cap) in r.get(fam, {}):
                        m, c = r[fam][str(cap)]
                        pts.append((c, 100 * (m - r["HUN"][0]) / r["HUN"][0]))
                if pts:
                    fh.write(f"{lab} {st.mean(p[0] for p in pts):.3f} {st.mean(p[1] for p in pts):.3f}\n")
    print("wrote pareto data", n)


# ---------------------------------------------------------------- Table: ablation
abl = []
for l in open(RES / "ablation.txt"):
    m = re.match(r"(.+?)\s+RL vs HUN on validation: mean ([+-][\d.]+)% \(per training seed: (.+)\)\s+pooled p=([\d.]+)", l)
    if m:
        abl.append((m.group(1).strip(), m.group(2), m.group(3), float(m.group(4))))
label = {"full": "all features", "-popularity": "without popularity", "-cascade": "without cascade",
         "quota/load only": "quota/load only"}
write("tab_ablation.tex", r"""\begin{table}[t]
\centering
\caption{Feature-group ablation: mean improvement over HUN on the validation set (30 instances, $n=500$, mixed $\rho$ and $z$), three training seeds per variant; $p$ pools the three seeds.}
\label{tab:ablation}
\small
\begin{tabular}{lrlr}
\toprule
features & $\Delta$ (\%) & per training seed (\%) & $p$ \\
\midrule
""" + "\n".join(f"{label.get(a, a)} & {b} & {c} & {fmt_p(d)} \\\\" for a, b, c, d in abl) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Table: weights
w = json.load(open(ROOT / "weights" / "rl.json"))
desc = ["bias", r"fill ratio $|M(p_j)|/c_j$", r"relative deficit $(\ell_j-|M(p_j)|)/\ell_j$", r"$\ell_j/c_j$",
        "applicants (normalised)", "first-choice applicants (normalised)", r"lecturer load $|M(l_k)|/d_k$",
        r"members whose next open choice is deficient", r"members whose next open choice is full",
        r"members with no further open choice", r"members whose next open choice has room",
        r"$|M(p_j)|/6$"]
group = ["--", "quota/load", "quota/load", "quota/load", "popularity", "popularity", "quota/load",
         "cascade", "cascade", "cascade", "cascade", "quota/load"]
write("tab_weights.tex", r"""\begin{table}[t]
\centering
\caption{Features $\phi(O_t,p_j)$ of a deficient project $p_j$ and the learned weights $\theta$ (larger score = more likely to be closed). Cascade features are fractions of the students currently assigned to $p_j$.}
\label{tab:weights}
\small
\begin{tabular}{rllr}
\toprule
\# & feature & group & $\theta$ \\
\midrule
""" + "\n".join(f"{i} & {desc[i]} & {group[i]} & {w[i]:+.2f} \\\\" for i in range(1, len(w))) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

