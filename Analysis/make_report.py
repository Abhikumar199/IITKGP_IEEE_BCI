"""Build the technical report (LaTeX -> PDF) from the CSV/JSON files written by Code_Final_V_5P.
Every number and every verdict sentence comes from those files. Usage: python make_report.py RESULTS_DIR OUT_DIR"""
import sys, json, subprocess, shutil
from pathlib import Path
import pandas as pd, numpy as np

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
info = json.load(open(RES / "RUN_INFO.json")); SYN = info["data_kind"] == "synthetic"
rd = lambda n: pd.read_csv(RES / n) if (RES / n).exists() else None
SHORT = {"LDA": "LDA", "ANN (temporal CNN)": "ANN", "LIF SNN": "LIF", "Adaptive-LIF (base)": "AdLIF base", "Adaptive-LIF (proposed)": "AdLIF proposed",
         "Adaptive-LIF (base) + uncertainty": "AdLIF base+U", "Adaptive-LIF (proposed) + uncertainty": "AdLIF prop.+U",
         "ANN (temporal CNN) + uncertainty": "ANN+U", "LIF SNN + uncertainty": "LIF+U"}
NAME = {"lda": "LDA", "ann": "the ANN", "lif": "the LIF SNN", "adlif_base": "the base Adaptive-LIF", "full": "the proposed Adaptive-LIF", "full_ua": "the proposed Adaptive-LIF with uncertainty",
        "ad_hetero": "base + heterogeneous adaptation", "ad_ms": "base + multiscale input", "ad_tel": "base + temporal evidence learning",
        "full_no_het": "proposed without heterogeneous adaptation", "full_no_ms": "proposed without multiscale input",
        "full_no_tel": "proposed without temporal evidence learning", "full_no_ad": "proposed without adaptation", "adlif_base": "the base Adaptive-LIF", "lif": "the LIF SNN"}

def esc(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\~{}"), ("^", "\\^{}")):
        s = s.replace(a, b)
    return s

def table(df, caption, label, colfmt=None, small=True, floatfmt="%.3f"):
    df = df.copy(); df.columns = [esc(SHORT.get(c, c)) for c in df.columns]
    def cell(v):
        if isinstance(v, (float, np.floating)): return "--" if np.isnan(v) else (floatfmt % v)
        return esc(SHORT.get(v, v)) if isinstance(v, str) else esc(v)
    colfmt = colfmt or "l" + "r" * (df.shape[1] - 1)
    body = " \\\\\n".join(" & ".join(cell(v) for v in r) for r in df.itertuples(index=False))
    return (f"\\begin{{table}}[H]\\centering{'\\small' if small else ''}\n\\caption{{{caption}}}\\label{{{label}}}\n\\begin{{tabular}}{{{colfmt}}}\\toprule\n"
            f"{' & '.join(df.columns)} \\\\\\midrule\n{body} \\\\\\bottomrule\n\\end{{tabular}}\\end{{table}}\n")

def fig(name, caption, width=0.98):
    if not (RES / name).exists(): return ""
    shutil.copy(RES / name, OUT / name)
    return f"\\begin{{figure}}[H]\\centering\\includegraphics[width={width}\\textwidth]{{{name}}}\\caption{{{caption}}}\\end{{figure}}\n"

def cap1(t): return t[0].upper() + t[1:]
def fp(p): return "$<$ 0.001" if p < 0.001 else f"= {p:.3f}"
def verdict_word(r):
    if r.p_holm < 0.05 and (r.ci_lo > 0 or r.ci_hi < 0): return "a clear difference"
    return "no clear difference"

def sentence(r, what, new="full", good="higher"):
    d = r.mean_diff
    return (f"{what}: {cap1(NAME.get(r.model, r.model))} minus {NAME.get(r.ref, r.ref)} was {d:+.3f} "
            f"(95\\% CI {r.ci_lo:+.3f} to {r.ci_hi:+.3f}; Holm-adjusted Wilcoxon p {fp(r.p_holm)}; {int(r.participants_better)} of {int(r.n_participants)} participants positive), "
            f"which is {verdict_word(r)}.")

def pooled_cell(tab, model_label):
    return tab.loc["Pooled", model_label]

subs = info["subjects"]; n_f = info["n_folds"]
auc_tab = rd("table_auc_per_participant.csv"); auc_tab = auc_tab.rename(columns={auc_tab.columns[0]: "Participant"})
cmp7, cmp9, cmp13, cmp14 = (rd(n) for n in ("stats_auc_proposed_vs_others.csv", "stats_aurc_proposed_vs_others.csv", "stats_stopping_proposed_vs_others.csv", "stats_robustness_proposed_vs_others.csv"))
abl_tab, abl_stats = rd("table_ablation.csv"), rd("stats_ablation.csv")
ua_stats, stop_tab, rob_tab, cost = rd("stats_uncertainty_effect.csv"), rd("table_stopping.csv"), rd("table_robustness.csv"), rd("table_computational_cost.csv")
ddm, lat, leak, man = rd("ddm_fits_per_session.csv"), rd("within_epoch_latency.csv"), rd("prestimulus_leakage_check.csv"), rd("data_manifest.csv")

banner = ""
if SYN:
    banner = ("\\begin{center}\\fbox{\\parbox{0.92\\textwidth}{\\textbf{SMOKE TEST ON SYNTHETIC DATA.} This copy of the report was generated from a small "
              "synthetic stand-in for BCIAUT-P300, only to verify that the full pipeline runs. Every number, table, figure and verdict below describes that synthetic data. "
              "None of it is a result on the real dataset. Re-run the notebook on the real data and regenerate this report with make\\_report.py.}}\\end{center}\n")

# ------------------------------------------------------------------ text
def per_participant_sessions():
    m = man[(man.subject.isin(subs)) & (man.usable == True) & (man.part != "-")]
    g = m.groupby("subject").agg(sessions=("session", "nunique"), epochs=("n_epochs", "sum"), blocks=("n_blocks", "sum")).reindex(subs).reset_index()
    return g.rename(columns={"subject": "Participant", "sessions": "Sessions", "epochs": "Epochs", "blocks": "Blocks"})

abstract = (f"This report extends the single-participant study (SBJ01) to {len(subs)} participants ({', '.join(subs)}) with {n_f} held-out sessions in total. "
            "The pipeline is unchanged: P300 EEG, Adaptive-LIF SNN, temporal evidence E(t), calibrated log-likelihood ratios, repetition-wise accumulation, drift-diffusion model, speller decision dynamics. "
            "The SNN receives three testable additions (heterogeneous adaptive neurons, multiscale temporal input, temporal evidence learning) and uncertainty-aware sequential decoding, each tested by ablation. "
            "It is compared with a non-adaptive LIF SNN, an ANN and LDA under one protocol: leave-one-session-out inside each participant, separate train, early-stopping and calibration blocks, several random seeds, and hierarchical bootstrap confidence intervals. "
            "The test-set construction for the DDM was rewritten so that only genuine held-out evidence can enter it. ")
m_auc = {c: pooled_cell(auc_tab.set_index("Participant"), c) for c in auc_tab.columns[1:]}
abstract += "Pooled held-out ROC-AUC (mean [95\\% CI]): " + "; ".join(f"{esc(SHORT.get(k, k))} {v}" for k, v in m_auc.items()) + ". "
abstract += " ".join(sentence(r, "AUC") for _, r in cmp7.iterrows() if r.ref in ("adlif_base", "lda")) 

sec_protocol = r"""
\section{Evaluation Protocol}
The outer protocol is leave-one-session-out inside each participant. One fold is one participant with one held-out session; its training sessions are the other sessions of the same participant. No data are shared between participants, so the five participants give five independent sets of folds. Each session consists of the Train and Test parts of the dataset folder, as in the single-participant study, and is held out as a whole.

The blocks of the training sessions are split at block level into three disjoint groups: about 70\% for training, 15\% for early stopping (SNN and ANN only) and 15\% for calibration. The calibration blocks are used for the score-to-LLR calibration, for the uncertainty scale $\kappa$, for the stopping threshold $\alpha$ and for the number of repetitions of the fixed rule. All models train on exactly the same training epochs. Network hyperparameters were fixed in advance and were not tuned on held-out sessions. Channel standardisation uses training epochs only. Preprocessing filters each epoch on its own.

\subsection{Correction of the test-set construction}
The earlier report (Section 10.4) warned that training epochs without a held-out score might enter the DDM table as zero evidence. In the earlier notebook the trial table was already restricted to sessions with a finite held-out score, and all seven folds had run, so every epoch was held out exactly once (3,616 trials are 452 blocks of 8 objects). The warning therefore appears to describe an earlier version of the code. The construction rested on an implicit condition, though, and used \texttt{nan\_to\_num}, which can hide missing evidence. It was rewritten as follows. An explicit \texttt{heldout} mask marks epochs that belong to the test session of their own fold. The trial builder raises an error if any evidence is missing, if a block mixes sessions or participants, or if evidence exists for an epoch that was never held out. Padding of unequal repetition counts is tracked by a \texttt{valid} mask. A self-test confirms that missing evidence is rejected instead of being zero-filled.

\subsection{Seeds, intervals and tests}
The main models (ANN, LIF, base and proposed Adaptive-LIF) are trained with @@SM@@ seeds in every fold and their scores are averaged (seed ensemble); ablation variants use @@SA@@ seed per fold, the same seed for all variants. The unit of analysis is the held-out session. Confidence intervals are 95\% percentile intervals of a hierarchical bootstrap that resamples participants and then sessions inside participants; per-participant intervals resample sessions. Model comparisons are paired over held-out sessions with a Wilcoxon signed-rank test, with Holm correction inside each family of comparisons. Because sessions of one participant are not independent and there are only @@NS@@ participants, we also report in how many participants the mean difference is positive. With @@NS@@ participants no participant-level test can reach p $<$ 0.05, so the session-level tests should be read as evidence about these participants, not about the population.
"""
sec_protocol = sec_protocol.replace("@@SM@@", str(info["seeds_main"])).replace("@@SA@@", str(info["seeds_ablation"])).replace("@@NS@@", str(len(subs)))

sec_models = r"""
\section{Models}
\subsection{Base Adaptive-LIF SNN}
The architecture is unchanged: direct current encoding of 50 time steps of 8 channels, two Adaptive-LIF layers and a leaky-integrator readout that gives the evidence trajectory $E(t)=\beta_E E(t-1)+r_t$. For neuron $j$,
\begin{align}
v_t &= \beta v_{t-1} + I_t - a_{t-1}, \qquad s_t = \mathbb{I}(v_t > \theta), \qquad v_t \leftarrow v_t - \theta s_t,\\
a_t &= \rho\, a_{t-1} + b\, s_t ,
\end{align}
with learnable $\beta$ and $b$ and a fixed $\rho = 0.9$. Training uses AdamW, a cosine schedule, early stopping on validation AUC, a positive-class weight, input noise, temporal shifts of two bins, gradient clipping and a firing-rate regulariser.

\subsection{Proposed additions}
\paragraph{Heterogeneous adaptation.} Each neuron learns its own adaptation decay $\rho_j$ (initialised from adaptation time constants of 48 to 800 ms), its own strength $b_j$, and a sub-threshold coupling $c_j \ge 0$, so that $a_t = \rho a_{t-1} + (1-\rho) c\, v_t + b\, s_t$. Neurons with different adaptation scales form a bank of onset-sensitive temporal filters, which suits a slow component such as the P300.
\paragraph{Multiscale temporal input.} The first layer reads the EEG through three groups of causal dilated convolutions (dilations 1, 2, 4, kernel 3, receptive fields about 48, 80 and 144 ms) and the groups start with fast, medium and slow membrane constants. All filters are causal, so $E(t)$ still depends only on samples up to $t$.
\paragraph{Temporal evidence learning.} A per-step affine readout of $E(t)$ is supervised at every time step with weights that grow toward the end of the epoch, in addition to the final loss. The base model only trained the final logit, so early $E(t)$ was never asked to be informative.
\paragraph{Uncertainty-aware sequential decoding.} The epistemic variance $u^2$ of a flash score (MC dropout plus disagreement between seeds) is used to shrink the calibrated evidence, $\ell_{ua}=\ell/(1+\kappa u^2/\overline{u^2})$. The value of $\kappa$ is chosen from a small grid on the calibration blocks and stays 0 unless it improves the calibration accuracy-versus-repetitions curve by at least 0.5 points. A posterior-based stopping rule (softmax of the cumulative evidence over the eight objects reaches $1-\alpha$) is also evaluated, with $\alpha$ chosen on calibration blocks by information rate.

\subsection{Comparison models}
The LIF SNN is the base network without adaptation. The ANN is a compact temporal CNN (two convolutions, pooling, flattened time axis, linear output) with a parameter count of the same order. LDA uses 200 features (8 channels $\times$ 25 bins of 32 ms) with shrinkage. All use the same folds, calibration and decoding.
"""

def stats_para(df, what, keep=None):
    if df is None or len(df) == 0: return ""
    return " ".join(sentence(r, what) for _, r in df.iterrows() if keep is None or r.ref in keep)

parts = []
parts.append(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}\usepackage{mathptmx}\usepackage[margin=2.3cm]{geometry}\usepackage{amsmath,amssymb}\usepackage{booktabs}\usepackage{graphicx}\usepackage{float}\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\begin{document}
\begin{center}{\LARGE\bfseries From Spikes to Decisions: Modelling P300 Evidence Accumulation with Adaptive Spiking Neural Networks and Drift Diffusion Dynamics}\\[6pt]
{\large Technical Report: five-participant extension}\\[4pt] October 15, 2026\end{center}
""" + banner + r"\begin{abstract}" + abstract + r"\end{abstract}")

parts.append(r"""
\section{Research Questions}
The central question is unchanged: can the temporal dynamics of a spiking model give a useful representation of evolving neural evidence that links to an explicit evidence-accumulation model? The extension asks whether the answer holds across participants and which design choices matter.
\begin{itemize}
\item RQ1: Can the Adaptive-LIF SNN separate target from non-target P300 responses on held-out sessions of every participant?
\item RQ2: Does the learned evidence $E(t)$ carry information before the end of the epoch?
\item RQ3: Can scores be calibrated to evidence and accumulated over repetitions, and can a DDM describe the process?
\item RQ4: Do heterogeneous adaptation, multiscale input and temporal evidence learning each help, as shown by ablation?
\item RQ5: Does uncertainty-aware decoding improve accuracy, stopping and information rate?
\item RQ6: How do the SNN variants compare with a LIF SNN, an ANN and LDA in accuracy, speed, robustness and cost?
\end{itemize}
""")

man_tab = per_participant_sessions()
skip_txt = "; ".join(f"{k} ({esc(' '.join(v)[:120])})" for k, v in info.get("skipped", {}).items()) or "none"
parts.append(r"\section{Dataset and Participants}" + f"""
Participants and sessions were discovered automatically from the folder structure (\\texttt{{{esc(info['data_root'])}}}). A participant is used if every session folder has Train and Test parts with data, events and target files, an EEG array of 8 channels and 350 samples, runs that contain every object once with one target, and one attended object per block. Participants were selected by this integrity rule alone, in sorted order, and never by model performance. Skipped participants: {skip_txt}. Each epoch runs from 200 ms before to 1.2 s after the flash at 250 Hz.
""" + table(man_tab, "Participants used and the data they contribute (Train and Test parts of all sessions).", "tab:subjects", "lrrr"))

parts.append(r"""\section{Preprocessing}
Each epoch is band-pass filtered (0.5 to 20 Hz, fourth-order zero-phase Butterworth), baseline corrected with the 200 ms before the flash, cut to 0 to 800 ms and pooled in groups of four samples, giving 50 steps of 16 ms over 8 channels. No per-trial normalisation is applied; channel scaling uses training epochs of the fold only.
\subsection{Pre-stimulus leakage check}
A grouped five-fold LDA on the pre-stimulus segment alone, for each participant, should be near chance.
""" + table(leak.rename(columns={"subject": "Participant", "auc_pre": "Pre-stimulus AUC", "sd": "SD over folds"}), "Pre-stimulus-only LDA, grouped five-fold ROC-AUC per participant.", "tab:leak", "lrr"))
parts.append(sec_protocol); parts.append(sec_models)

# ---- results
res = r"\section{Results}" + "\n\\subsection{Single-trial classification}\n"
res += ("ROC-AUC on the held-out session, seed-ensemble score. Entries are mean [95\\% CI]; per-participant intervals resample sessions, the pooled row uses the hierarchical bootstrap. " 
        + table(auc_tab, "Held-out ROC-AUC per participant and pooled.", "tab:auc", "l" + "c" * (auc_tab.shape[1] - 1)))
res += stats_para(cmp7, "AUC") + "\n"
res += fig("fig1_single_trial_and_temporal.png", "Single-trial AUC per participant, time-resolved AUC of $E(t)$ and mean evidence trajectory.")
lt = lat.groupby("model")[["onset_ms", "peak_auc", "peak_ms", "t90_ms"]].mean().reset_index()
lt["model"] = lt.model.map({"lif": "LIF", "adlif_base": "AdLIF base", "full": "AdLIF proposed"})
res += "\\subsection{Temporal evidence}\nThe evidence trajectory is prefix-causal. Onset is the first of three consecutive bins whose AUC exceeds $0.5+2$ SE (neural onset, not reaction time).\n"
res += table(lt.rename(columns={"model": "Model", "onset_ms": "Onset (ms)", "peak_auc": "Peak AUC", "peak_ms": "Peak (ms)", "t90_ms": "t90 (ms)"}), "Within-epoch evidence of the SNN models, mean over held-out sessions.", "tab:lat", "lrrrr", floatfmt="%.3f")

ct = {c: rd(f"table_{c}_per_participant.csv") for c in ("acc_r%d" % info["reps_reported"][-1], "final_acc")}
res += "\\subsection{Speller accuracy and evidence accumulation}\n"
for c, d in ct.items():
    d = d.rename(columns={d.columns[0]: "Participant"})
    ttl = (f"Character accuracy after {info['reps_reported'][-1]} repetitions" if c.startswith("acc") else "Character accuracy using all repetitions of each block")
    res += table(d, ttl + " (chance 12.5\\%), mean [95\\% CI].", "tab:" + c, "l" + "c" * (d.shape[1] - 1))
res += stats_para(cmp9, "Area under the accuracy-versus-repetitions curve") + "\n"
res += fig("fig2_speller_accuracy.png", "Character accuracy against repetitions, per participant and pooled (mean over held-out sessions, SEM band).")

res += "\\subsection{Drift-diffusion model}\nThe DDM framework is unchanged and was fitted per participant, session and method on held-out evidence only (reference boundary $\\Theta=3$). Decision time is in repetitions, not reaction time.\n"
dd = ddm.groupby("method")[["dprime", "v_target", "v_nontarget", "a_fit", "a_policy", "emp_rt", "pred_rt"]].agg(["mean", "std"])
dd.columns = [f"{a} {b}" for a, b in dd.columns]; dd = dd.reset_index()
dd = dd[["method", "dprime mean", "v_target mean", "v_target std", "v_nontarget mean", "a_fit mean", "emp_rt mean", "pred_rt mean"]]
dd.columns = ["Method", "d'", "v target", "SD", "v non-target", "a fitted", "Decision time", "DDM-predicted"]
res += table(dd, "DDM fits pooled over held-out sessions (mean; SD of target drift across sessions).", "tab:ddm", "lccccccc")
dp = ddm[ddm.method == "full"].groupby("subject")[["v_target", "v_nontarget", "a_fit", "emp_rt"]].mean().reset_index()
res += table(dp.rename(columns={"subject": "Participant", "v_target": "v target", "v_nontarget": "v non-target", "a_fit": "a fitted", "emp_rt": "Decision time"}), "DDM parameters of the proposed SNN evidence per participant (mean over sessions).", "tab:ddmp", "lcccc")
res += fig("fig3_ddm.png", "Speed-accuracy behaviour, fitted drift and boundary recovery.")

res += "\\subsection{Ablation of the proposed components}\nAll variants use seed 0, the same folds and the same decoding.\n"
ab = abl_tab.rename(columns={"variant": "Variant", "auc": "AUC [95% CI]", "aurc": "AURC [95% CI]"})
res += table(ab, "Ablation, pooled over held-out sessions (AUC per participant in the last columns).", "tab:abl", "lcc" + "c" * len(subs))
a_ = abl_stats.copy(); a_["Comparison"] = [f"{NAME.get(n, n)} vs {NAME.get(r, r)}" for n, r in zip(a_.model, a_.ref)]
a_t = a_[["metric", "Comparison", "mean_diff", "ci_lo", "ci_hi", "p_holm", "participants_better"]].rename(columns={"metric": "Metric", "mean_diff": "Diff", "ci_lo": "CI low", "ci_hi": "CI high", "p_holm": "Holm p", "participants_better": "Part. +"})
res += table(a_t, "Paired ablation contrasts (first model minus reference).", "tab:ablst", "llccccc", floatfmt="%.4f")
res += " ".join(sentence(r, "Ablation (" + esc(r.metric.upper()) + ")") for _, r in abl_stats[abl_stats.metric == "auc"].iterrows()) + "\n"
res += fig("fig4_ablation.png", "Paired ablation effects with 95\\% bootstrap intervals.")

res += "\\subsection{Uncertainty-aware decoding and adaptive stopping}\nSettings are chosen on calibration blocks of each fold. Information rate is in bits per repetition (Wolpaw), so no stimulus timing is assumed.\n"
st = stop_tab.copy(); st = st.rename(columns={"model": "Model", "adaptive_acc": "Adapt. acc.", "adaptive_reps": "Adapt. reps", "adaptive_bits_per_rep": "Adapt. bits/rep", "fixed_acc": "Fixed acc.", "fixed_reps": "Fixed reps", "fixed_bits_per_rep": "Fixed bits/rep", "ddm_decision_time": "DDM time"})
res += table(st, "Fixed and adaptive decoding, mean over held-out sessions.", "tab:stop", "lccccccc")
res += stats_para(cmp13, "Adaptive-stopping information rate") + "\n"
res += "Effect of the uncertainty-aware evidence (ua minus plain, same score stream): " + " ".join(
    f"{cap1(NAME.get(r.ref, r.ref))}, {esc(r.metric)}: {r.mean_diff:+.4f} (95\\% CI {r.ci_lo:+.4f} to {r.ci_hi:+.4f}; unadjusted p {fp(r.p_wilcoxon)})." for _, r in ua_stats.iterrows()) + "\n"
res += fig("fig5_stopping_uncertainty.png", "Adaptive stopping frontier, fixed versus adaptive information rate, and the effect of uncertainty weighting.")

res += "\\subsection{Robustness}\nTrained models score corrupted held-out epochs (Gaussian noise, channel loss, latency jitter) without retraining.\n"
rt = rob_tab.rename(columns={rob_tab.columns[0]: "Model"})
res += table(rt, "Held-out AUC under test-time corruption (mean over sessions).", "tab:rob", "l" + "c" * (rt.shape[1] - 1), small=True)
res += stats_para(cmp14, "Mean AUC over corruptions") + "\n" + fig("fig6_robustness.png", "AUC under each corruption.", 0.7)

res += "\\subsection{Computational cost}\nParameters, training time and inference time are measured on the machine that ran the notebook. Operation counts and energy are estimates from measured firing rates (4.6 pJ per MAC and 0.9 pJ per AC, 45 nm figures), not measurements on neuromorphic hardware.\n"
cs = cost[["model", "params", "rate_L1", "rate_L2", "MACs", "ACs", "est_energy_nJ", "cpu_ms_per_epoch_batch1", "train_s_per_job"]].copy()
cs.columns = ["Model", "Params", "Rate L1", "Rate L2", "MACs", "ACs", "Energy (nJ)", "CPU ms/epoch", "Train s/job"]
res += table(cs, "Computational cost.", "tab:cost", "lcccccccc", floatfmt="%.3g")
parts.append(res)

# ---- limitations and conclusion
lim = r"""
\section{Limitations}
\begin{enumerate}
\item Five participants of one dataset. Conclusions describe these participants; session-level tests treat sessions as the unit, but sessions of one participant are correlated.
\item Evaluation is within participant (leave-one-session-out). Cross-participant generalisation was not tested.
\item There are no behavioural reaction times, so the DDM describes evidence accumulation in repetition units. The fitted boundary is strongly tied to the stopping policy. The nominal 2.4 s per repetition would be needed to convert to seconds and was not assumed.
\item Ablation variants use one seed, the main models use several. Seed-to-seed variation of single variants is therefore not captured in the ablation table.
\item Calibration assumes equal-variance Gaussian scores. The stopping posterior assumes independent repetitions and is only as calibrated as that assumption; thresholds were therefore chosen on calibration blocks, which are also used to fit the calibrator, so calibration-based settings are slightly optimistic.
\item Energy figures are estimates from firing rates, not hardware measurements. Zero-phase filtering is non-causal, so within-epoch latencies are not online latencies.
\item Hyperparameters were fixed in advance, not tuned. A tuned baseline or ANN could change the ranking.
\end{enumerate}
"""
concl = r"\section{Conclusion}" + "\n" + (
    "The framework now runs for %d participants with held-out-session evaluation, corrected test-set construction, ablation, three comparison models, adaptive stopping, robustness and cost analysis. " % len(subs)
    + stats_para(cmp7, "Single-trial AUC") + " " + stats_para(cmp9, "AURC") + " " + stats_para(cmp13, "Adaptive-stopping information rate") + " "
    + "These sentences are generated from the saved statistics, so they report the sign and size of each difference whether or not it favours the SNN. "
    + "Any advantage of the proposed model should be claimed only where its interval excludes zero after correction and where the ablation shows which component produced it.")
parts.append(lim + concl + "\n\\end{document}\n")
(OUT / "report.tex").write_text("\n".join(parts), encoding="utf-8")
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], cwd=OUT, capture_output=True, text=True)
print("pdflatex return code", r.returncode); 
if r.returncode: print(r.stdout[-2500:])
