"""Small helpers used by Code_Final_V_5P: statistics, perturbations, cost accounting.
Kept out of the notebook so the notebook stays focused on the pipeline."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

# ----------------------------------------------------------------------------- statistics
def two_level_bootstrap(df, value, group="subject", n_boot=4000, seed=0):
    """95% CI of the grand mean with a hierarchical bootstrap: resample participants, then
    sessions inside each drawn participant. df has one row per held-out session."""
    rng = np.random.default_rng(seed)
    groups = {g: d[value].to_numpy(float) for g, d in df.groupby(group)}
    keys = list(groups); means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(len(keys), len(keys), replace=True)
        means[b] = np.mean([rng.choice(groups[keys[i]], len(groups[keys[i]]), replace=True).mean() for i in draw])
    return float(df[value].mean()), *np.percentile(means, [2.5, 97.5])

def session_bootstrap(x, n_boot=4000, seed=0):
    """95% CI of a mean over the sessions of ONE participant."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return float(x.mean()), *np.percentile(m, [2.5, 97.5])

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); out = np.empty_like(p); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(1.0, run)
    return out

def paired_table(df, value, ref, models, unit=("subject", "session"), group="subject", n_boot=2000):
    """Each model against `ref`: median paired difference, hierarchical-bootstrap CI of the mean difference,
    Wilcoxon over held-out sessions (Holm-corrected over the listed models), and how many participants favour the model."""
    wide = df.pivot_table(index=list(unit), columns="model", values=value).dropna(subset=[ref])
    rows = []
    for m in models:
        if m == ref or m not in wide: continue
        d = (wide[m] - wide[ref]).dropna()
        dd = d.reset_index(); dd.columns = list(unit) + ["diff"]
        mean, lo, hi = two_level_bootstrap(dd, "diff", group, n_boot)
        try: p = wilcoxon(d.values).pvalue if np.any(d.values != 0) else 1.0
        except ValueError: p = 1.0
        per_p = dd.groupby(group)["diff"].mean()
        rows.append(dict(model=m, ref=ref, n_sessions=len(d), mean_diff=mean, ci_lo=lo, ci_hi=hi,
                         median_diff=float(np.median(d.values)), p_wilcoxon=p,
                         participants_better=int((per_p > 0).sum()), n_participants=len(per_p)))
    out = pd.DataFrame(rows)
    if len(out): out["p_holm"] = holm(out.p_wilcoxon.values)
    return out

def friedman(df, value, models, unit=("subject", "session")):
    wide = df.pivot_table(index=list(unit), columns="model", values=value)[models].dropna()
    if len(wide) < 3 or len(models) < 3: return np.nan, np.nan
    s = friedmanchisquare(*[wide[m].values for m in models]); return float(s.statistic), float(s.pvalue)

# ----------------------------------------------------------------------------- perturbations
def make_perturbations(n_ch, seed=0):
    """Test-time corruptions applied to the standardised held-out input (models are NOT retrained)."""
    rng = np.random.default_rng(seed)
    P = {}
    for s in (0.5, 1.0, 2.0): P[f"noise_{s}"] = ("noise", s)
    for k in (1, 2): P[f"drop_{k}ch"] = ("drop", rng.choice(n_ch, k, replace=False).tolist())
    for j in (2, 4): P[f"jitter_{j}bins"] = ("jitter", j)
    return P

def apply_perturbation(X, spec, seed=0):
    """X: (N, T, C) numpy array, already standardised."""
    kind, arg = spec; rng = np.random.default_rng(seed); X = X.copy()
    if kind == "noise": X += arg * rng.standard_normal(X.shape).astype(X.dtype)
    elif kind == "drop": X[:, :, arg] = 0.0
    elif kind == "jitter":
        sh = rng.choice([-arg, arg], len(X))                         # each epoch is shifted by +-arg bins (circular)
        for s in (-arg, arg): X[sh == s] = np.roll(X[sh == s], s, axis=1)
    return X

# ----------------------------------------------------------------------------- cost accounting
E_MAC_PJ, E_AC_PJ = 4.6, 0.9      # 32-bit float multiply-accumulate vs accumulate, 45 nm (Horowitz, ISSCC 2014)

def energy_pj(macs, acs): return macs * E_MAC_PJ + acs * E_AC_PJ

def family_vs(df, value, new, refs, **kw):
    """The proposed model `new` against each reference in `refs` (diff = new - ref); Holm over this family."""
    parts = [paired_table(df, value, ref=r, models=[new], **kw) for r in refs if r != new]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True) if parts else pd.DataFrame()
    if len(out): out["p_holm"] = holm(out.p_wilcoxon.values)
    return out
