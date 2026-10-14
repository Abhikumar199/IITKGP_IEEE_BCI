# From Spikes to Decisions: P300 Evidence Accumulation with Adaptive SNNs and Drift Diffusion Dynamics

A research implementation for analyzing P300-based brain–computer interface (BCI) signals using an Adaptive-Leaky Integrate-and-Fire (Adaptive-LIF) spiking neural network (SNN) and a discrete-time drift-diffusion model (DDM).

The project goes beyond static target/non-target classification: it studies whether a spiking model can produce a temporally resolved neural-evidence trajectory, calibrate this evidence into approximate log-likelihood-ratio (LLR) increments, and accumulate evidence across repeated presentations in a P300 speller task.

> **Status:** Research prototype / technical report (October 2026). The reported SNN results do **not** outperform the matched LDA baseline. The DDM estimates in the current report are preliminary and need to be recomputed after restricting the DDM table to held-out test epochs only.

## Overview

**Pipeline**

`P300 EEG → Adaptive-LIF SNN → temporal evidence E(t) → calibrated LLR → repetition-wise accumulation → DDM`

The implementation uses the BCIAUT-P300 dataset and, in the current report, evaluates seven sessions from participant SBJ01 with leave-one-session-out validation.

## Research Questions

- Can an Adaptive-LIF SNN distinguish target and non-target P300 EEG responses when an entire session is held out?
- Does the model provide a time-resolved evidence signal before the full EEG epoch is available?
- Can classifier scores be calibrated into approximate evidence increments and accumulated across repeated presentations of an object?
- Can a discrete-time DDM summarize the resulting accumulation using target/non-target drift and a decision boundary?
- How do the resulting sequential decoding curves compare with a conventional LDA evidence stream?

## Dataset and Preprocessing

The analysis uses the **BCIAUT-P300** dataset. The current implementation described in the technical report uses seven available sessions from participant **SBJ01**, with eight EEG channels and repeated presentations of eight objects.

Preprocessing and input representation:

- Sampling frequency: **250 Hz**
- Fourth-order, zero-phase Butterworth band-pass filter: **0.5–20 Hz**
- Baseline correction using the **−200 to 0 ms** pre-stimulus interval
- Post-stimulus analysis window: **0–800 ms**
- Temporal pooling: four samples per SNN time step
- SNN input shape: **samples × 50 time steps × 8 channels**
- Each SNN step represents approximately **16 ms**

The report's pre-stimulus leakage check gives an LDA AUC of **0.509 ± 0.016**, close to chance.

**Dataset access:** Consult the original BCIAUT-P300 data source and its usage terms before downloading or redistributing the data. A paper describing the dataset is referenced in the `PApers/` directory. The repository also contains a local PDF named `BCIAUT-P300.pdf`; check its source and licensing before redistribution.

## Model

The SNN uses direct current encoding of the scaled EEG waveform and the following architecture:

`8 input channels → 64 Adaptive-LIF units → 64 Adaptive-LIF units → leaky-integrator readout`

The readout produces a temporal evidence trajectory (E(t)). The model is trained with AdamW, a cosine learning-rate schedule, early stopping based on validation ROC-AUC, class-weighted binary cross-entropy, input noise, small temporal shifts, gradient clipping, and firing-rate regularization. Surrogate gradients are used to train the spiking layers.

The time-resolved evidence is computed from the observed temporal prefix so that an earlier evidence value does not use later post-stimulus samples. The current offline preprocessing uses zero-phase filtering, however, so the full pipeline is **not strictly causal for real-time deployment**.

## Evidence Calibration and DDM

SNN scores are calibrated on validation blocks using an equal-variance Gaussian class-conditional approximation to obtain LLR-like evidence increments. The increments are accumulated across repetitions of each object.

The DDM treats each object repetition as one discrete accumulation step, with class-dependent drift and absorbing decision boundaries. Its decision time is expressed in **repetitions**, not participant reaction time. The data used in this analysis do not provide trial-level button-press timestamps.

## Current Results

### Single-trial discrimination

| Metric | Adaptive-LIF SNN | Shrinkage LDA |
|---|---:|---:|
| Mean held-out ROC-AUC | 0.663 ± 0.022 | 0.781 ± 0.048 |

The matched LDA baseline outperformed the SNN in all seven held-out sessions; the report gives a paired Wilcoxon signed-rank test of **p = 0.0156**.

### Temporal evidence

- Peak mean temporal evidence AUC: **0.657 at approximately 568 ms**
- Median neural evidence onset latency: **approximately 472 ms**
- Mean evidence AUC reaches 90% of its above-chance peak at approximately **536 ms**

These are neural-evidence timing measures, not behavioral reaction times.

### Eight-way speller accuracy across repetitions

| Repetitions | SNN | LDA |
|---:|---:|---:|
| 1 | 22.6% | 40.3% |
| 2 | 30.5% | 52.0% |
| 3 | 35.2% | 59.1% |
| 5 | 43.0% | 71.4% |
| 10 | 58.3% | 80.5% |
| 15 | 91.7% | 100.0% |
| 18 | 100.0% | 100.0% |

Chance level for eight choices is **12.5%**. In this run, LDA provides stronger speller-level decoding at the same number of repetitions.

### Preliminary DDM estimates

At the reference threshold (Theta = 3), the current report gives mean target drift of **0.289 ± 0.103**, mean non-target drift of **−0.251 ± 0.053**, a fitted boundary of **5.10**, and a model-derived decision time of **7.82 repetitions**.

**Important:** These DDM numbers are labelled preliminary in the technical report. The current notebook can include training-session records with missing held-out scores filled with zero evidence. Rebuild the DDM trial table using held-out test epochs only and rerun the fits before treating these estimates as final results or publication-ready findings.

## Repository Structure

```text
IITKGP_IEEE_BCI/
├── Analysis/                  # Linked Git submodule; see note below
├── BCIAUT-P300.pdf             # Dataset-related PDF in repository
├── PApers/
│   ├── Abhishek_IEEE_EMBS.pdf
│   └── To Read/                 # Background papers and reading material
└── results_snn_ddm/
    ├── ddm_fits_per_session.csv
    ├── single_trial_auc.csv
    ├── speller_accuracy_vs_repetitions.csv
    ├── speed_accuracy_sweep.csv
    ├── scores_and_evidence.npz
    ├── snn_fold*_seed0.pt
    ├── fig1_single_trial_and_temporal.png
    ├── fig2_speller_accuracy.png
    └── fig3_ddm.png
```

The `Analysis/` entry is currently a **Git submodule pointer**, not a regular directory stored in this repository. To retrieve it when cloning, initialize submodules:

```bash
git clone --recurse-submodules https://github.com/Abhikumar199/IITKGP_IEEE_BCI.git
```

Or, after cloning:

```bash
git submodule update --init --recursive
```

If the submodule is private or unavailable to you, the linked content may not clone. The results folder includes selected outputs and figures; the current root tree does not itself show a top-level training notebook or environment specification.

## Reproducing the Analysis

The tracked outputs support inspection of the current reported results. To reproduce the full training and evaluation pipeline:

1. Clone the repository, including the `Analysis/` submodule.
2. Obtain the BCIAUT-P300 data from its official source and arrange it as expected by the analysis code.
3. Install the dependencies used by the notebook or scripts. No verified `requirements.txt` is currently documented at the repository root, so use the imports and environment notes in `Analysis/` to build a matching environment.
4. Run the preprocessing and leave-one-session-out experiments.
5. Reproduce the matched LDA baseline and the temporal evidence analysis.
6. Restrict the DDM construction stage to held-out test epochs only, then refit the DDM.
7. Repeat the outer evaluation with multiple random seeds. The report notes that its current executed SNN result used one seed; at least three seeds are planned for a more robust analysis.

The exact commands depend on the contents of the linked `Analysis/` submodule and the local dataset layout.

## Limitations

- **Single participant in the current implementation:** results use seven sessions from SBJ01; cross-subject generalization has not been established.
- **SNN below LDA:** the current SNN is not the best-performing classifier in this experiment.
- **Preliminary DDM table:** the zero-filled non-held-out-score issue must be fixed and the analysis rerun.
- **Single random seed:** robustness across seeds remains to be established.
- **Offline filtering:** zero-phase filtering is not strictly causal and limits claims about online latency.
- **Approximate calibration:** the LLR conversion assumes equal-variance Gaussian score distributions.
- **No behavioral reaction times:** DDM decision times are in repetition units and should not be described as human reaction times.

## Figures

The repository contains the currently reported result plots:

- [Single-trial and temporal evidence results](results_snn_ddm/fig1_single_trial_and_temporal.png)
- [Speller accuracy across repetitions](results_snn_ddm/fig2_speller_accuracy.png)
- [DDM results](results_snn_ddm/fig3_ddm.png) — preliminary; see the caveat above.

## References and Data

- **Dataset:** BCIAUT-P300. Please consult the original dataset documentation for citation and terms of use.
- Papers collected for project reading are in `PApers/To Read/`. Cite the original publications rather than the local PDF copies when writing up the research.

## Citation

If you use or refer to this repository, cite the project repository and the associated technical report where appropriate. A formal publication citation should only be added after publication details are available.

## Acknowledgement

This repository documents a research-stage implementation connecting spiking neural temporal representations with sequential decision modelling for P300 BCI analysis. Results and interpretations should be considered in light of the limitations stated above.
