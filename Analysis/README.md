<<<<<<< HEAD
# IITKGP_IEEE_BCI
=======
# P300 BCI: Spiking Neural Networks and Drift Diffusion Modeling

This project explores P300-based brain–computer interface (BCI) decoding using an Adaptive Leaky Integrate-and-Fire (Adaptive-LIF) spiking neural network (SNN). It investigates how time-resolved neural evidence can be calibrated, accumulated across repeated stimulus presentations, and described using a drift-diffusion model (DDM).

**Status:** Research prototype. The current SNN performs below the LDA baseline, and the DDM results are preliminary.

## Pipeline

P300 EEG → Adaptive-LIF SNN → temporal evidence → evidence calibration → accumulation across repetitions → DDM

The current analysis uses seven sessions from participant SBJ01 in the BCIAUT-P300 dataset, evaluated with leave-one-session-out validation.

## Dataset and preprocessing

- Dataset: BCIAUT-P300
- Participant: SBJ01
- Sessions: 7
- EEG channels: 8
- Sampling rate: 250 Hz
- Band-pass filter: fourth-order, zero-phase Butterworth, 0.5–20 Hz
- Baseline interval: −200 to 0 ms
- Post-stimulus window: 0–800 ms
- Temporal pooling: 4 samples per SNN time step
- SNN input: samples × 50 time steps × 8 channels

The pre-stimulus leakage check reports an LDA AUC of 0.509 ± 0.016, close to chance.

Use the original dataset documentation for access, citation, and redistribution terms. Check the source and license of the local BCIAUT-P300.pdf before sharing it.

## Model and evidence accumulation

The SNN architecture is:

8 input channels → 64 Adaptive-LIF units → 64 Adaptive-LIF units → leaky-integrator readout

The model uses direct current encoding of scaled EEG signals and is trained with AdamW, a cosine learning-rate schedule, validation-based early stopping, class-weighted binary cross-entropy, input noise, small temporal shifts, gradient clipping, firing-rate regularization, and surrogate gradients.

The readout produces a time-resolved evidence trajectory. Scores are calibrated on validation blocks using an equal-variance Gaussian class-conditional approximation to obtain LLR-like evidence increments. These increments are accumulated across object repetitions.

The DDM models each repetition as one accumulation step. Its decision time is measured in repetitions, not human reaction time; trial-level button-press timestamps are not available in this analysis.

The current preprocessing uses zero-phase filtering, so the full pipeline is not strictly causal and is not yet suitable for claims of real-time latency.

## Results

### Single-trial discrimination

| Metric | Adaptive-LIF SNN | Shrinkage LDA |
| --- | ---: | ---: |
| Mean held-out ROC-AUC | 0.663 ± 0.022 | 0.781 ± 0.048 |

LDA outperformed the SNN in all seven held-out sessions. The report gives a paired Wilcoxon signed-rank test of p = 0.0156.

### Temporal evidence

- Peak mean temporal evidence AUC: 0.657 at approximately 568 ms
- Median evidence onset latency: approximately 472 ms
- Time to reach 90% of the above-chance peak: approximately 536 ms

These values describe neural-evidence timing, not behavioral reaction times.

### Eight-choice speller accuracy

| Repetitions | SNN | LDA |
| ---: | ---: | ---: |
| 1 | 22.6% | 40.3% |
| 2 | 30.5% | 52.0% |
| 3 | 35.2% | 59.1% |
| 5 | 43.0% | 71.4% |
| 10 | 58.3% | 80.5% |
| 15 | 91.7% | 100.0% |
| 18 | 100.0% | 100.0% |

Chance level for eight choices is 12.5%. LDA provides stronger decoding at the same number of repetitions in this experiment.

### Preliminary DDM estimates

At a reference threshold of 3, the report gives mean target drift of 0.289 ± 0.103, mean non-target drift of −0.251 ± 0.053, a fitted boundary of 5.10, and a model-derived decision time of 7.82 repetitions.

These estimates are preliminary. The current DDM table may include training-session records with missing held-out scores filled with zero evidence. Rebuild the table using held-out test epochs only and rerun the fits before treating these values as final or publication-ready.

## Repository contents

- `Analysis/`: linked Git submodule containing the analysis code
- `BCIAUT-P300.pdf`: dataset-related PDF
- `PApers/`: project paper and background reading
- `results_snn_ddm/`: reported metrics, model checkpoints, evidence arrays, and figures

Key result files:

- `ddm_fits_per_session.csv`
- `single_trial_auc.csv`
- `speller_accuracy_vs_repetitions.csv`
- `speed_accuracy_sweep.csv`
- `scores_and_evidence.npz`
- `snn_fold*_seed0.pt`

Figures:

- [Single-trial and temporal evidence](results_snn_ddm/fig1_single_trial_and_temporal.png)
- [Speller accuracy across repetitions](results_snn_ddm/fig2_speller_accuracy.png)
- [DDM results (preliminary)](results_snn_ddm/fig3_ddm.png)

## Clone the repository

Clone the repository and its submodule:

```bash
git clone --recurse-submodules https://github.com/Abhikumar199/IITKGP_IEEE_BCI.git
```

If you already cloned the repository, initialize the submodule with:

```bash
git submodule update --init --recursive
```

The Analysis folder is currently a submodule pointer, not a regular folder tracked directly in this repository. Access to the linked repository may be required.

## Reproducing the analysis

1. Clone the repository with its submodule.
2. Obtain the BCIAUT-P300 dataset from its official source and arrange the files as expected by the analysis code.
3. Install the dependencies used by the notebook or scripts. A root-level requirements file is not currently documented; check the analysis code for imports and environment details.
4. Run preprocessing and leave-one-session-out evaluation.
5. Reproduce the LDA baseline and temporal evidence analysis.
6. Rebuild the DDM table using held-out test epochs only, then refit the DDM.
7. Evaluate multiple random seeds. The current reported SNN result uses one seed.

Exact execution commands depend on the code in the Analysis submodule and the local dataset layout.

## Limitations

- Results currently cover seven sessions from one participant; cross-subject generalization has not been established.
- The SNN performs below the LDA baseline in the reported experiment.
- DDM estimates require recomputation using held-out test epochs only.
- Results from a single random seed do not establish robustness.
- Zero-phase filtering is non-causal.
- LLR calibration relies on an equal-variance Gaussian approximation.
- DDM decision time is measured in repetitions, not human reaction time.

## References and citation

The `PApers/To Read/` directory contains background papers. When using the dataset or discussing prior methods, cite the original publications and follow the dataset's terms of use.

If you use this repository, cite the project repository and associated technical report where appropriate. Add a formal publication citation only when publication details are available.

## Acknowledgement

This repository documents a research-stage investigation of spiking neural representations and sequential decision modeling for P300 BCI analysis. Interpret the results in light of the limitations above.
>>>>>>> 1bde18de14c674226f5663998ade990b68de7cd2
