# TWC upgrade package: what changed and why

## 1) Judgment on the current repo
The current repo is **correct for the ICC-workshop baseline scope**, not for the new TWC scope.

Why:
- the repo is explicitly scoped to **classical WMMSE baseline only**;
- user weights are effectively **equal/uniform**;
- there is **no deep unfolding** in the current code;
- FS protection is present, but the BS-tone mask is currently all-ones, so there is **no cognitive per-cell/per-tone band adaptation** yet.

## 2) What this package adds
This package adds a new additive package `src/fr3_twc/` plus run scripts, configs, and slurms.

### A. Proportional-fair benchmark family
Implemented benchmark variants:
- `ew_no_fs`: equal weights, no FS protection
- `ew_fs_soft`: equal weights, soft FS dual protection
- `edge_fs_soft`: static edge-biased weights, soft FS dual protection
- `pf_fs_soft`: proportional fairness (PF) + soft FS dual protection
- `pf_fs_hybrid`: PF + soft dual + hard nulling trigger
- `pf_fs_cognitive`: PF + soft FS protection + per-BS/per-tone cognitive masking

### B. Deep unfolding
Implemented trainable unrolled weighted-WMMSE:
- `du_pf_soft`
- `du_pf_cognitive`

Trainable per-layer parameters:
- damping
- BS-power dual step
- FS dual step

This preserves interpretability and keeps the unfolded model very close to the classical solver.

### C. Advanced protection / coexistence direction
Implemented now:
- topology-aware **cognitive tone masking** from FS coupling statistics;
- **hybrid nulling** mode on top of soft protection;
- **flat-vs-selective** diagnostics;
- **5G NR FER** post-processing through Sionna when available.

Recommended main TWC technical direction from here:
1. **Two-timescale protection**:
   - slow layer: sensing / incumbent-activity map / tone availability map;
   - fast layer: PF beamforming with instantaneous user channels.
2. **Robust angular protection** instead of only average scalar FS budgets:
   - angular radiated-power constraints over uncertainty sectors around each FS direction;
   - useful when FS angle or steering model is imperfect.
3. **Context-aware sensing-assisted protection**:
   - co-locate sensing with only a subset of BSs and feed a shared protection map to the beamformer.
4. **Unfolded low-complexity solver**:
   - keep the matrix-inverse-free / light-parameterized style so scaling remains practical.

A strong TWC story is therefore:
**PF baseline -> unfolded PF -> cognitive PF -> sensing-assisted robust PF**.

### D. Reviewer comments addressed here
This package directly addresses these comments:
- **user weights**: PF and non-uniform benchmarks are added;
- **deep unfolding behavior**: unfolded solvers and convergence plots are added;
- **flat vs selective channel concern**: selectivity suite and coherence-bandwidth diagnostics are added;
- **cognitive radio / different bands per cell**: per-BS tone masking is added;
- **larger arrays and more users**: scaling suite is added;
- **reference geometry / better visualization**: reference geometry figure is added;
- **practicality / rerun frequency**: runtime metrics are logged and all experiments are reproducible from slurms.

The following are partly addressed analytically, not fully solved in code yet:
- stronger robustness to FS angle/channel uncertainty;
- explicit incumbent sensing architecture tightly coupled to beamforming;
- full site-specific selective FR3 channel modeling beyond the current tractable diagnostic model.

## 3) Why a smaller reference network is used in the TWC package
The original repo geometry is large and is fine for the baseline paper scope. For PF outer loops, deep unfolding, FER, and selectivity studies, that geometry becomes unnecessarily heavy.

So the TWC package uses:
- a **smaller reproducible reference multi-cell geometry** for the main algorithmic comparisons;
- a separate **scaling suite** to increase antenna count and user load.

This keeps the study clean, reproducible, and computationally realistic on Narval.

## 4) Main files added
- `src/fr3_twc/solver.py`: weighted PF/WMMSE + FS protection
- `src/fr3_twc/unfolding.py`: trainable unfolded solver
- `src/fr3_twc/fs_masks.py`: cognitive per-BS/per-tone masking
- `src/fr3_twc/selectivity.py`: flat-vs-selective channel diagnostics
- `src/fr3_twc/fer.py`: Sionna-based 5G NR FER post-processing
- `scripts/run_twc_baselines.py`
- `scripts/train_unfolding.py`
- `scripts/eval_unfolding.py`
- `scripts/run_scaling_suite.py`
- `scripts/run_selectivity_suite.py`
- `scripts/plot_twc_figures.py`
- `slurm/*.slurm`

## 5) Figures this package generates
- reference geometry
- weighted sum-rate vs SNR
- sum-rate vs SNR
- fairness vs SNR
- protection satisfaction vs SNR
- coverage vs SNR
- runtime vs SNR
- convergence (`w_delta`, weighted sum-rate)
- FER curves
- scaling heatmaps
- selectivity gap plots
- fairness vs protection trade-off

## 6) Clean-repo rule
Before a fresh full run, remove the old `results_twc/` directory if you do not need it.

## 7) Sionna version note
This package pins `sionna==0.19.2` and TensorFlow 2.13-2.15 compatibility.
That keeps the FER chain on the mature **TensorFlow-based** Sionna branch used by this codebase, instead of the new Sionna 2.x line.
