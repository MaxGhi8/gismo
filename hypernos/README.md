# Standalone multi-patch AFIETI transformer example

This directory reproduces `hypernos/examples/train_mp_afieti_transformer.py`
from an application outside the HyperNOs source tree.  The dataset pipeline,
normalizer, model, and best configuration are local; the installed `hypernos`
package is used only for its generic training/tuning utilities and `lpLoss`.

## Files

- `train.py`: command-line entry point and HyperNOs integration.
- `dataset.py`: CSV parsing, deterministic splits, normalization, and loaders.
- `model.py`: complete geometry-conditioned linear operator definition.
- `default_config.json`: the original rank-20 configuration.
- `solver_config.json`: the rank-48 configuration selected by the solver study.
- `fine_p2_r3_config.json`: full-rank configuration for the finer `p=2,r=3`
  retraining comparison.
- `ray_tune.py`: Ray Tune/HyperOpt search using the local model and dataset.
- `export_onnx.py`: checkpoint export with the cacheable
  `u,Q,K_scaled,epsilon` output contract required by the C++ solver.
- `augment_spectral_probe_dataset.py`: reconstruct local Schur matrices from
  canonical rows and add mixtures of the checkpoint's worst generalized
  eigenmodes as active probe family 3.

Supply the training CSV with `--data`.  The new generated fine-grid dataset is
`data/mp_afieti/yeti_dataset_p2_r3_rich.csv`; the original data path remains
`data/mp_afieti/yeti_dataset.csv` when it is available.

## Run

Create an environment with the separately installed library, then run the
script directly so its sibling modules are imported locally:

```bash
python -m pip install -r requirements.txt
python train.py --data /absolute/path/to/yeti_dataset.csv
```

The original configuration trains for 500 epochs on 16,000 samples.  A short
pipeline check can use overrides, while retaining 2,000 validation and 2,000
test rows:

```bash
python train.py \
  --data /absolute/path/to/yeti_dataset.csv \
  --epochs 1 \
  --training-samples 100 \
  --batch-size 10 \
  --output ./runs/smoke-test
```

Training artifacts, TensorBoard logs, the flattened configuration, norm
description, and model checkpoint are written below the selected output folder.

The data generator supports mode-complete and mixed probe distributions.  The
`rich` profile is recommended for learning a linear Schur operator: it emits a
complete canonical-basis sweep for every patch, then alternates GRF and IID
probes. Unit normalization prevents a few large random vectors from dominating
the relative loss:

```bash
./build/bin/ieti_dataset_generation_example \
  -p 2 -r 3 --NumSamples 4032 --seed 20260716 \
  --probeType rich --grfLength 0.08 --normalizeProbes \
  --out hypernos/data/mp_afieti/yeti_dataset_p2_r3_rich.csv

python hypernos/train.py \
  --data hypernos/data/mp_afieti/yeti_dataset_p2_r3_rich.csv \
  --config hypernos/fine_p2_r3_config.json \
  --output hypernos/runs/fine_p2_r3
```

The retained fine-grid result was continued for 15 + 50 + 100 epochs (batch
sizes 16, 32, 32). Its held-out probe loss fell from 0.0155 to 0.00518 to
0.00282, and the corresponding PCG count fell from 44 to 38 to 28. The final
checkpoint and cacheable ONNX export are in
`runs/fine_p2_r3_rich_epoch165/`. At this discretization, the same solve takes
46 iterations without a preconditioner, 32 with the projected coarse model,
and 16 with exact scaled Dirichlet.

For other generated discretizations, `--auto-shapes --full-rank` infers the
maximum geometry/skeleton sizes directly from the CSV header. Validation and
test split sizes can be overridden with `--validation-samples` and
`--test-samples`.

## Solver-selected rank-48 checkpoint

The reported model was trained in three deterministic 100-epoch stages.  The
weights are continued while the optimizer is deliberately restarted at each
stage:

```bash
python train.py --config solver_config.json --epochs 100 \
  --output runs/solver_fullrank_stage1

python train.py --config solver_config.json --epochs 100 \
  --checkpoint runs/solver_fullrank_stage1/model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar \
  --output runs/solver_fullrank_stage2

python train.py --config solver_config.json --epochs 100 \
  --checkpoint runs/solver_fullrank_stage2/model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar \
  --output runs/solver_fullrank_stage3
```

Export the final weights and validate the ONNX graph with:

```bash
python export_onnx.py \
  --checkpoint runs/solver_fullrank_stage3/model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar \
  --config solver_config.json \
  --output runs/solver_fullrank_stage3/yeti_schur_fullrank.onnx
```

From `examples/`, the unified classical/neural experiment matrix is then:

```bash
python run_neural_preconditioner_experiments.py \
  --profile full \
  --model ../hypernos/runs/solver_fullrank_stage3/yeti_schur_fullrank.onnx \
  --num-run 40 --apply-repeats 5000 --threads 1
```

The degree/refinement transfer study uses the same coarse `p=2,r=2` checkpoint
at finer meshes and higher degrees. `native` intentionally records the fixed
ONNX-capacity failures; `nearest` and `idw` construct coordinate projectors.
The projected low modes are combined with a stiffness-Jacobi complement on the
unresolved fine modes, following the two-level construction in
`latex_experiments/close_ref/iganet_prec.tex`:

```bash
python examples/run_neural_preconditioner_experiments.py \
  --profile discretization \
  --model hypernos/runs/solver_fullrank_epoch300/yeti_schur_fullrank_epoch300.onnx \
  --num-run 10 --apply-repeats 2000 --threads 1
```

This profile writes a benchmark table, per-case residual histories, a manifest,
and PNG/PDF iteration, gain, ablation, and convergence plots.

The test split used during training contains new probe vectors on the same 21
yeti patches.  It is not a held-out-geometry split; use the full C++ experiment
matrix for the teapot and discretization transfer tests.

## Active spectral teapot update

Ordinary teapot fine-tuning lowered action loss but remained at 20 PCG
iterations.  The active update targets the multiplicative local errors of that
checkpoint.  Round one adds 32 mixtures of the eight worst modes on each of 32
patches (1,024 new rows) and is the selected result:

```bash
python hypernos/augment_spectral_probe_dataset.py \
  --data hypernos/data/mp_afieti/teapot_p2_r2_rich.csv \
  --checkpoint hypernos/runs/teapot_finetuned_p2_r2_epoch400/model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar \
  --config hypernos/teapot_finetune_stage2_config.json \
  --output hypernos/data/mp_afieti/teapot_p2_r2_spectral_active_round1.csv \
  --modes-per-patch 8 --probes-per-patch 32

python hypernos/train.py \
  --data hypernos/data/mp_afieti/teapot_p2_r2_spectral_active_round1.csv \
  --config hypernos/teapot_spectral_active_config.json \
  --checkpoint hypernos/runs/teapot_finetuned_p2_r2_epoch400/model_GeometryConditionedLinearOperator_default_YetiSchurTransformer.tar \
  --output hypernos/runs/teapot_spectral_active_round1_epoch150
```

The uncorrected active model matches scaled Dirichlet at 19 iterations.  The
final global-correction study, including matrix-free randomized Ritz and
full-spectrum diagnostic constructions, is reproduced with:

```bash
OMP_NUM_THREADS=1 python3 examples/run_adaptive_spectral_study.py \
  --tag adaptive_spectral_20260716
```

The study reports solve-only time: active-data generation, training, dense
materialization, and spectral coarse-space construction are setup and are not
included in the timing tables.

## Hyperparameter optimization

The search changes both optimizer settings and real model-capacity parameters:
the hidden width/head count (sampled jointly so it is always valid), geometry
encoder depth, dropout, and activation. `n_heads_A` remains fixed because this
specific SPD implementation averages those projections, making extra values a
redundant parameterization rather than genuinely separate attention heads.
Batch size is also fixed: the current HyperNOs Ray validation reduction is not
comparable across different batch sizes.

```bash
python ray_tune.py \
  --data /absolute/path/to/yeti_dataset.csv \
  --num-samples 40 \
  --max-epochs 300 \
  --grace-period 50 \
  --cpus-per-trial 2 \
  --gpus-per-trial 1
```

Each concurrent trial loads the wide CSV independently. Keep the number of
simultaneous trials conservative when system memory is limited.
