# Standalone multi-patch AFIETI transformer example

This directory reproduces `hypernos/examples/train_mp_afieti_transformer.py`
from an application outside the HyperNOs source tree.  The dataset pipeline,
normalizer, model, and best configuration are local; the installed `hypernos`
package is used only for its generic training/tuning utilities and `lpLoss`.

## Files

- `train.py`: command-line entry point and HyperNOs integration.
- `dataset.py`: CSV parsing, deterministic splits, normalization, and loaders.
- `model.py`: complete geometry-conditioned linear operator definition.
- `best_config.json`: the hyperparameters used by the original example.
- `ray_tune.py`: Ray Tune/HyperOpt search using the local model and dataset.

The 277 MB `yeti_dataset.csv` is intentionally not duplicated.  Supply its path
with `--data`.  When this folder remains in the HyperNOs repository, the default
path already points to `data/mp_afieti/yeti_dataset.csv`.

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
