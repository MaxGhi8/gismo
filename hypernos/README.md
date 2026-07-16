# Standalone multi-patch AFIETI transformer example

This directory reproduces `hypernos/examples/train_mp_afieti_transformer.py`
from an application outside the HyperNOs source tree.  The dataset pipeline,
normalizer, model, and best configuration are local; the installed `hypernos`
package is used only for its fixed-model training loop and `lpLoss`.

## Files

- `train.py`: command-line entry point and HyperNOs integration.
- `dataset.py`: CSV parsing, deterministic splits, normalization, and loaders.
- `model.py`: complete geometry-conditioned linear operator definition.
- `best_config.json`: the hyperparameters used by the original example.

The 277 MB `yeti_dataset.csv` is intentionally not duplicated.  Supply its path
with `--data`.  When this folder remains in the HyperNOs repository, the default
path already points to `data/mp_afieti/yeti_dataset.csv`.

## Run

Create an environment with the separately installed library, then run the
script directly so its sibling modules are imported locally:

```bash
# NVIDIA GPU on Linux (CUDA 12.8 wheel):
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0

# Includes HyperNOs from its v0.1.3 source tag. The published 0.1.3 wheel
# omits subpackages required by hypernos.loss_fun.
python -m pip install --requirement requirements.txt
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
