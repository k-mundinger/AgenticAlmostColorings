# Best m018 2048 Construction

This package contains the best clean verified construction found in the
128-model sweep.

## Result

- Run id: `best128_paper_params_20260621_120017_1291273_m018_seed7000018_combo000`
- Model index: `18`
- Seed: `7000018`
- Grid: `2048 x 2048`
- Verified bonus-color fraction: `3.86731625%`
- Real-color conflict cells after verification: `0`

## Files

- `model/trained_model.pt`: trained PyTorch checkpoint.
- `model/pipeline_config.json`: config used by the verifier, including the learned parallelogram.
- `model/hparams.json`: sweep metadata for this model.
- `model/train_losses.csv`: recorded training losses.
- `verification/verified_paralellograms_ip.csv`: verifier result row.
- `verification/fixed_coloring_2048.npz`: compressed fixed coloring grid.
- `visualizations/`: square-grid, bonus-mask, and parallelogram renderings in PNG, PDF, and SVG.

Load the fixed coloring with:

```python
import numpy as np

grid = np.load("verification/fixed_coloring_2048.npz")["fixed_coloring"]
```

## Learned Parallelogram

```text
v1 = (1.8702629804611206, 0.609319806098938)
v2 = (0.39184680581092834, 1.882055401802063)
```

## Checksums

```text
a765406e3e733ac213aa25374015d569b99799e4b9db98fdf160e392d12a9c36  verification/fixed_coloring_2048.npz
97a7adf84212e7c33314fb9431cd7c369e4276a36e703b1654fd86fe8e8772c2  model/trained_model.pt
```
