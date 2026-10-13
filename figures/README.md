# Figures

For the two preserved mobility experiments, run:

```bash
python scripts/reproduce.py plot-all --output-dir figures/reproduced
```

This writes `training_curve_50.pdf`, `training_curve_500.pdf` and
`rate_bar_50kmh_vs_500kmh.pdf`, their PNG previews and data/audit files.
The three panels have identical axis-frame dimensions and a shared PDF crop.
Their underlying terminal records are tracked under `experiments/`.

Generated figures are ignored by Git.
