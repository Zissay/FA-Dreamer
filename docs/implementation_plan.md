# Minimal Repository Implementation Plan

Goal: keep only the code needed to train, evaluate, and generate the three paper figures.

Steps:

1. Copy the FA-AFDM channel/environment package.
2. Keep short training and evaluation entry points in `scripts/`.
3. Keep one plotting script per final figure.
4. Write generated outputs to `runs/` and `figures/`.
5. Ignore generated results in `.gitignore`.
6. Verify all Python files compile.
