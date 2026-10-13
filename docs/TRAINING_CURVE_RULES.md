# Formal Online Training-Curve Rules

These rules are mandatory for every formal Dreamer/PPO comparison figure in
this project. They describe plotting only; they do not alter training,
environment interaction, replay, or policy updates.

## Data source and x-axis

- Use only actual online environment records whose `episode_terminal` field is
  `1`.
- Keep the original global real-interaction index from the log as the x-axis,
  labelled **Environment Interaction Steps**.
- Do not include reset-only records, imagined rollouts, model predictions,
  checkpoint evaluations, exhaustive-search queries, candidate searches, or
  best-so-far values.

## Batch definition and point preservation

- Preserve every real terminal action in a raw CSV. For the matched 50-step,
  20-environment, 50K experiment, this is exactly `50000 / 50 = 1000` raw
  terminal points per method.
- One synchronized parallel batch consists of the 20 terminal episodes that
  finish after each 1,000 real interactions. Its x-coordinate is the final
  real interaction index of that batch: 1K, 2K, ..., 50K.
- The batch value is the arithmetic mean of its 20 real terminal rates. Thus
  a 50K run has 50 raw batch-mean points. This is not an evaluation, imagined
  rollout, best-so-far statistic, or a replacement of raw records.

## Display rule

- The formal displayed line is a **causal trailing mean of the most recent
  five synchronized batch means**. Each displayed point therefore summarizes
  at most `5 x 20 = 100` real terminal episodes; the early points use all
  available preceding batches.
- A 50K matched run displays 50 points, one at every 1K real interaction
  boundary. The horizontal axis starts at zero, but the first measured point
  is correctly located at 1K; no artificial `(0, 0)` rate point is added.
- No interpolation, decimation, exponential moving average, cumulative
  maximum, future-looking smoothing, artificial monotonic adjustment,
  confidence shading, or replacement by a frozen-policy evaluation is
  allowed.

## Fair comparison and audit

- Dreamer and PPO must use the same interaction budget, horizon, physical
  action constraint, fixed channel, reset rule, and terminal metric.
- Each figure must save an audit JSON with 1,000 raw terminal points, 50
  batch-mean points, 50 displayed points, the 20-environment batch definition,
  and the causal five-batch display window.
