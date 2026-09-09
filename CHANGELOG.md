# Change log

## 2026-09-09 — Trainer gradient-collapse fix

- Initialize SDF output with a configurable positive bias (`0.1` by default),
  giving approximately `0.0067` rather than `0.5` initial occupancy at
  `sdf_alpha=50`.
- Keep ODL/ASTRA projection, SDF-to-occupancy conversion, binary silhouette
  exponential, differentiable distance transform, and losses in FP32; mixed
  precision remains enabled only for the neural network.
- Require Kornia whenever the 2D SDF training loss has nonzero weight, avoiding
  the detached SciPy fallback that previously contributed a constant loss with
  no gradient.
- Record projection and SDF loss components, occupancy/projection ranges,
  gradient norm, parameter-probe delta, AMP scale, and skipped-step state in
  `training_log_<case>.jsonl` and the console.
- Abort on non-finite gradients immediately or after a configurable number of
  consecutive dead-gradient epochs instead of silently running to epoch 5000.
- Embed the active trainer-fix list and diagnostic-log path in the per-case
  timing JSON.
