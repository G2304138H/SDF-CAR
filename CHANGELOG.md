# Change log

## 2026-09-09 — First-epoch non-finite-gradient fix

- Bound Kornia soft distance-transform inputs to `[1e-6, 1 - 1e-6]`.
  Kornia internally computes a logarithm; exact-zero detector margins could
  have finite sanitized forward values but non-finite derivatives.
- Run the first 100 epochs with projection loss only, then linearly ramp the
  2D SDF weight over 400 epochs. This stabilizes initialization and makes an
  epoch-one failure specifically identify the ODL/ASTRA projection path.
- Skip the AdamW update whenever the loss or any gradient is non-finite, so a
  failed diagnostic epoch cannot corrupt the network parameters.
- Log the effective SDF weight and the new epsilon/warm-up/ramp settings in
  `training_log_<case>.jsonl`; the raised error now states whether Kornia was
  present in the failing backward graph.

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
