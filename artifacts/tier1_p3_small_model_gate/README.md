# Tier-1 P3 small-model gate

The gate is intentionally fail-closed.  P2 promotion is `NO-GO`, `peft` and
`bitsandbytes` are unavailable, and no CUDA device is present.  Therefore no LoRA,
QLoRA, online-RL, or parameter-internalization experiment was executed.

See `small_model_gate.json` and `run_manifest.json` for the machine-readable audit.
