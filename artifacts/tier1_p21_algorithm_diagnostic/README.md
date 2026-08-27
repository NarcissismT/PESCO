# P2.1 fresh algorithm diagnostic

This bundle joins isolated CPU runs for SFT, GRPO-Terminal, GRPO-FourState, PESCO-BranchOnly, PESCO-NoFlipLoss, PESCO-Full, an Evidence-Gated SMOPD-inspired adapter, a dynamic-Lagrangian/PCGrad constrained policy, gradient cosine probes, and Logistic/Random-Forest/GBDT shortcut baselines. It is diagnostic-only; it is not a formal final and does not authorize LoRA, 7B, or online RL.

The aggregate keeps stable workspace copies of the isolated method, gradient, and constrained receipts under `input_methods/`, `input_gradient/`, and `input_constrained/`; its run manifest therefore does not depend on ephemeral `/tmp` paths. The promotion table reports the repaired Pairwise Reversal Ranking Accuracy separately from ordinary exact top-1 reversal accuracy.
