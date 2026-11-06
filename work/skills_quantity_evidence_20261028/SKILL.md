---
name: rw-quantity-evidence
description: Explain current motion, numerical comparisons and attack timing using deterministic supplied facts.
version: draft-20261028-v1
evidence_status: offline_prototype_not_closed_loop_validated
---

# Current-state and numerical evidence

This is an optional supplement to scene analysis and dynamic attack ranking. Preserve the existing response schema, ranking authority, candidate compiler, request budget and safety validators. Do not output controls or dense trajectories. This supplement does not certify prose, tactics or safety.

1. Use `context.quantity_facts.actor_current_motion` for **current** source-state motion. A positive `radial_gap_rate_mps` means current center distance is increasing, even if a future attack candidate could close that gap. Do not describe a separating current actor as already closing. Current pose, kinematic heading and lane topology are different facts; do not invent topology from heading alone.
2. For numerical comparisons, state the quantity and candidate set. The supplied `active_center_gap_CV_increment_m` values are differences between minimum center distances under sparse intent and constant-velocity references, not measured danger or real-policy causal benefit. Their descending tie groups supply arithmetic order only. They do not decide the attack ranking: explain other relevant tradeoffs explicitly, without claiming a smaller quantity is larger.
3. Use `supplied_background_margin_m` only for that supplied geometric margin. A candidate is not the minimum if another candidate in the stated set has a smaller finite value. Null margins are not zero and are not safety certificates. Final joint safety checks remain mandatory.
4. `recipe_design_time_seconds`, `active_horizon_seconds` and `anchor_forecast_horizon_seconds` are distinct. A recipe designed around 0.5 seconds may still have a 1-second validity and 3-second anchors. Do not infer a shorter validity, different execution duration or calibrated uncertainty from the recipe time.
5. Prefer short, evidence-referenced reasons. Avoid unsupported superlatives, uncertainty comparisons and guaranteed ego reactions. Approximate ties should not be exaggerated into meaningful differences. Preserve exploratory options and honest no-attack; these facts introduce no rejection threshold.

When `quantity_facts.available` is false or a field is missing, state the uncertainty and continue with the original evidence and validators. Do not fabricate a value or trigger extra requests to populate the supplement. The model still chooses discrete candidates and strategies; diffusion alone refines continuous trajectories.
