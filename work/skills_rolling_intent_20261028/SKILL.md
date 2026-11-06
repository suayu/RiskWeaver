---
name: future-ego-and-rolling-attack-intent
status: independent-default-off-prototype-not-closed-loop-validated
---

Use rolling_intent and ego_future_reference before selecting a discrete attack.
They contain source-only forecast hypotheses, never ground-truth future ego states.
Compare positions at the SAME future world time; targeting the present ego position
does not imply future interaction. CV and route-constant-speed are uncertain models,
not the actual controller's reaction or calibrated probabilities.

Distinguish the 3-second anchor forecast, 1-second execution lease, replan period,
and the absolute interaction event time. A terminal anchor can be a post-pass point.
An event outside the lease has not been executed. Replanning a continuing attack
should preserve the previous target, discrete intent, and overlapping future geometry
when the supplied feasible candidates support it. Use a remaining-intent candidate
where appropriate; its fresh tail is explicitly labeled and is NOT old known intent.
Do not mechanically select continuity over a safe, more relevant alternative.

Explain target/strategy changes and failed continuation in candidate_reasons or
no_attack_reason, citing the supplied timing and continuity facts. An inactive target,
expired forecast, past event, missing safe continuation, or changed ego motion may
justify a reset. Do not silently push a missed interaction event into the future.
Tracking error at the source is separate from changes to the future plan's shape.

Use only the closed catalog, unchanged response schema and request budget. Do not
invent dense trajectories or continuous controls. All original validators remain.
Future center-gap proxies and cached accepted plans do not certify actual threat,
execution, background safety, tactical correctness, or measured D.
