---
name: rw-future-interaction-events
version: 20261030-v1
evidence_status: source_prechecked_not_closed_loop_validated
description: Rank sparse interaction intents using source-time ego future hypotheses and explicit lease timing.
---

# Future interaction events

Use `future_event_evidence` for candidates whose recipe is
`full_grid_source_CV_sparse_intercept_v1`.

Their `event_time_seconds` is a hypothesized encounter with the **ego's future
constant-velocity position**, NOT the ego's current position and NOT a measured future
or certified collision. This reference can become wrong when ego reacts.

The coordinates remain four sparse world anchors at 0/1/2/3 seconds. An encounter
can occur between anchors. The event is not a fifth anchor, dense trajectory, or control.

If `event_within_current_lease` is false, the 10-frame lease does not reach that event.
Describe this choice as the first phase of a future interaction. Do not claim contact
or added threat already occurs during the lease. Future replanning, target deactivation,
or tracking error may invalidate the intended event. Keep source/relative/world times
distinct and explain a necessary revision of an ongoing intent.

Compare every supported candidate using the supplied current-motion, active-window,
continuity and later-forecast evidence. Do not prefer this recipe merely because it is new.
A future intercept hypothesis alone does not prove unavoidable collision, executable
joint dynamics, positive reachable area, or background safety.

Keep the original response JSON fields, identity hashes and typed grounding. Cite the
original candidate evidence IDs; do not invent event evidence IDs or extra response fields.
The compiler reads target, strategy, four sparse anchors and duration from the directory.
Do not output continuous controls or alter thresholds, solver parameters, request budget,
or metric definitions.

## Evidence boundary

The full 0.1–3 second grid uses only the current source state, not observed future poses.
Each target publishes at most two screened representatives: minimum sparse acceleration
requirement and earliest passing time. The original 3.8 m/s² sparse-coefficient cap,
background OBB margin and semantic prechecks remain. Continuous execution has not yet
been certified for this recipe. Empty directories and failed attempts stay in the denominator.
