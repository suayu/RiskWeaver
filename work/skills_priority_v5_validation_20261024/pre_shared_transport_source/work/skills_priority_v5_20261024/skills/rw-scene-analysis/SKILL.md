---
name: rw-scene-analysis
description: Analyze RiskWeaver's deterministic scene features and interaction evidence before ranking discrete attack intents.
---
# Scene analysis
Use [contract](../shared/contract.md) and [recipes](../shared/recipes.md).
Read scene_features: relative_forward/right are ego-heading coordinates. Candidate target_id always identifies a background actor, never the ego.
Use time-aligned distances and scene conditions. Distinguish facts, forecasts and missing evidence. The first-segment velocity delta is a continuity quantity, not attack strength.
A null margin with status no_non_target_objects means the sampled screening set was empty, not that safety checking was disabled. With measurement_unavailable, do not invent clearance. The mandatory margin is the explicit threshold, not zero.

## Mandatory identity and time checks
Start with ego_reference=separate_ego, then identify each selected background target. Compare source-time roles, current relative motion and the active-window proxy before discussing later forecasts. If no published catalog exists, preserve the exact no-opportunity/preflight reason; do not call it a service failure.
