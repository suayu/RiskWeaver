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

## Distinguish natural motion from attack contribution
An oncoming actor can already approach ego without intervention. A slowing candidate may reduce the active-window threat even though its absolute gap keeps shrinking.
Inspect the target/ego natural CV reference and candidate :contrast facts. Relative heading only describes kinematics; it is not proof of intersection/merge topology.
No active background actors now does not imply no interaction throughout the scene. Keep source-time and whole-scene opportunity statements separate.

## Decision-relevant facts and motion
Describe only actors relevant to the selected decision. Before stating current ahead/behind, left/right, closing/separating, bind the actor identity to its supplied current-motion fact ID and copy its explicit categorical relation. A negative radial gap rate means closing; do not transfer another actor's relation. Current motion, candidate future motion, and past executed outcomes are different quantities.
Displayed rounded values can be approximately tied even when an exact-extremum ID list differs. Do not invent precision, exact ordering or safety certificates from rounded values.
