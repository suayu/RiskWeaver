---
name: rw-shared-fact-view
version: 20261030-v1
evidence_status: prototype_not_LLM_validated
description: Read exact shared defaults without inventing values, candidates or evidence IDs.
---

# Shared defaults in this request only

`context._factored_defaults_v1.groups` holds exact common fields omitted from each
record. Every record inherits its group's defaults recursively. A local value is
explicit; an empty fact object still inherits the shared fields. Apply:

- `candidates` to every candidate; `priority_outer` to each priority_evidence entry.
- `priority_fact:K` to the existing `CID:K` fact object of each candidate.
- `rolling_candidate_facts`, `comparison_rows`, and `fractional_rows` to the respective
  per-candidate dictionaries under rolling_intent, comparison_facts.per_candidate,
  and temporal_candidate_evidence.per_candidate_fractional_minimum.

This changes presentation only. All source numbers, nulls, target roles, durations,
original candidate IDs, order and fact IDs are unchanged. Null never means zero.
Use the ORIGINAL `CID:actor`, `CID:validity`, `CID:active/:after` and `CID:contrast`
evidence IDs; defaults groups are NOT new evidence IDs or response fields.
Read finite background margin using its explicit extrema sets: larger supplied
margin is larger, not lower. Forecast-only evidence and a recipe named rolling
do not prove a prior episode exists. Inherit the validity fields before reasoning
about event time. Keep all original response fields and types. Copy the three
identity strings exactly; do not recompute them from the presented view.
