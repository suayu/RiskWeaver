---
name: rw-identity-and-comparison-evidence
version: 20261030-v1
evidence_status: unit_tested_not_majority_correct_certified
description: Copy request identity without changing it and distinguish exact source-time comparison quantities.
---

# Identity and comparison evidence

Copy request_id, state_hash and catalog_hash EXACTLY from
`required_identity_copy_exactly`, the final payload member. Do not regenerate hashes,
copy a previous response's wrong identity, shorten them, or add this reference object
to your response. A single changed character must remain a validation failure.

Use `context.comparison_facts` rather than recomputing order or differences from prose.

- A **smaller active-window gap** is a smaller
  `active_window_min_center_gap_m`. Claim 'smallest' only for IDs listed in
  `smallest_active_gap.candidate_ids`, or explicitly state a narrower comparison set.
- `natural_min_minus_candidate_min_m` compares two minima, possibly at different times.
  A positive value means the candidate is closer; a negative value means it is farther.
  Do not write 'closer than natural CV' for a supplied `farther_than_natural_CV` fact.
- `natural_gap_at_candidate_min_time_minus_candidate_min_m` is the separate SAME-TIME
  comparison. Do not silently substitute it for the difference between minima.
- A drop from CURRENT distance includes natural closing; it is not attack-added benefit.
- Claim 'largest added closeness' only for listed IDs. Do not call one candidate the ONLY
  improving choice when multiple candidates have positive values.
- Finite background margin comparisons use their own lists. Null is not zero and not a
  finite margin; do not invent a number for it or infer an execution certificate.
- Recipe event time, actual proxy minimum time and active lease duration are different.
  An event beyond the lease is a future-phase hypothesis, not an executed encounter.

When several useful metrics disagree, give a tradeoff instead of calling the selected
candidate 'best' under a metric where it is not best. Keep reasons qualitative if a
number is unnecessary. Copy an applicable supplied direction label and cite existing
evidence IDs. Keep every original response field and type; do not add comparison fields
or extra reasoning outside the JSON. No extra API requests or changes to safety/solver.
