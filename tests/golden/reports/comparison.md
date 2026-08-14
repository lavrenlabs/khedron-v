# Comparison Report

**Generated at:** 2026-05-06T12:34:56Z

## Methodology and Compatibility

| Field | Value |
|---|---|
| Comparison mode | audited |
| Baseline run ID | baseline-run |
| Candidate run ID | candidate-run |
| Compatible | No |

Compatibility warnings:

- Methodology profile differs: baseline=canonical-v1 candidate=canonical-v2
- Question ID sets differ: q-baseline-only / q-candidate-only



Detailed per-run methodology metadata is available in the corresponding run reports.

## Score Deltas

| Metric | Mode | Baseline | Candidate | Point delta | CI overlap | Significant |
|---|---|---|---|---:|---|---|
| overall | audited | 70.0% [48.1, 85.5] | 80.0% [58.4, 91.9] | +10.0 pp | CI-overlapping | no |
| temporal | audited | 40.0% [16.8, 68.7] | 70.0% [39.7, 89.2] | +30.0 pp | CI-separated | yes |
| multi_hop | audited | n/a | 50.0% [18.8, 81.2] | n/a | n/a | n/a |


## Question Differences

| Question ID | Category | baseline-run verdict | baseline-run score | candidate-run verdict | candidate-run score |
|---|---|---|---:|---|---:|
| q-temporal | temporal | correct | 1.0 | incorrect | 0.0 |
