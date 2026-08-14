# Executive Summary - Synthetic Memory Run

**Generated at:** 2026-05-06T12:34:56Z

**Run ID:** run-1
**Status:** completed

## Bottom Line

Synthetic Memory Run achieved **80.0% [54.8, 93.0]** in audited mode on locomo, scored over all evaluated categories.


## Overall Scores

| Mode | Score |
|---|---|
| Audited | 80.0% [54.8, 93.0] |
| Standard | 70.0% [48.1, 85.5] |

## Per-Category Scores

### Audited

| Category | Score |
|---|---|
| temporal | 66.7% [30.0, 90.3] |
| single_hop | 88.9% [56.5, 98.0] |

### Standard

| Category | Score |
|---|---|
| temporal | 60.0% [23.1, 88.2] |
| single_hop | 80.0% [49.0, 94.3] |

## Failure Analysis Summary

| Metric | Value |
|---|---|
| Questions evaluated | 3 |
| Failed questions | 2 |
| Detected patterns | 1 |

Top detected patterns:

- temporal_arithmetic_failure: 1 affected question(s), 75.0% rule confidence.



## Cost Summary

Total cost: $0.09

| Phase | Cost |
|---|---|
| generation | $0.03 |
| judgment | $0.06 |


## Methodology Disclosure

| Field | Value |
|---|---|
| Methodology version | 1.0 |
| Methodology profile | canonical-v1 |
| Methodology fingerprint | n/a |
| Scoring mode | audited |
| Confidence interval | Wilson 95% |
| Framework version | 0.0.0 |
| Reference | n/a |
| Reference URL | n/a |
| Reference commit | n/a |
| Benchmark | locomo 1.0 |
| Benchmark checksum | sha256:synthetic |
| Corpus coverage | not recorded |
| Conversation ingestion | not recorded |
| Answer model | claude-sonnet-4-5 (anthropic) |
| Judge model | gpt-4o-2024-08-06 (openai) |
| Cross-vendor | Yes |
| Same-vendor warning | No |
| Top-K retrieval | 10 |
| Multi-run | 2 runs, seed=42 |

### Conformance

This report uses binary canonical scoring: only CORRECT counts as success, and every displayed binomial score includes a Wilson 95% confidence interval.

### Reproducibility Identifiers

| Field | Value |
|---|---|
| Run ID | run-1 |
| Suite ID | suite-1 |
| Experiment ID | exp-1 |
| Experiment name | Synthetic Memory Run |
| Run number | 1 |
| Provider | full_context 2.5.0 |
| Seed | 42 |
