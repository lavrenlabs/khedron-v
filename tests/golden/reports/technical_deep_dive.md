# Technical Deep Dive - Synthetic Memory Run

**Generated at:** 2026-05-06T12:34:56Z

## Run Configuration

| Field | Value |
|---|---|
| Suite ID | suite-1 |
| Experiment ID | exp-1 |
| Run ID | run-1 |
| Run number | 1 |
| Status | completed |
| Started | 2026-05-06T12:00:00+00:00 |
| Finished | 2026-05-06T12:00:00+00:00 |
| Provider | full_context 2.5.0 |
| Answer model | claude-sonnet-4-5 |
| Judge model | gpt-4o-2024-08-06 |
| Methodology | 1.0 / canonical-v1 |
| Methodology fingerprint | n/a |
| Framework version | 0.0.0 |
| Top-K retrieval | 10 |
| Max concurrent questions | 4 |

## Lifecycle and Reproducibility

| Field | Value |
|---|---|
| Run-start event ID | event-run-started-1 |
| Run-start timestamp | 2026-05-06T12:00:00+00:00 |
| Suite status | completed |
| Suite last event | suite_completed at 2026-05-06T12:00:00+00:00 |
| Benchmark | locomo 1.0 |
| Benchmark checksum | sha256:synthetic |
| Methodology reference | n/a |
| Reference commit | n/a |
| Seed | 42 |

## Aggregate Results

Overall scores cover all evaluated categories. Per-category figures below
report every evaluated category, including any excluded from the overall.

| Mode | Overall score |
|---|---|
| Audited | 80.0% [54.8, 93.0] |
| Standard | 70.0% [48.1, 85.5] |

### Experiment Aggregate

| Mode | Runs | Pooled score |
|---|---:|---|
| Audited | 2 | 80.0% [54.8, 93.0] |
| Standard | 2 | 70.0% [48.1, 85.5] |


## Per-Category Analysis

### Audited

| Category | n | Correct | Errors | Partial | Unknown | Score |
|---|---:|---:|---:|---:|---:|---|
| temporal | 6 | 4 | 2 | 0 | 0 | 66.7% [30.0, 90.3] |
| single_hop | 9 | 8 | 1 | 0 | 0 | 88.9% [56.5, 98.0] |



### Standard

| Category | n | Correct | Errors | Partial | Unknown | Score |
|---|---:|---:|---:|---:|---:|---|
| temporal | 5 | 3 | 2 | 0 | 0 | 60.0% [23.1, 88.2] |
| single_hop | 10 | 8 | 2 | 0 | 0 | 80.0% [49.0, 94.3] |



## Question Totals

| Metric | Value |
|---|---:|
| Conversations processed | 1 |
| Questions attempted | 20 |
| Questions succeeded | 18 |
| Questions errored | 1 |
| Questions in report context | 3 |
| Failed questions in report context | 2 |

## Cost Analysis

Total cost: $0.09

### Cost by Phase

| Phase | Cost |
|---|---:|
| generation | $0.03 |
| judgment | $0.06 |


### Cost by Model

| Model | Cost |
|---|---:|
| claude-sonnet-4-5 | $0.03 |
| gpt-4o-2024-08-06 | $0.06 |


## Failure Analysis Summary

| Metric | Value |
|---|---:|
| Category breakdown entries | 2 |
| Failed question summaries | 2 |
| Detected patterns | 1 |
| Traceability entries | 2 |

### Detected Patterns

| Pattern | Affected | Rule confidence | Suggested remedy |
|---|---:|---:|---|
| temporal_arithmetic_failure | 1 | 75.0% | Add explicit date reasoning to the answer prompt. |



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
