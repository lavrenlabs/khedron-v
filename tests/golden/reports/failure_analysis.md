# Failure Analysis - Synthetic Memory Run

**Generated at:** 2026-05-06T12:34:56Z

**Run ID:** run-1

## Summary

2 of 3 question(s) in this report context failed.

## Category Failure Breakdown

| Category | Total | Failed | Failure rate |
|---|---:|---:|---:|
| multi_hop | 1 | 1 | 100.0% |
| temporal | 2 | 1 | 50.0% |


## Detected Patterns

| Pattern | Affected | Rule confidence | Affected question IDs |
|---|---:|---:|---|
| temporal_arithmetic_failure | 1 | 75.0% | q-temporal |


### temporal_arithmetic_failure

Temporal answer was inconsistent with the expected date.

Suggested remedy: Add explicit date reasoning to the answer prompt.

This is rule confidence, not a statistical probability.



## Failed Question Summaries

| Question ID | Category | Verdict | Error phase | Question | Expected | Generated or error |
|---|---|---|---|---|---|---|
| q-temporal | temporal | incorrect | n/a | What is the answer for q-temporal? | Expected answer | Wrong month |
| q-multihop | multi_hop | error | generate | What is the answer for q-multihop? | Expected answer | Generation timed out |


## Traceability Index

| Question ID | Retrieval ID | Response ID | Judgment ID |
|---|---|---|---|
| q-temporal | ret-q-temporal | resp-q-temporal | judge-q-temporal |
| q-multihop | ret-q-multihop | resp-q-multihop | judge-q-multihop |


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
