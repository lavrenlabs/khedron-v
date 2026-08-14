# Khedron methodology

Khedron evaluates AI memory systems on the [LoCoMo](https://github.com/snap-research/locomo)
benchmark. This document specifies the **canonical measurement methodology**: how questions are
asked, answered, retrieved, scored, and aggregated, and how each run is fingerprinted so that two
results are comparable only when they measured the same thing.

The measurement-affecting settings are frozen in named **methodology profiles**
(`src/khedron/methodology/profiles.py`); every run records the profile it used, and its
methodology disclosure names each of the fields below.

## The benchmark

LoCoMo is a set of long, multi-session conversations between two speakers, each paired with a
question-answering set. Questions fall into five categories:

- **single-hop** — answerable from one piece of evidence
- **multi-hop** — require combining several pieces of evidence
- **temporal** — about when something happened, or relative timing
- **open-domain** — draw on commonsense in addition to the conversation
- **adversarial** — unanswerable from the conversation; the correct response is to decline

About a fifth of LoCoMo turns also carry a shared image. How those images enter the corpus is a
declared policy (see *Image descriptions* below), not an implicit default.

> **Dataset licensing.** The LoCoMo dataset is distributed under **CC BY-NC 4.0** and is *not*
> vendored in this repository. Fetch it locally with `scripts/download_locomo.py`.

## The measurement pipeline

For each experiment, Khedron runs:

```
ingest conversation → provider stores memory → for each question:
    retrieve → generate an answer → score the answer → aggregate
```

The **memory provider** is the system under test: it ingests the conversation and, at query time,
returns the material the answer model sees. A full-context (non-retrieving) provider stands in for
passing the whole conversation to the model and is used as a reference arm.

## Scoring

Answers are scored with **canonical binary CORRECT-only scoring**: an LLM judge compares each answer
to the gold answer and returns correct or incorrect. Per-category and overall rates are reported with
**Wilson 95% confidence intervals**.

The **adversarial** category is *asked* — it is a useful diagnostic — but kept out of the headline
aggregate. A system that answers nothing scores near-perfectly on unanswerable questions, so that
category is not discriminative on its own; it is reported per-category instead.

## Aggregation

- The headline is a **micro-average over the answerable subset** (single-hop, multi-hop, temporal,
  open-domain).
- Across runs, intervals are **pooled** rather than averaged.
- A single run (**N = 1**) is admissible only as a **labelled non-aggregate** descriptive score,
  never as a measured multi-run aggregate. A sample standard deviation is defined for any n ≥ 2, so
  the advisory minimum of three runs is about characterising run-to-run dispersion, not
  impossibility.

## Retrieval budget

Each profile pins a retrieval budget (`top_k`) and the depth cutoffs used to describe the
retrieval sweep. A provider that returns everything (full context) **discards `top_k` by design** —
honouring a budget would destroy the baseline it exists to be — so it runs under a separate
no-retrieval baseline profile. That way a budget the provider never applied is not recorded as if it
had been. The difference is deliberate and enters the fingerprint (below), so the two arms are never
presented as a shared measurement.

## Cross-vendor judging

The judge model is a **different vendor than the answer model**, which avoids the same-model bias of
having a model grade its own family's output. Model identifiers are **version-pinned** (not rolling
aliases), so a silent vendor-side model change cannot alter the measurement without an explicit edit.
Each run's disclosure reports whether the answer and judge vendors differed, and the exact pinned
model IDs.

## Methodology fingerprint

Every run stamps a SHA-256 **fingerprint** computed over the measurement-affecting fields —
generator and judge prompts (by content hash), evaluated and scored categories, scoring and
aggregation rules, answer and judge models, retrieval budget and cutoffs, and the image policy.

Identity and governance fields (profile name, human-readable reference, successor) are **excluded**
from the hash, so renaming a profile or declaring its successor does not change the fingerprint of a
run whose measurement was untouched. **Two results are comparable only when their fingerprints
match.**

## Build identity and reproducibility

Every run records the framework version that produced it. A run from a dirty or otherwise
unidentifiable build is refused as a publishable measurement (with an explicit override for a
knowingly throwaway run), because a result must be reproducible from a specific commit.

## Image descriptions

Image handling is a **named policy** (`none` or `blip_caption_only_v1`) rather than a boolean. A
boolean cannot distinguish two different renderings of "images included"; a named, versioned policy
can, and it participates in the fingerprint so the corpus cannot change without the hash noticing.

## Canonical profiles

The profiles form a versioned evolution: each version refines the measurement over the previous one.
Superseded profiles are refused for new runs.

### canonical-v1

The initial canonical profile. It fixes the scoring and aggregation rules but leaves the answer and
judge models, retrieval budget, and category set unset, so a run under it constrains little about
what was measured. **Superseded by canonical-v2.**

### canonical-v2

- Puts the **session date** in front of the model, so temporal questions become answerable.
- Pins the **answer and judge models** (cross-vendor, version-locked).
- Pins a **retrieval budget** and depth cutoffs.
- Separates **evaluated** from **scored** categories, so adversarial can be asked without entering
  the headline.

A companion **canonical-v2-baseline** leaves the retrieval budget unpinned for full-context
providers.

### canonical-v3

Three changes over v2, each isolated so its effect can be attributed independently:

1. the corpus gains **image-caption descriptions** (`blip_caption_only_v1`);
2. the generator no longer hands the model a scripted refusal;
3. the judge no longer penalises a defensible answer where the ground truth is itself vague.

Companion profiles — **canonical-v3-baseline** (full-context) and **canonical-v3-generator-only**
(the generator change in isolation) — exist so each change can be attributed independently.

## Reproducing a measurement

1. Fetch the dataset: `python scripts/download_locomo.py`.
2. Choose a canonical profile and configure an experiment suite (see `experiments/quickstart.yaml`).
3. Run from a clean, committed tree so the build is identifiable.
4. Compare only results whose methodology fingerprints match.
