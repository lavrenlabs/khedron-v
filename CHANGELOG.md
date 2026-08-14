# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Initial public release. Highlights:

- LoCoMo benchmark loading with audited and standard scoring, binary outcomes,
  and Wilson 95% confidence intervals.
- Full-context and pluggable memory providers; OpenAI, Anthropic, and Google
  answer-model and judge adapters.
- Hybrid JSONL + SQLite persistence with JSONL as the source of truth.
- Pre-dispatch budget reservation bounding worst-case API spend, request pacing,
  stage-level recovery, and run resume.
- CLI for validation, running, cost estimation, inspection, reporting, and
  comparison; Markdown reports and a Streamlit dashboard.
