# Khedron

Khedron is a Python framework for evaluating AI memory systems on the LoCoMo
benchmark. It is built for reproducible AI-memory benchmarking: controlled
benchmark loading, memory-provider plugins, answer and judge model adapters,
Wilson confidence intervals, JSONL-first persistence, Markdown reports, and a
Streamlit dashboard.

## Features

- LoCoMo benchmark support with audited and standard scoring.
- Full-context and pluggable memory providers.
- OpenAI, Anthropic, and Google answer-model adapters.
- OpenAI, Anthropic, and Google judge adapters.
- Hybrid JSONL + SQLite persistence, with JSONL as the source of truth.
- Pre-dispatch budget reservation that bounds worst-case API spend, request
  pacing (rate limiting), stage-level recovery, and run resume.
- Binary scoring with Wilson 95% confidence intervals on every reported rate.
- A CLI for validation, running, cost estimation, inspection, reporting,
  comparison, and SQLite rebuild/version checks.
- Markdown reports and a five-page Streamlit dashboard.
- An offline-by-default test suite (unit, integration, end-to-end).

## Setup

Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and sync
dependencies:

```bash
git clone <repository-url>
cd khedron
uv sync
```

The project targets Python 3.11. If needed:

```bash
uv python install 3.11
uv sync
```

## Get the LoCoMo dataset

The LoCoMo corpus (Maharana et al., 2024) is released under CC BY-NC 4.0 and is
**not distributed with this framework**. Fetch it into `data/locomo/` with:

```bash
uv run python scripts/download_locomo.py
```

Tests that exercise the real corpus skip automatically when it is absent, so the
offline suite stays green without it.

## Safe offline verification

These commands require no vendor API keys and make no live model calls:

```bash
uv run khedron --help
uv run khedron validate --config experiments/quickstart.yaml
uv run pytest tests/unit tests/integration tests/e2e
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
uv run pyright src/ scripts/
```

## Live quickstart

`experiments/quickstart.yaml` is a real benchmark configuration. Running it with
the CLI can call live OpenAI, Anthropic, or Google APIs and can spend money:

```bash
uv run khedron run --config experiments/quickstart.yaml
```

Do not run this unless API keys are configured and you intend to spend. Copy the
environment template and fill in only the keys you intend to use:

```bash
cp .env.example .env
```

Never commit `.env`, API keys, auth headers, or raw provider credentials.

## Dashboard

Launch the Streamlit dashboard once you have results to inspect:

```bash
uv run streamlit run src/khedron/dashboard/app.py
```

The dashboard reads through repository APIs backed by SQLite/JSONL and is a
read-only analysis surface.

## Reports

After a persisted run exists:

```bash
uv run khedron report --help
uv run khedron compare --help
```

Every report discloses the methodology version and profile, dataset checksum,
model and judge IDs, audit mode, and Wilson confidence intervals.

## License

Distributed under the terms of the [LICENSE](LICENSE) file.
