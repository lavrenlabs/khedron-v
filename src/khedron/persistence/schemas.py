# ruff: noqa: E501

from __future__ import annotations

import base64
import gzip
from typing import Final

SCHEMA_VERSION: Final[int] = 2

SCHEMA_DDL: Final[str] = """-- File: results/benchmark.db
-- Schema version: 2

-- Metadata table
CREATE TABLE IF NOT EXISTS schema_meta (
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    framework_version TEXT NOT NULL
);

-- ============================================================
-- Lifecycle Events (event store)
-- ============================================================
-- These tables hold immutable lifecycle events. They are the source of truth
-- for state. The suites and runs tables (below) are materialized views
-- computed by reducing these events.

-- Suite lifecycle events
CREATE TABLE IF NOT EXISTS suite_lifecycle_events (
    event_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'suite_started', 'experiment_started', etc.
    event_data TEXT NOT NULL,  -- Full event JSON
    UNIQUE (suite_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_suite_events_suite ON suite_lifecycle_events(suite_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_suite_events_type ON suite_lifecycle_events(event_type);

-- Run lifecycle events
CREATE TABLE IF NOT EXISTS run_lifecycle_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,  -- 'run_started', 'conversation_processed', 'run_completed', 'run_failed'
    event_data TEXT NOT NULL,
    UNIQUE (run_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_lifecycle_events(run_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_run_events_type ON run_lifecycle_events(event_type);

-- ============================================================
-- Materialized State Tables
-- ============================================================
-- These tables hold the current state derived from lifecycle events.
-- They are recomputed when lifecycle events are inserted.

-- Suites (materialized state)
CREATE TABLE IF NOT EXISTS suites (
    suite_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed', 'partial'
    total_cost_usd REAL,
    n_experiments_planned INTEGER NOT NULL,
    n_experiments_completed INTEGER NOT NULL DEFAULT 0,
    n_experiments_failed INTEGER NOT NULL DEFAULT 0,
    methodology_version TEXT NOT NULL,
    methodology_profile TEXT NOT NULL DEFAULT 'canonical-v2',
    framework_version TEXT NOT NULL,
    config_yaml TEXT NOT NULL  -- Embedded for reproducibility
);

CREATE INDEX IF NOT EXISTS idx_suites_started_at ON suites(started_at);

-- Experiment Results (aggregate of multiple runs)
CREATE TABLE IF NOT EXISTS experiment_results (
    experiment_id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    n_runs INTEGER NOT NULL,
    overall_standard_pooled_score REAL,
    overall_standard_ci_low REAL,
    overall_standard_ci_high REAL,
    overall_standard_mean REAL,
    overall_standard_stddev REAL,
    overall_audited_pooled_score REAL,
    overall_audited_ci_low REAL,
    overall_audited_ci_high REAL,
    overall_audited_mean REAL,
    overall_audited_stddev REAL,
    total_cost_usd REAL NOT NULL,
    config_json TEXT NOT NULL,
    FOREIGN KEY (suite_id) REFERENCES suites(suite_id)
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_suite
    ON experiment_results(suite_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_name
    ON experiment_results(experiment_name);

-- Runs (materialized state)
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    provider_type TEXT NOT NULL,
    provider_version TEXT,
    benchmark_type TEXT NOT NULL,
    benchmark_version TEXT,
    benchmark_checksum TEXT NOT NULL,
    answer_model_vendor TEXT NOT NULL,
    answer_model_id TEXT NOT NULL,
    judge_model_vendor TEXT NOT NULL,
    judge_model_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    n_questions_attempted INTEGER,
    n_questions_succeeded INTEGER,
    n_questions_errored INTEGER,
    overall_score_standard REAL,  -- micro overall, binary scoring
    overall_score_standard_ci_low REAL,
    overall_score_standard_ci_high REAL,
    overall_score_audited REAL,   -- micro overall, binary scoring
    overall_score_audited_ci_low REAL,
    overall_score_audited_ci_high REAL,
    total_cost_usd REAL,
    methodology_version TEXT NOT NULL,
    methodology_profile TEXT NOT NULL DEFAULT 'canonical-v2',
    framework_version TEXT NOT NULL,
    runtime_environment TEXT,  -- JSON
    seed INTEGER,
    error_message TEXT,
    error_phase TEXT,  -- 'initialization', 'ingest', 'retrieve', 'generate', 'judge', 'analysis', or NULL
    -- experiment_id is a logical reference to ExperimentStartedEvent. Do not
    -- FK to experiment_results: that aggregate row is written only after all
    -- runs in the experiment complete.
    FOREIGN KEY (suite_id) REFERENCES suites(suite_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider_type);
CREATE INDEX IF NOT EXISTS idx_runs_models
    ON runs(answer_model_id, judge_model_id);

-- Question Evaluations (canonical question metadata for a run)
CREATE TABLE IF NOT EXISTS question_evaluations (
    question_evaluation_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    category TEXT NOT NULL,
    question_text TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    is_audited_error INTEGER NOT NULL,  -- 0 or 1
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    UNIQUE (run_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_question_evaluations_run
    ON question_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_question_evaluations_category
    ON question_evaluations(run_id, category);
CREATE INDEX IF NOT EXISTS idx_question_evaluations_audited
    ON question_evaluations(is_audited_error);

-- Retrievals (lightweight summary; full memories list stays in JSONL)
CREATE TABLE IF NOT EXISTS retrievals (
    retrieval_id TEXT PRIMARY KEY,
    question_evaluation_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    n_returned INTEGER NOT NULL,
    retrieval_latency_ms REAL NOT NULL,
    recovery_attempt_id TEXT,
    ingestion_attempt_id TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (question_evaluation_id) REFERENCES question_evaluations(question_evaluation_id),
    UNIQUE (question_evaluation_id)
);

CREATE INDEX IF NOT EXISTS idx_retrievals_run ON retrievals(run_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_question ON retrievals(question_id, run_id);
CREATE INDEX IF NOT EXISTS idx_retrievals_question_eval ON retrievals(question_evaluation_id);

-- Planned question manifests (JSONL remains authoritative).
CREATE TABLE IF NOT EXISTS question_plan (
    run_id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL,
    categories TEXT NOT NULL,
    corpus_checksum TEXT NOT NULL,
    question_ids TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Recovery causality projection. These tables are populated in a later stage.
CREATE TABLE IF NOT EXISTS recovery_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    error_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS error_resolutions (
    resolution_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    error_id TEXT NOT NULL UNIQUE,
    recovery_attempt_id TEXT NOT NULL,
    resolved_by_stage TEXT NOT NULL,
    resolved_by_stage_record_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Question Results (denormalized for fast queries)
-- Materialized as LEFT JOIN of question_evaluations + retrievals + responses + judgments
CREATE TABLE IF NOT EXISTS question_results (
    question_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    question_evaluation_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    category TEXT NOT NULL,
    question_text TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    generated_answer TEXT,
    retrieval_id TEXT,
    response_id TEXT,
    judgment_id TEXT,
    verdict TEXT NOT NULL,
    score REAL NOT NULL,
    judgment_reasoning TEXT,
    is_audited_error INTEGER NOT NULL,  -- 0 or 1
    n_memories_retrieved INTEGER,
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    judgment_latency_ms REAL,
    total_latency_ms REAL,
    generation_input_tokens INTEGER,
    generation_output_tokens INTEGER,
    judgment_input_tokens INTEGER,
    judgment_output_tokens INTEGER,
    total_cost_usd REAL,
    error_message TEXT,
    error_phase TEXT,  -- 'retrieve', 'generate', 'judge', or NULL
    PRIMARY KEY (question_id, run_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id),
    FOREIGN KEY (question_evaluation_id) REFERENCES question_evaluations(question_evaluation_id),
    FOREIGN KEY (retrieval_id) REFERENCES retrievals(retrieval_id)
);

CREATE INDEX IF NOT EXISTS idx_qr_verdict ON question_results(verdict);
CREATE INDEX IF NOT EXISTS idx_qr_category ON question_results(category);
CREATE INDEX IF NOT EXISTS idx_qr_run_category
    ON question_results(run_id, category);
CREATE INDEX IF NOT EXISTS idx_qr_run_verdict
    ON question_results(run_id, verdict);
CREATE INDEX IF NOT EXISTS idx_qr_audited
    ON question_results(is_audited_error);
CREATE INDEX IF NOT EXISTS idx_qr_retrieval ON question_results(retrieval_id);

-- Scores by Category
CREATE TABLE IF NOT EXISTS scores_by_category (
    run_id TEXT NOT NULL,
    category TEXT NOT NULL,
    mode TEXT NOT NULL,  -- 'standard' or 'audited'
    n_total INTEGER NOT NULL,
    n_correct INTEGER NOT NULL,
    n_errors INTEGER NOT NULL,
    n_partial INTEGER NOT NULL DEFAULT 0,
    n_unknown INTEGER NOT NULL DEFAULT 0,
    point_estimate REAL NOT NULL,
    ci_95_low REAL NOT NULL,
    ci_95_high REAL NOT NULL,
    PRIMARY KEY (run_id, category, mode),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Failure Patterns
CREATE TABLE IF NOT EXISTS failure_patterns (
    pattern_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    description TEXT NOT NULL,
    suggested_remedy TEXT NOT NULL,
    n_affected_questions INTEGER NOT NULL,
    confidence REAL NOT NULL,
    affected_question_ids TEXT,  -- JSON array of strings
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_patterns_run ON failure_patterns(run_id);
CREATE INDEX IF NOT EXISTS idx_patterns_name ON failure_patterns(pattern_name);

-- Errors
CREATE TABLE IF NOT EXISTS error_log (
    error_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    phase TEXT NOT NULL,
    question_id TEXT,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    recovered INTEGER NOT NULL,
    retryable INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""

# Frozen gzip+base64 UTF-8 artifact of the historical v1 DDL. It is deliberately
# independent from SCHEMA_DDL: changing v2 cannot silently redefine migration input.
_V1_SCHEMA_DDL_GZIP_BASE64: Final[str] = (
    "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"
)
V1_SCHEMA_DDL: Final[str] = gzip.decompress(base64.b64decode(_V1_SCHEMA_DDL_GZIP_BASE64)).decode(
    "utf-8"
)

__all__ = ["SCHEMA_DDL", "SCHEMA_VERSION", "V1_SCHEMA_DDL"]
