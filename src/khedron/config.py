from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Final, Literal, Self, TypeAlias, cast

import structlog
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from khedron.errors import ConfigurationError
from khedron.pacing import RateLimitsConfig
from khedron.utils.redaction import looks_like_credential

__all__ = [
    "AnswerModelConfig",
    "BenchmarkConfig",
    "ExperimentConfig",
    "ExperimentSuiteConfig",
    "JudgeConfig",
    "ProviderConfig",
    "format_validation_errors",
    "load_suite_config",
    "warn_on_same_vendor_judges",
]

AuditMode: TypeAlias = Literal["standard", "audited", "both"]
# Kept in step with khedron.methodology.profiles.ProfileName, which cannot be imported here
# without a cycle. The duplication is guarded by a test rather than left to memory.
MethodologyProfile: TypeAlias = Literal[
    "canonical-v1",
    "canonical-v2",
    "canonical-v2-baseline",
    "canonical-v3",
    "canonical-v3-baseline",
    "canonical-v3-generator-only",
    "canonical-v3-prior-judge",
    "snap-original",
]
SeedStrategy: TypeAlias = Literal["deterministic", "random"]

_ENV_VAR_PATTERN: Final = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LOG: Final = structlog.get_logger(__name__)


def _empty_config() -> dict[str, Any]:
    return {}


# The free-form plugin block. A credential belongs inside one -- that is how a provider or
# model receives its key -- and `redact_secrets` keeps its values out of artifacts. Every
# other field is copied verbatim into run records, so a credential must never reach one.
_PLUGIN_CONFIG_KEY: Final = "config"


class _CredentialSafeModel(BaseModel):
    """Base for configuration models whose declared fields are recorded verbatim.

    Why on the base class rather than on the fields that were reported: four review rounds
    each found another field of this same class -- an arbitrary bag, a mapping key, then
    `name`, then the model ids -- because the guard was a list of the fields somebody had
    already noticed. `RunStartedEvent` copies almost every declared scalar of this schema into
    the run record (`types.py:414`), outside the `config` blob that redaction covers, so the
    correct scope is *every declared string field*, checked by construction and still correct
    when a field is added.

    The `config` bags are exempt: that is where a credential legitimately lives, and
    redaction covers them.

    This catches a credential typed in literally. A credential arriving through ``${VAR}`` is
    refused at substitution time instead, because interpolation into surrounding text
    ("debug ${KEY}") produces a value no shape test on the final string would recognise.

    Neither check is complete -- a token matching no known shape still passes -- and no
    stricter test is available: high-entropy heuristics would reject the dataset checksums
    this schema also carries.
    """

    @model_validator(mode="after")
    def _reject_credential_in_declared_fields(self) -> Self:
        for field_name in type(self).model_fields:
            if field_name == _PLUGIN_CONFIG_KEY:
                continue
            value = getattr(self, field_name, None)
            if isinstance(value, str) and looks_like_credential(value):
                # Deliberately does not echo the value: this message reaches logs and stderr.
                raise ValueError(
                    f"{field_name} looks like a credential, and declared configuration "
                    "fields are recorded verbatim in run artifacts"
                )
        return self


class ProviderConfig(_CredentialSafeModel):
    """Configuration for a memory provider implementation."""

    model_config = ConfigDict(frozen=True)

    type: str
    config: dict[str, Any] = Field(default_factory=_empty_config)


class BenchmarkConfig(_CredentialSafeModel):
    """Configuration for a benchmark dataset and audit mode."""

    model_config = ConfigDict(frozen=True)

    type: str
    config: dict[str, Any] = Field(default_factory=_empty_config)
    audit_mode: AuditMode = "both"


class AnswerModelConfig(_CredentialSafeModel):
    """Configuration for the answer-generating model."""

    model_config = ConfigDict(frozen=True)

    type: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0)
    max_output_tokens: int = Field(default=1024, ge=1)
    config: dict[str, Any] = Field(default_factory=_empty_config)


class JudgeConfig(_CredentialSafeModel):
    """Configuration for the LLM judge."""

    model_config = ConfigDict(frozen=True)

    type: str
    model: str
    temperature: float = Field(default=0.0, ge=0.0)
    config: dict[str, Any] = Field(default_factory=_empty_config)


class ExperimentConfig(_CredentialSafeModel):
    """Configuration for one benchmark experiment within a suite."""

    model_config = ConfigDict(frozen=True)

    name: str
    note: str = ""
    provider: ProviderConfig
    benchmark: BenchmarkConfig
    answer_model: AnswerModelConfig
    judge: JudgeConfig
    top_k_retrieval: int = Field(default=10, ge=0)
    max_concurrent_questions: int = Field(default=5, ge=1)


class ExperimentSuiteConfig(_CredentialSafeModel):
    """Configuration for a suite of benchmark experiments."""

    model_config = ConfigDict(frozen=True)

    runs: int = Field(default=3, ge=1)
    seed_strategy: SeedStrategy = "deterministic"
    # Publication guard: a run whose questions errored above this fraction was degraded
    # by infrastructure (rate limits, TLS, provider or judge failures), not by the memory
    # system under test. Errored questions score as wrong, so aggregating such a run
    # publishes a number that understates the system. Runs above the threshold are
    # excluded from the experiment aggregate instead of silently polluting it.
    max_run_error_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    # Companion guard for the ingestion phase. A turn that never reached the provider
    # leaves the memory store different from the corpus the methodology specifies, and
    # unlike a failed question it produces no ERROR verdict, so max_run_error_rate cannot
    # see it. Default 0: any ingestion loss makes a run unpublishable.
    max_ingestion_failure_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    # Recovery attempts are persisted before dispatch, so this small per-stage
    # budget remains effective across a process crash, as this design requires.
    max_recovery_attempts: int = Field(default=3, ge=0, le=10)
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    # Opt-in request pacing. Absent means the pacer is off and every model call behaves exactly as
    # before it existed; when present, one shared limiter paces the answer models and judges of the
    # whole suite so a run stays under the account's published rate limits.
    rate_limits: RateLimitsConfig | None = None
    output_dir: str = "results"
    methodology_version: str = "2.0"
    methodology_profile: MethodologyProfile = "canonical-v2"
    experiments: list[ExperimentConfig]


def load_suite_config(path: Path) -> ExperimentSuiteConfig:
    """Load, substitute environment variables, validate, and warn for a YAML suite config."""

    try:
        raw_content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            "Unable to read configuration file.",
            path=str(path),
            error=str(exc),
        ) from exc

    try:
        raw_data = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            "Invalid YAML configuration syntax.",
            path=str(path),
            error=str(exc),
        ) from exc

    if raw_data is None:
        raise ConfigurationError("Configuration file is empty.", path=str(path))
    if not isinstance(raw_data, dict):
        raise ConfigurationError(
            "Configuration YAML must be a mapping at the document root.",
            path=str(path),
            yaml_type=type(raw_data).__name__,
        )

    raw_mapping = cast(dict[object, object], raw_data)
    resolved_data = _resolve_env_vars(raw_mapping)

    try:
        config = ExperimentSuiteConfig.model_validate(resolved_data)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid suite configuration: {_format_validation_errors(exc)}",
            path=str(path),
            errors=_simplify_validation_errors(exc),
        ) from exc

    from khedron.methodology import validate_suite_methodology_profile

    validate_suite_methodology_profile(config)
    warn_on_same_vendor_judges(config)
    return config


def warn_on_same_vendor_judges(config: ExperimentSuiteConfig) -> None:
    """Emit a methodology warning for same-vendor answer model and judge pairs."""

    for experiment in config.experiments:
        if experiment.answer_model.type == experiment.judge.type:
            _LOG.warning(
                "same_vendor_configuration",
                warning=(
                    "Same-vendor configuration: answer_model and judge use the same "
                    "vendor. Results may be biased upward; see methodology section 8."
                ),
                experiment_name=experiment.name,
                vendor=experiment.answer_model.type,
                answer_model=experiment.answer_model.model,
                judge_model=experiment.judge.model,
            )


def _resolve_env_vars(value: object, *, credentials_allowed: bool = False) -> object:
    """Substitute ``${VAR}`` references, tracking whether a credential may land here.

    Substitution stays available everywhere -- parameterising ``output_dir`` or a model id
    from the environment is ordinary practice -- but a *credential-valued* substitution is
    confined to the plugin ``config`` blocks, the only place one belongs and the only place
    redaction covers. Everywhere else the value is copied verbatim into run records.

    This is checked per substituted value rather than on the finished string, because
    interpolation into surrounding text -- ``name: "debug ${OPENAI_API_KEY}"`` -- yields a
    string that begins with "debug" and would defeat any shape test applied afterwards. That
    exact form is how the previous two review rounds' leaks reached artifacts.
    """
    if isinstance(value, str):
        return _substitute_env_vars(value, credentials_allowed=credentials_allowed)
    if isinstance(value, list):
        value_list = cast(list[object], value)
        return [
            _resolve_env_vars(item, credentials_allowed=credentials_allowed) for item in value_list
        ]
    if isinstance(value, dict):
        value_dict = cast(dict[object, object], value)
        return {
            _reject_env_var_reference_in_key(key): _resolve_env_vars(
                nested_value,
                credentials_allowed=credentials_allowed or key == _PLUGIN_CONFIG_KEY,
            )
            for key, nested_value in value_dict.items()
        }
    return value


def _reject_env_var_reference_in_key(key: object) -> object:
    """Refuse ``${VAR}`` in a mapping key; substitution applies to values only.

    Why: substituting keys turned a resolved credential into a *key* of the config tree,
    and persistence preserves keys as provenance -- so a secret referenced from key position
    was written to artifacts verbatim, past the redaction that only inspects values. No
    configuration has ever needed it, so refusing is both safe and free, and it fails at
    load time with a message rather than silently producing a leaky artifact.
    """
    if isinstance(key, str) and _ENV_VAR_PATTERN.search(key):
        raise ConfigurationError(
            "Environment variable references are substituted in configuration values "
            "only, not in keys.",
            key=key,
        )
    return key


def _substitute_env_vars(value: str, *, credentials_allowed: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name not in os.environ:
            raise ConfigurationError(
                f"Missing environment variable {variable_name} referenced by configuration.",
                variable=variable_name,
            )
        resolved = os.environ[variable_name]
        if not credentials_allowed and looks_like_credential(resolved):
            # Names the variable, never its value: this message reaches logs and stderr.
            raise ConfigurationError(
                f"Environment variable {variable_name} holds a credential, which may only "
                f"be substituted inside a plugin '{_PLUGIN_CONFIG_KEY}' block. This field is "
                "recorded verbatim in run artifacts.",
                variable=variable_name,
            )
        return resolved

    return _ENV_VAR_PATTERN.sub(replace, value)


def format_validation_errors(exc: ValidationError) -> str:
    """Render a Pydantic error without echoing the values that failed.

    Pydantic's own ``str(exc)`` embeds ``input_value=...``. Handing that to an error context
    persisted a plugin block's contents -- the one place a credential legitimately lives --
    into ``error_message`` and from there into the SQLite index.
    """
    return _format_validation_errors(exc)


def _format_validation_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{_format_error_location(error.get('loc', ()))}: {error.get('msg', 'invalid value')}"
        for error in _simplify_validation_errors(exc)
    )


def _simplify_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "loc": _format_error_location(error.get("loc", ())),
            "msg": str(error.get("msg", "invalid value")),
            "type": str(error.get("type", "validation_error")),
        }
        for error in exc.errors(include_url=False, include_context=False, include_input=False)
    ]


def _format_error_location(location: object) -> str:
    if isinstance(location, tuple):
        location_tuple = cast(tuple[object, ...], location)
        return ".".join(str(part) for part in location_tuple)
    return str(location)
