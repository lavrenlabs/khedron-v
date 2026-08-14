from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from structlog.testing import capture_logs

from khedron.config import (
    AnswerModelConfig,
    BenchmarkConfig,
    ExperimentConfig,
    ExperimentSuiteConfig,
    JudgeConfig,
    ProviderConfig,
    load_suite_config,
    warn_on_same_vendor_judges,
)
from khedron.errors import ConfigurationError

# Every configuration model that carries operator-supplied strings into run records. The sweep
# below asserts it covers all of them, so adding a model without covering it fails.
_CONFIG_MODELS = {
    ProviderConfig,
    BenchmarkConfig,
    AnswerModelConfig,
    JudgeConfig,
    ExperimentConfig,
    ExperimentSuiteConfig,
}


def write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def valid_config_yaml(
    *,
    answer_type: str = "openai",
    judge_type: str = "anthropic",
    methodology_profile: str | None = "canonical-v2",
    extra_suite_fields: str = "",
    extra_experiment_fields: str = "",
) -> str:
    """A configuration that is valid under the *current* contract.

    Moved from `canonical-v1` when that profile stopped being usable. The values below are not
    decoration: `canonical-v2` pins the models, the retrieval budget and the evaluated category
    set, so a fixture diverging from any of them is no longer a valid config -- and these tests
    are about loading valid configs.
    """
    profile_line = (
        "" if methodology_profile is None else f'methodology_profile: "{methodology_profile}"\n'
    )
    return f"""\
runs: 2
methodology_version: "2.0"
{profile_line}{extra_suite_fields}experiments:
  - name: "Synthetic experiment"
    note: "Unit-test config"
    provider:
      type: full_context
      config:
        max_memories_per_search: 10
    benchmark:
      type: locomo
      config:
        dataset_path: data/locomo/synthetic.json
        categories:
          - single_hop
          - multi_hop
          - temporal
          - open_domain
          - adversarial
      audit_mode: both
    answer_model:
      type: {answer_type}
      model: gpt-4o-mini-2024-07-18
      temperature: 0
      max_output_tokens: 512
    judge:
      type: {judge_type}
      model: claude-haiku-4-5-20251001
      temperature: 0
    top_k_retrieval: 200
    max_concurrent_questions: 2
{extra_experiment_fields}"""


def test_valid_yaml_parses_successfully(tmp_path: Path) -> None:
    path = write_config(tmp_path, valid_config_yaml())

    config = load_suite_config(path)

    if not isinstance(config, ExperimentSuiteConfig):
        raise AssertionError(config)
    if config.runs != 2:
        raise AssertionError(config.runs)
    if config.experiments[0].name != "Synthetic experiment":
        raise AssertionError(config.experiments[0])


def test_defaults_are_applied() -> None:
    # Validated directly rather than through `load_suite_config`, because there is no longer such a
    # thing as a minimal *loadable* config: the default profile pins models, retrieval budget and
    # category set, so a config omitting them is correctly refused. This test is about field
    # defaults, so it asserts them without dragging in methodology conformance.
    config = ExperimentSuiteConfig.model_validate(
        {
            "experiments": [
                {
                    "name": "Defaults",
                    "provider": {"type": "full_context"},
                    "benchmark": {"type": "locomo"},
                    "answer_model": {"type": "openai", "model": "answer-model"},
                    "judge": {"type": "anthropic", "model": "judge-model"},
                }
            ]
        }
    )
    experiment = config.experiments[0]

    if config.runs != 3:
        raise AssertionError(config.runs)
    if config.seed_strategy != "deterministic":
        raise AssertionError(config.seed_strategy)
    if config.max_cost_usd is not None:
        raise AssertionError(config.max_cost_usd)
    if config.output_dir != "results":
        raise AssertionError(config.output_dir)
    if config.methodology_version != "2.0":
        raise AssertionError(config.methodology_version)
    if config.methodology_profile != "canonical-v2":
        raise AssertionError(config.methodology_profile)
    if experiment.benchmark.audit_mode != "both":
        raise AssertionError(experiment.benchmark.audit_mode)
    if experiment.answer_model.temperature != 0.0:
        raise AssertionError(experiment.answer_model.temperature)
    if experiment.answer_model.max_output_tokens != 1024:
        raise AssertionError(experiment.answer_model.max_output_tokens)
    if experiment.judge.temperature != 0.0:
        raise AssertionError(experiment.judge.temperature)
    if experiment.top_k_retrieval != 10:
        raise AssertionError(experiment.top_k_retrieval)
    if experiment.max_concurrent_questions != 5:
        raise AssertionError(experiment.max_concurrent_questions)


def test_invalid_yaml_syntax_raises_configuration_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, "runs: [\n")

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    if "Invalid YAML configuration syntax" not in str(exc_info.value):
        raise AssertionError(str(exc_info.value))


def test_missing_required_fields_raise_configuration_error(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """\
experiments:
  - name: "Missing answer model"
    provider:
      type: full_context
    benchmark:
      type: locomo
    judge:
      type: anthropic
      model: judge-model
""",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "Invalid suite configuration" not in rendered:
        raise AssertionError(rendered)
    if "answer_model" not in rendered:
        raise AssertionError(rendered)


@pytest.mark.parametrize("yaml_content", ["", "[]\n", "plain scalar\n"])
def test_empty_and_non_mapping_yaml_raise_configuration_error(
    tmp_path: Path,
    yaml_content: str,
) -> None:
    path = write_config(tmp_path, yaml_content)

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    is_expected_empty_error = "Configuration file is empty" in rendered
    is_expected_type_error = "mapping at the document root" in rendered
    if not is_expected_empty_error and not is_expected_type_error:
        raise AssertionError(rendered)


def test_environment_variable_substitution_is_recursive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESULTS_DIR", "custom-results")
    monkeypatch.setenv("DATA_ROOT", "data/locomo")
    monkeypatch.setenv("ANSWER_MODEL", "gpt-4o-mini-2024-07-18")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "777")

    path = write_config(
        tmp_path,
        """\
output_dir: "${RESULTS_DIR}/suite"
methodology_version: "2.0"
experiments:
  - name: "Recursive env"
    provider:
      type: full_context
      config:
        search_paths:
          - "${DATA_ROOT}/a.json"
          - nested: "${DATA_ROOT}/b.json"
    benchmark:
      type: locomo
      config:
        dataset_path: "${DATA_ROOT}/synthetic.json"
        categories:
          - single_hop
          - multi_hop
          - temporal
          - open_domain
          - adversarial
      audit_mode: both
    answer_model:
      type: openai
      model: "${ANSWER_MODEL}"
      max_output_tokens: "${MAX_OUTPUT_TOKENS}"
    judge:
      type: anthropic
      model: claude-haiku-4-5-20251001
    top_k_retrieval: 200
""",
    )

    config = load_suite_config(path)
    experiment = config.experiments[0]
    provider_config = experiment.provider.config
    search_paths = provider_config["search_paths"]

    if config.output_dir != "custom-results/suite":
        raise AssertionError(config.output_dir)
    if experiment.benchmark.config["dataset_path"] != "data/locomo/synthetic.json":
        raise AssertionError(experiment.benchmark.config)
    if experiment.answer_model.model != "gpt-4o-mini-2024-07-18":
        raise AssertionError(experiment.answer_model.model)
    if experiment.answer_model.max_output_tokens != 777:
        raise AssertionError(experiment.answer_model.max_output_tokens)
    if not isinstance(search_paths, list):
        raise AssertionError(search_paths)
    if search_paths[0] != "data/locomo/a.json":
        raise AssertionError(search_paths)
    nested = search_paths[1]
    if not isinstance(nested, dict):
        raise AssertionError(search_paths)
    if nested["nested"] != "data/locomo/b.json":
        raise AssertionError(nested)


def test_missing_environment_variable_raises_configuration_error(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        valid_config_yaml().replace("gpt-4o-mini-2024-07-18", "${MISSING_CONFIG_MODEL}"),
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "MISSING_CONFIG_MODEL" not in rendered:
        raise AssertionError(rendered)


def test_env_var_reference_in_a_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Substituting keys turned a resolved credential into a config *key*, and persistence
    # keeps keys as provenance -- so the secret was written to artifacts past a redaction
    # that only inspects values. Refusing at load time closes the vector at its source.
    monkeypatch.setenv("CONFIG_KEY_SECRET", "sk-proj-should-never-become-a-key")
    path = write_config(
        tmp_path,
        valid_config_yaml().replace(
            "        max_memories_per_search: 10",
            '        "${CONFIG_KEY_SECRET}": value',
            1,
        ),
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "keys" not in rendered:
        raise AssertionError(rendered)
    # The error must not itself print the resolved secret.
    if "sk-proj-should-never-become-a-key" in rendered:
        raise AssertionError(rendered)


def test_a_credential_interpolated_outside_a_plugin_config_block_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `name` reaches artifacts as itself, not inside the redacted config blob: RunStartedEvent
    # copies it verbatim, SQLite projects it and the dashboard shows it. Reports and run
    # selection need it, so it cannot be sanitised on the way out -- only refused on the way in.
    # Note the interpolation into surrounding text: the resolved value starts with "debug", so
    # no shape test applied to the finished string would recognise it. The check therefore runs
    # per substituted value.
    monkeypatch.setenv("CONFIG_NAME_SECRET", "sk-proj-must-never-label-a-run")
    path = write_config(
        tmp_path,
        valid_config_yaml().replace(
            '- name: "Synthetic experiment"', '- name: "debug ${CONFIG_NAME_SECRET}"', 1
        ),
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "CONFIG_NAME_SECRET" not in rendered:
        raise AssertionError(rendered)
    # Naming the variable is provenance; printing its value would be the leak itself.
    if "sk-proj-must-never-label-a-run" in rendered:
        raise AssertionError(rendered)


def test_a_credential_substituted_into_a_model_id_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # answer_model.model and judge.model are the same channel as `name`: persisted raw as
    # answer_model_id / judge_model_id, projected to SQLite, rendered by reports and the
    # dashboard. They were missed while the guard was a list of the fields already reported.
    monkeypatch.setenv("CONFIG_MODEL_SECRET", "sk-proj-must-never-name-a-model")
    path = write_config(
        tmp_path,
        valid_config_yaml().replace(
            "      model: gpt-4o-mini-2024-07-18", '      model: "${CONFIG_MODEL_SECRET}"', 1
        ),
    )

    with pytest.raises(ConfigurationError):
        load_suite_config(path)


def test_a_credential_inside_a_plugin_config_block_still_substitutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The capability this must not break: a plugin config block is exactly where a credential
    # belongs -- a provider config passes OPENAI_API_KEY this way -- and
    # redaction keeps those values out of artifacts.
    monkeypatch.setenv("CONFIG_PROVIDER_SECRET", "sk-proj-legitimately-passed-through")
    path = write_config(
        tmp_path,
        valid_config_yaml().replace(
            "        max_memories_per_search: 10",
            '        openai_api_key: "${CONFIG_PROVIDER_SECRET}"',
            1,
        ),
    )

    config = load_suite_config(path)

    if config.experiments[0].provider.config["openai_api_key"] != (
        "sk-proj-legitimately-passed-through"
    ):
        raise AssertionError(config.experiments[0].provider.config)


def test_non_credential_substitution_remains_available_everywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Confining *substitution* to plugin blocks would have closed the leak too, at the cost of
    # a legitimate capability: parameterising an output directory or a model id from the
    # environment. Only credential-valued substitutions are confined.
    monkeypatch.setenv("CONFIG_RESULTS_DIR", "custom-results")
    monkeypatch.setenv("CONFIG_ANSWER_MODEL", "gpt-4o-mini-2024-07-18")
    path = write_config(
        tmp_path,
        valid_config_yaml(extra_suite_fields='output_dir: "${CONFIG_RESULTS_DIR}/suite"\n').replace(
            "      model: gpt-4o-mini-2024-07-18", '      model: "${CONFIG_ANSWER_MODEL}"', 1
        ),
    )

    config = load_suite_config(path)

    if config.output_dir != "custom-results/suite":
        raise AssertionError(config.output_dir)
    if config.experiments[0].answer_model.model != "gpt-4o-mini-2024-07-18":
        raise AssertionError(config.experiments[0].answer_model.model)


def test_a_credential_pasted_into_an_experiment_name_is_refused(tmp_path: Path) -> None:
    # The companion case: typed in literally rather than referenced through a variable.
    path = write_config(
        tmp_path,
        valid_config_yaml().replace(
            '- name: "Synthetic experiment"', '- name: "sk-ant-api03-pasted-here"', 1
        ),
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    if "credential" not in str(exc_info.value):
        raise AssertionError(exc_info.value)


def test_every_declared_string_field_of_every_config_model_refuses_a_credential() -> None:
    # The anti-drift test. Four review rounds each found another field of this same class --
    # an arbitrary bag, a mapping key, `name`, then the model ids -- because both the guard and
    # its tests were lists of the fields somebody had already noticed. This enumerates the
    # fields from the schema at runtime, so a field added later is covered without anyone
    # remembering to extend it, and a field that stops being guarded fails here.
    experiment_payload: dict[str, Any] = {
        "name": "probe",
        "provider": {"type": "full_context", "config": {}},
        "benchmark": {"type": "locomo", "config": {}},
        "answer_model": {"type": "openai", "model": "gpt-4o-mini"},
        "judge": {"type": "anthropic", "model": "claude-haiku-4-5"},
    }
    payloads: list[tuple[type[BaseModel], dict[str, Any]]] = [
        (ProviderConfig, {"type": "full_context", "config": {}}),
        (BenchmarkConfig, {"type": "locomo", "config": {}}),
        (AnswerModelConfig, {"type": "openai", "model": "gpt-4o-mini"}),
        (JudgeConfig, {"type": "anthropic", "model": "claude-haiku-4-5"}),
        (ExperimentConfig, experiment_payload),
        (ExperimentSuiteConfig, {"experiments": [experiment_payload]}),
    ]
    if {model for model, _ in payloads} != _CONFIG_MODELS:
        raise AssertionError(
            "a configuration model is missing from this sweep: "
            f"{_CONFIG_MODELS - {model for model, _ in payloads}}"
        )

    canary = "sk-proj-canary-must-never-be-accepted"
    for model_cls, payload in payloads:
        valid = model_cls.model_validate(payload)
        # `config` is exempt by design: a credential belongs in a plugin block, and redaction
        # keeps it out of artifacts.
        string_fields = [
            name
            for name in model_cls.model_fields
            if name != "config" and isinstance(getattr(valid, name, None), str)
        ]
        if not string_fields:
            raise AssertionError(
                f"{model_cls.__name__}: introspection found no string fields, so this test "
                "would pass without checking anything"
            )
        for field_name in string_fields:
            try:
                model_cls.model_validate({**payload, field_name: canary})
            except ValidationError:
                continue
            raise AssertionError(f"{model_cls.__name__}.{field_name} accepted a credential")


def test_a_credential_identity_is_refused_on_the_programmatic_path_too() -> None:
    # A suite built in process -- Runner(ExperimentSuiteConfig.model_validate(...)) -- never
    # passes through load_suite_config, so a check living only in the loader would have left
    # this path open. The guard is a field validator for exactly that reason.
    payload = {
        "experiments": [
            {
                "name": "sk-proj-pasted-into-a-programmatic-config",
                "provider": {"type": "full_context", "config": {}},
                "benchmark": {"type": "locomo", "config": {}},
                "answer_model": {"type": "openai", "model": "gpt-4o-mini"},
                "judge": {"type": "anthropic", "model": "claude-haiku-4-5"},
            }
        ]
    }

    with pytest.raises(ValidationError) as exc_info:
        ExperimentSuiteConfig.model_validate(payload)

    if "credential" not in str(exc_info.value):
        raise AssertionError(exc_info.value)


def test_unsupported_methodology_profile_fails_with_clear_configuration_error(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, valid_config_yaml(methodology_profile="unknown-profile"))

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "methodology_profile" not in rendered:
        raise AssertionError(rendered)
    if "unknown-profile" in rendered:
        raise AssertionError(rendered)


def test_same_vendor_answer_and_judge_warns_but_remains_valid() -> None:
    # No longer reachable through `load_suite_config`: `canonical-v2` pins an OpenAI answer model
    # and an Anthropic judge, so a same-vendor suite is refused by the profile before the warning
    # could fire. That is the methodology's P3 bias rule enforced rather than merely announced.
    #
    # The warning still has a job -- a future profile that leaves the vendors unpinned would need
    # it -- so it is exercised directly instead of being deleted along with the path that reached
    # it, which would have quietly dropped the coverage.
    config = ExperimentSuiteConfig.model_validate(
        {
            "experiments": [
                {
                    "name": "Same vendor",
                    "provider": {"type": "full_context"},
                    "benchmark": {"type": "locomo"},
                    "answer_model": {"type": "openai", "model": "gpt-4o-mini-2024-07-18"},
                    "judge": {"type": "openai", "model": "gpt-4o"},
                }
            ]
        }
    )

    with capture_logs() as logs:
        warn_on_same_vendor_judges(config)

    if config.experiments[0].judge.type != "openai":
        raise AssertionError(config.experiments[0])
    warning_logs = [
        log
        for log in logs
        if log.get("event") == "same_vendor_configuration" and log.get("log_level") == "warning"
    ]
    if len(warning_logs) != 1:
        raise AssertionError(logs)
    if warning_logs[0].get("vendor") != "openai":
        raise AssertionError(warning_logs)


def test_frozen_config_model_mutation_raises_validation_error(tmp_path: Path) -> None:
    config = load_suite_config(write_config(tmp_path, valid_config_yaml()))

    with pytest.raises(ValidationError):
        config.runs = 4


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        ("runs: 2", "runs: 0"),
        ("top_k_retrieval: 200", "top_k_retrieval: -1"),
        ("max_concurrent_questions: 2", "max_concurrent_questions: 0"),
        ("max_output_tokens: 512", "max_output_tokens: 0"),
        ("temperature: 0", "temperature: -0.1"),
    ],
)
def test_numeric_validation_rejects_invalid_values(
    tmp_path: Path,
    field_path: str,
    replacement: str,
) -> None:
    path = write_config(tmp_path, valid_config_yaml().replace(field_path, replacement, 1))

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    rendered = str(exc_info.value)
    if "Invalid suite configuration" not in rendered:
        raise AssertionError(rendered)


def test_max_cost_must_be_non_negative(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        valid_config_yaml(extra_suite_fields="max_cost_usd: -0.01\n"),
    )

    with pytest.raises(ConfigurationError) as exc_info:
        load_suite_config(path)

    if "max_cost_usd" not in str(exc_info.value):
        raise AssertionError(str(exc_info.value))
