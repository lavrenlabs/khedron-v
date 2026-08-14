from __future__ import annotations

from khedron.utils.redaction import REDACTED, redact_secrets, scrub_credentials


def test_redacts_an_unlisted_field_by_default() -> None:
    # The property that matters: a field nobody anticipated is redacted because it is not on
    # the allowlist, with no need for its name to resemble a credential. Every earlier round
    # of this file failed on a name a denylist had not thought of.
    result = redact_secrets({"pat": "x", "sig": "y", "cookie": "z", "totally_new_knob": "w"})
    if set(result.values()) != {REDACTED}:
        raise AssertionError(result)


def test_keeps_allowlisted_configuration_fields() -> None:
    result = redact_secrets(
        {
            "name": "stage1",
            "type": "openai",
            "model": "gpt-4o-mini",
            "audit_mode": "both",
            "embedder": "ollama",
            "ollama_url": "http://localhost:11434",
            "categories": ["multi_hop", "temporal"],
        }
    )
    expected = {
        "name": "stage1",
        "type": "openai",
        "model": "gpt-4o-mini",
        "audit_mode": "both",
        "embedder": "ollama",
        "ollama_url": "http://localhost:11434",
        "categories": ["multi_hop", "temporal"],
    }
    if result != expected:
        raise AssertionError(result)


def test_redacts_credential_keys_in_every_separator_and_case_style() -> None:
    # These all fell to the allowlist rather than to name matching, but they are the exact
    # forms that leaked in rounds 1-3, so they stay as regression coverage.
    for key in (
        "openai_api_key",
        "api-key",
        "apiKey",
        "API_KEY",
        "privateKey",
        "accessKey",
        "sessionToken",
        "clientSecret",
        "PRIVATE_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ):
        result = redact_secrets({key: "some-live-value"})
        if result[key] != REDACTED:
            raise AssertionError(f"{key} was not redacted: {result}")


def test_redacts_arbitrary_env_bags_under_any_spelling() -> None:
    # additional_env, additionalEnv or any other arbitrary bag: no special case is needed
    # now, because an unlisted field is redacted by the general rule. Keys stay, because
    # knowing WHICH variables a provider subprocess received is the provenance that matters.
    for bag_name in ("additional_env", "additionalEnv", "env", "some_other_bag"):
        result = redact_secrets(
            {"provider": {"config": {bag_name: {"PRIVATE_KEY": "x", "AWS_ACCESS_KEY_ID": "y"}}}}
        )
        bag = result["provider"]["config"][bag_name]
        if set(bag) != {"PRIVATE_KEY", "AWS_ACCESS_KEY_ID"}:
            raise AssertionError(f"keys must be preserved as provenance: {bag}")
        if set(bag.values()) != {REDACTED}:
            raise AssertionError(bag)


def test_redacts_free_text_notes_including_embedded_credentials() -> None:
    # `note` is operator prose, so it is off the allowlist entirely. Trusting it to the
    # value trap was a real hole: the trap matches by prefix, and "debug key sk-proj-..."
    # does not start with one -- it would have been persisted verbatim.
    for value in (
        "sk-ant-api03-at-the-start",
        "debug key sk-proj-embedded-mid-string",
        "stage 1 baseline",
    ):
        result = redact_secrets({"note": value})
        if result["note"] != REDACTED:
            raise AssertionError(result)


def test_redacts_a_whole_credential_sitting_in_an_allowlisted_field() -> None:
    # The trap that remains: an allowlisted field whose entire value is plainly a credential.
    for value in ("sk-ant-api03-lives-here", "Bearer eyJhbGciOi", "-----BEGIN PRIVATE KEY-----"):
        result = redact_secrets({"model": value})
        if result["model"] != REDACTED:
            raise AssertionError(result)


def test_keeps_identifiers_that_merely_contain_a_prefix_substring() -> None:
    # Why the trap matches by prefix and not substring: "multitask-v2" contains "sk-", and
    # redacting a model id would make the artifact unusable for re-validation and reports.
    result = redact_secrets({"model": "multitask-v2", "dataset_path": "data/basic/set.json"})
    if result != {"model": "multitask-v2", "dataset_path": "data/basic/set.json"}:
        raise AssertionError(result)


def test_keeps_the_full_locomo_provenance_set() -> None:
    # These are exactly the keys LoCoMo accepts. Omitting one silently strips reproducibility
    # evidence -- expected_dataset_checksum is the only record of which corpus was measured.
    config = {
        "dataset_path": "data/locomo/locomo10.json",
        "audit_path": "data/locomo/audit.json",
        "expected_dataset_checksum": "9f2b" + "0" * 60,
        "conversations": ["conv-1", "conv-2"],
        "categories": ["multi_hop"],
    }
    if redact_secrets(config) != config:
        raise AssertionError(redact_secrets(config))


def test_keeps_env_var_names() -> None:
    # `*_env_var` holds the NAME of a variable, which is useful provenance, not a secret.
    result = redact_secrets({"api_key_env_var": "OPENAI_API_KEY"})
    if result["api_key_env_var"] != "OPENAI_API_KEY":
        raise AssertionError(result)


def test_walks_nested_structures() -> None:
    result = redact_secrets(
        {
            "experiments": [
                {
                    "provider": {
                        "config": {
                            "binary_path": "provider",
                            "embedder_config": {
                                "openai_api_key": "sk-proj-x",
                                "ollama_model": "nomic-embed-text",
                            },
                        }
                    }
                }
            ]
        }
    )
    config = result["experiments"][0]["provider"]["config"]
    if config["embedder_config"]["openai_api_key"] != REDACTED:
        raise AssertionError(result)
    if config["embedder_config"]["ollama_model"] != "nomic-embed-text":
        raise AssertionError(result)
    if config["binary_path"] != "provider":
        raise AssertionError(result)


def test_keeps_non_string_scalars_and_redacts_bare_strings() -> None:
    # Numbers, booleans and None cannot carry a credential, so they pass through untouched --
    # which is what keeps top_k_retrieval, temperature and the guard thresholds readable in
    # artifacts. A string with no field name behind it has no allowlist entry to vouch for it.
    payload = {"runs": 3, "temperature": 0.0, "enabled": True, "none": None}
    if redact_secrets(payload) != payload:
        raise AssertionError(redact_secrets(payload))
    if redact_secrets("a-bare-string") != REDACTED:
        raise AssertionError(redact_secrets("a-bare-string"))


def test_scrub_redacts_credential_keys_and_values_in_any_structure() -> None:
    # The boundary backstop covers keys too: a credential typed in as a mapping key inside a
    # plugin block is neither a declared field nor an env-var reference, so no earlier rule saw
    # it, and redaction preserves keys as provenance.
    scrubbed = scrub_credentials(
        {"sk-proj-a-key-as-a-key": "v", "nested": [{"token": "sk-proj-a-value"}]}
    )
    if "sk-proj-a-key-as-a-key" in scrubbed:
        raise AssertionError(scrubbed)
    if scrubbed["nested"][0]["token"] != REDACTED:
        raise AssertionError(scrubbed)


def test_scrub_leaves_measurement_text_and_non_strings_alone() -> None:
    payload = {
        "answer": "Basic recall of the first session",
        "rationale": "Bearer in mind the June conversation",
        "score": 0.5,
        "n": 3,
        "ok": True,
    }
    if scrub_credentials(payload) != payload:
        raise AssertionError(scrub_credentials(payload))


def test_scrub_finds_a_token_embedded_in_a_diagnostic_blob() -> None:
    # The real shape of the channel: a provider's stderr, persisted as error context. The
    # value does not *start* with the token, so prefix matching walked straight past it.
    blob = "provider: request rejected\n  detail: key sk-proj-abcdefghijklmnop invalid\n  exit 1"
    if scrub_credentials({"stderr": blob})["stderr"] != REDACTED:
        raise AssertionError(scrub_credentials({"stderr": blob}))


def test_scrub_distinguishes_a_google_key_from_the_name_it_starts_with() -> None:
    # `AIza` is both a Google API key prefix and a given name, so a bare prefix test forced a
    # choice between missing the key and redacting a person out of the corpus. Requiring the
    # key's length removes the choice.
    google_key = "AIza" + "b3F9x" * 7
    if scrub_credentials({"detail": google_key})["detail"] != REDACTED:
        raise AssertionError(google_key)
    for innocent in ("Aiza", "Aiza said she would call", "Aizawa"):
        if scrub_credentials({"answer": innocent})["answer"] != innocent:
            raise AssertionError(innocent)


def test_scrub_keeps_short_identifiers_that_contain_a_marker() -> None:
    # A model id may legitimately contain "sk-"; redacting it would make artifacts unusable
    # for re-validation and reports.
    payload = {"model": "multitask-v2", "path": "data/basic/set.json", "note_like": "task-sk-1"}
    if scrub_credentials(payload) != payload:
        raise AssertionError(scrub_credentials(payload))
