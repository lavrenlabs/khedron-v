from __future__ import annotations

import math
import re
from typing import Any, TypeVar, cast

__all__ = ["REDACTED", "looks_like_credential", "redact_secrets", "scrub_credentials"]

REDACTED = "[REDACTED]"


def _canonical(text: str) -> str:
    """Lowercase and drop every separator so api_key, api-key and apiKey all coincide.

    Replacing separators with underscores was not enough: `privateKey` became
    `privatekey`, which matched no underscore-bearing marker, and the test that was meant
    to cover this passed only because `apiKey` happened to have a separator-less marker of
    its own. Collapsing both sides to one form removes the whole class.
    """
    return "".join(char for char in text if char.isalnum()).lower()


# The closed set of configuration fields whose string value is persisted verbatim. A field
# not listed here is redacted, so the cost of forgetting an entry is a `[REDACTED]` in an
# artifact -- visible and harmless -- rather than a live credential on disk.
#
# Adding a knob to a provider, benchmark or model plugin therefore means adding its name
# here if its value belongs in the artifacts. That is deliberate: it puts the decision at
# the point where someone can see what the value is.
_PUBLISHABLE_STRING_KEYS = frozenset(
    _canonical(key)
    for key in (
        # Suite-level provenance: where results went and under which methodology.
        "output_dir",
        "methodology_version",
        "methodology_profile",
        "seed_strategy",
        # Experiment identity and plugin selectors. `type` and `model` must survive because
        # the persisted experiment config is re-validated against ExperimentConfig and read
        # by reports; a redacted plugin name would make the artifact unusable.
        #
        # `note` is deliberately absent. It is the one field whose content the schema does
        # not constrain at all -- operator prose -- so vouching for it would reintroduce
        # exactly the guessing this allowlist exists to remove: "debug key sk-proj-..." does
        # not *start* with a credential prefix, so the value trap below would pass it
        # through. The note stays in the operator's own YAML; nothing reproducible is lost,
        # because reproducibility rides on the selectors and paths below.
        "name",
        "type",
        "model",
        "audit_mode",
        # Provider bag knobs that define the system under test.
        "binary_path",
        "sqlite_path",
        "project",
        "embedder",
        # Embedder bag knobs. The api_key entries alongside them are absent on purpose.
        "ollama_url",
        "ollama_model",
        "openai_model",
        "gemini_model",
        # Benchmark bag knobs that define the corpus and the question set measured. These are
        # the complete set LoCoMo accepts (`_locomo_constructor_args`, runner.py:1185); an
        # omission here silently strips reproducibility evidence from the artifact, which is
        # how `expected_dataset_checksum` -- the only record of *which* corpus was measured --
        # was being lost.
        "dataset_path",
        "audit_path",
        "expected_dataset_checksum",
        "conversations",
        "categories",
    )
)

# `*_env_var` fields hold the NAME of an environment variable (e.g. "OPENAI_API_KEY"),
# which is provenance rather than a secret, and the suffix is a convention rather than a
# fixed field list -- so it is matched as a rule instead of enumerated above.
_ENV_VAR_NAME_SUFFIX = _canonical("_env_var")

# Credential token *shapes*, matched anywhere in a string rather than at its start.
#
# Prefix matching had two failures that a shape does not. It missed an embedded token -- a
# provider's stderr saying `error: key sk-... rejected` is persisted as error context, and the
# value does not begin with the token -- and it forced a choice between missing Google keys and
# redacting the name "Aiza", because `AIza` is a real given name and a key prefix. Requiring
# the vendor marker *plus* a run of token characters resolves both: prose cannot satisfy it,
# and a short identifier cannot either. "multitask-v2" contains "sk-" followed by two
# characters, far below any threshold here, so model ids stay readable.
_CREDENTIAL_TOKEN_PATTERN = re.compile(
    "|".join(
        (
            r"sk-[A-Za-z0-9_-]{12,}",  # OpenAI, Anthropic
            r"gh[pousr]_[A-Za-z0-9]{20,}",  # GitHub tokens
            r"github_pat_[A-Za-z0-9_]{20,}",
            r"xox[baprs]-[A-Za-z0-9-]{10,}",  # Slack
            r"whsec_[A-Za-z0-9]{20,}",  # Stripe webhook secrets
            r"AIza[A-Za-z0-9_-]{30,}",  # Google API keys
            r"-----BEGIN",  # PEM blocks: private keys, certificates
        )
    ),
    re.IGNORECASE,
)

# Auth *scheme* words. Unlike the shapes above these are ordinary English, so they are matched
# only at the start of a value and only by the field-level checks, never by the persistence
# floor: a judge rationale beginning "Basic recall of the session..." must survive, and
# corrupting a measurement to protect it would defeat the point of recording it.
_AMBIGUOUS_SCHEME_PREFIXES = ("bearer ", "basic ")

T = TypeVar("T")


def scrub_credentials(value: T) -> T:
    """Redact unmistakable credential shapes anywhere in a record about to be persisted.

    The backstop at the persistence boundary. Six review rounds each found another channel by
    which operator-supplied text reached an artifact -- an arbitrary config bag, a mapping key,
    an experiment name, the model ids, an exception message, a structured error context -- and
    each fix closed one channel. This one runs where every record passes on its way to disk
    (``JsonlWriter.write``), so a channel nobody has thought of yet is covered too, including
    record types added later.

    It is deliberately weaker than the field-level policies rather than a replacement for
    them: it can only redact what is unmistakably a credential, because the records it walks
    also carry the measurement itself. An allowlist here would erase the corpus, the answers
    and the judgments -- the data the benchmark exists to record. Keys are scrubbed as well as
    values, since a mapping key is one of the channels this closes.

    Also normalises non-finite floats to null, matching what this writer produced before the
    scrub existed: redacting credentials must not quietly change what the artifacts mean.
    """
    return _scrub(value)


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {
            REDACTED if isinstance(key, str) and _contains_credential_token(key) else key: _scrub(
                inner_value
            )
            for key, inner_value in mapping.items()
        }
    if isinstance(value, list):
        return [_scrub(item) for item in cast(list[Any], value)]
    if isinstance(value, str) and _contains_credential_token(value):
        return REDACTED
    if isinstance(value, float) and not math.isfinite(value):
        # Preserves the serialization this writer had before the scrub was added:
        # `model_dump_json` renders a non-finite float as null, while
        # `json.dumps(model_dump(mode="json"))` emits bare `Infinity`, which strict JSON
        # parsers reject. Redacting credentials must not change what the artifacts mean.
        return None
    return value


def _contains_credential_token(value: str) -> bool:
    return _CREDENTIAL_TOKEN_PATTERN.search(value) is not None


def looks_like_credential(value: str) -> bool:
    """Report whether ``value`` carries a credential rather than configuration.

    Shared with configuration loading, which uses it to *refuse* a suite rather than redact
    one: some fields cannot be sanitised on the way out. ``ExperimentConfig.name`` is
    persisted raw into seven lifecycle and result records, projected into SQLite and shown by
    the dashboard, because reports and run selection are unusable without it -- so a
    credential there has to be rejected at the door instead.

    Wider than the persistence floor by exactly the ambiguous scheme words, which are safe to
    act on here because these fields hold identifiers and paths, not prose.
    """
    return _contains_credential_token(value) or value.lower().startswith(_AMBIGUOUS_SCHEME_PREFIXES)


def redact_secrets(value: T) -> T:
    """Project ``value`` onto the strings that are known to be safe to persist.

    Experiment configuration reaches the runner with ``${VAR}`` references already
    substituted, so persisting it verbatim writes live API keys into run artifacts.

    This is an allowlist, not a search for things that look secret. Three review rounds
    each found another key form a denylist had missed (``api-key``, then ``PRIVATE_KEY``
    inside an arbitrary env bag, then ``privateKey``), and the reason is structural:
    ``ProviderConfig.config``, ``BenchmarkConfig.config`` and the model ``config`` bags are
    ``dict[str, Any]`` by design, so no enumeration of secret-looking names can be complete
    over content the schema does not constrain. The polarity is therefore inverted -- a
    string is persisted only if its field is one of a small, reviewed set -- which makes the
    default outcome for anything unanticipated safe rather than unsafe.

    Non-string scalars (numbers, booleans, ``None``) pass through: they cannot carry a
    credential. Mapping keys are preserved as provenance -- knowing *which* variables a
    provider subprocess received is useful, their values are not. A bare string with no
    field name behind it is redacted, because no allowlist entry can vouch for it.
    """
    return _project(value, key=None)


def _project(value: Any, *, key: str | None) -> Any:
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {
            # Keys are provenance, so they are kept -- but a key that is itself a credential is
            # not provenance. Substitution into key position is already refused at load
            # (`config.py`); this covers one typed in literally, which that check cannot see.
            _redact_credential_key(inner_key): _project(inner_value, key=str(inner_key))
            for inner_key, inner_value in mapping.items()
        }
    if isinstance(value, list):
        # List items inherit the field name: `categories: [...]` is allowlisted as a field,
        # not per item.
        return [_project(item, key=key) for item in cast(list[Any], value)]
    if isinstance(value, str):
        return value if _is_publishable(value, key) else REDACTED
    return value


def _redact_credential_key(key: Any) -> Any:
    if isinstance(key, str) and looks_like_credential(key):
        return REDACTED
    return key


def _is_publishable(value: str, key: str | None) -> bool:
    if not value:
        return True  # nothing to leak, and an empty string is meaningful provenance
    if key is None:
        return False
    canonical = _canonical(key)
    if canonical not in _PUBLISHABLE_STRING_KEYS and not canonical.endswith(_ENV_VAR_NAME_SUFFIX):
        return False
    return not looks_like_credential(value)
