from __future__ import annotations

import sys

__all__ = ["configure_os_trust_store"]

_configured = False


def configure_os_trust_store() -> bool:
    """Route Python TLS verification through the operating system trust store.

    The OpenAI and Anthropic SDKs (via httpx) verify server certificates against
    certifi's Mozilla bundle, which does not contain enterprise or antivirus
    TLS-inspection root CAs (for example AVG's "Web/Mail Shield" root). On a machine
    behind such a proxy every HTTPS call fails with CERTIFICATE_VERIFY_FAILED even
    though the OS trusts the intercepting CA. ``truststore`` makes the standard ``ssl``
    module verify against the OS trust store, which does include those roots, so a
    benchmark run survives TLS interception without ever disabling verification.

    Must run before the model clients build their SSL contexts; the CLI invokes it at
    startup. Idempotent, and best-effort: if ``truststore`` is missing or fails to
    inject, certifi verification is left in place rather than crashing the run. Returns
    ``True`` when the OS trust store is active.

    The fallback notice goes to stderr, never stdout: the CLI runs this before every
    command, and documented automation parses stdout (for example
    ``inspect suite --field total_cost_usd``), which an extra line would corrupt.
    """
    global _configured
    if _configured:
        return True
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as exc:  # trust-store setup must never crash a run
        print(
            f"warning: OS trust store unavailable, using certifi verification ({exc})",
            file=sys.stderr,
        )
        return False
    _configured = True
    return True
