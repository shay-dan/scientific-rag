"""Parsing and validation of configured API credentials.

Kept free of web-framework imports so startup validation and request-time
authentication share one definition of a valid credential.
Error messages name entry positions and identities, never key material.
"""
from __future__ import annotations

from dataclasses import dataclass
import hmac

SUPPORTED_ROLES = frozenset({"admin", "user"})
# Identity reserved for HEALTH_COACH_API_KEY; the evidence-job endpoint authorizes it by name.
RESERVED_IDENTITIES = frozenset({"health_coach"})
# Values shipped in .env.example or used as code defaults. Comparison is case-insensitive.
KNOWN_PLACEHOLDERS = frozenset({
    "replace-with-a-long-random-access-key",
    "replace-with-a-different-long-random-service-key",
    "replace-with-a-different-32-character-minimum-secret",
    "development-only-change-me",
    "changeme",
    "change-me",
})
_PLACEHOLDER_PREFIXES = ("replace-with-", "replace_with_", "your-", "your_", "<")


class CredentialConfigurationError(ValueError):
    """Raised for unsafe or malformed credential configuration."""


@dataclass(frozen=True)
class ApiCredential:
    name: str
    role: str
    key: str


def is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in KNOWN_PLACEHOLDERS or lowered.startswith(_PLACEHOLDER_PREFIXES)


def _is_printable_ascii(value: str) -> bool:
    return all(0x21 <= ord(char) <= 0x7E for char in value)


def validate_secret_value(label: str, value: str) -> None:
    """Shared checks for any secret: non-empty, no padding, printable ASCII, not a placeholder."""
    if not value or not value.strip():
        raise CredentialConfigurationError(f"{label} must not be empty or whitespace")
    if value != value.strip():
        raise CredentialConfigurationError(f"{label} must not have leading or trailing whitespace")
    if not _is_printable_ascii(value):
        raise CredentialConfigurationError(f"{label} must contain only printable ASCII characters without spaces")
    if is_placeholder(value):
        raise CredentialConfigurationError(f"{label} still contains a placeholder value; set a real secret")


def parse_api_keys(configured: str, service_key: str = "") -> tuple[ApiCredential, ...]:
    """Parse APP_API_KEYS (``name[|role]:key`` entries separated by commas).

    Whitespace before an entry (after a comma or at the start) is ignored, as
    are empty or whitespace-only segments such as a trailing comma. Whitespace
    anywhere else belongs to the identity or key and is rejected, never
    normalized away. Every malformed or unsafe entry raises
    CredentialConfigurationError.
    """
    credentials: list[ApiCredential] = []
    for position, raw_entry in enumerate(configured.split(","), start=1):
        if not raw_entry.strip():
            continue
        entry = raw_entry.lstrip()
        label = f"APP_API_KEYS entry {position}"
        if ":" not in entry:
            raise CredentialConfigurationError(f"{label} must have the form name[|role]:key")
        identity, key = entry.split(":", 1)
        name, separator, declared_role = identity.partition("|")
        if not name or not _is_printable_ascii(name):
            raise CredentialConfigurationError(f"{label} has an empty or invalid identity name")
        if name in RESERVED_IDENTITIES:
            raise CredentialConfigurationError(f"{label} uses reserved identity {name!r}; configure it with HEALTH_COACH_API_KEY")
        if separator and declared_role not in SUPPORTED_ROLES:
            raise CredentialConfigurationError(f"{label} ({name}) declares an unsupported role; use one of {sorted(SUPPORTED_ROLES)}")
        validate_secret_value(f"{label} ({name}) key", key)
        role = declared_role or ("admin" if name == "admin" else "user")
        credentials.append(ApiCredential(name=name, role=role, key=key))

    if not credentials:
        raise CredentialConfigurationError("APP_API_KEYS must contain at least one name[|role]:key entry")
    seen_keys: dict[str, str] = {}
    roles_by_name: dict[str, str] = {}
    for credential in credentials:
        if credential.key in seen_keys:
            raise CredentialConfigurationError(
                f"APP_API_KEYS assigns the same key to {seen_keys[credential.key]!r} and {credential.name!r}"
            )
        seen_keys[credential.key] = credential.name
        if roles_by_name.setdefault(credential.name, credential.role) != credential.role:
            raise CredentialConfigurationError(f"APP_API_KEYS gives identity {credential.name!r} conflicting roles")
    if service_key and service_key in seen_keys:
        raise CredentialConfigurationError("HEALTH_COACH_API_KEY must differ from every APP_API_KEYS key")
    return tuple(credentials)


def keys_match(presented: str, expected: str) -> bool:
    """Constant-time comparison that never raises on non-ASCII input and never matches empty keys."""
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented.encode("utf-8", "surrogatepass"), expected.encode("utf-8", "surrogatepass"))
