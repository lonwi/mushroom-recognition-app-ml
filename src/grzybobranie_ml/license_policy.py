"""Licensing policy and validation for Grzybobranie AI ML dataset.

Adheres to DATA_CONTRACT.md and AGENTS.md:
- Training set may include: CC0, CC BY, CC BY-NC, CC BY-SA, pd_self.
- App UI display (ui_safe): strictly CC0 and CC BY.
- Forbidden: DF20, DF24 (images and weights), ND, AI-generated images, illustrations.
"""

from __future__ import annotations

import re

PUBLISHABLE = "publishable"
TRAINING_ONLY = "training_only"

# Closed set of allowable exact license keys
_ALLOWLIST_LICENSES: dict[str, tuple[str, str, bool]] = {
    # id: (usage, family, is_ui_safe)
    "cc0-1.0": (PUBLISHABLE, "cc0", True),
    "cc0": (PUBLISHABLE, "cc0", True),
    "cc-by-1.0": (PUBLISHABLE, "cc-by", True),
    "cc-by-2.0": (PUBLISHABLE, "cc-by", True),
    "cc-by-2.5": (PUBLISHABLE, "cc-by", True),
    "cc-by-3.0": (PUBLISHABLE, "cc-by", True),
    "cc-by-4.0": (PUBLISHABLE, "cc-by", True),
    "cc-by": (PUBLISHABLE, "cc-by", True),
    "cc-by-nc-1.0": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-nc-2.0": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-nc-2.5": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-nc-3.0": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-nc-4.0": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-nc": (TRAINING_ONLY, "cc-by-nc", False),
    "cc-by-sa-1.0": (TRAINING_ONLY, "cc-by-sa", False),
    "cc-by-sa-2.0": (TRAINING_ONLY, "cc-by-sa", False),
    "cc-by-sa-2.5": (TRAINING_ONLY, "cc-by-sa", False),
    "cc-by-sa-3.0": (TRAINING_ONLY, "cc-by-sa", False),
    "cc-by-sa-4.0": (TRAINING_ONLY, "cc-by-sa", False),
    "cc-by-sa": (TRAINING_ONLY, "cc-by-sa", False),
    "pd-self": (TRAINING_ONLY, "pd_self", False),
    "pd_self": (TRAINING_ONLY, "pd_self", False),
}

_REJECT_TOKENS = (
    "by-nd",
    "nc-nd",
    "non-derivative",
    "noderivatives",
    "all rights reserved",
    "all-rights-reserved",
    "df20",
    "df24",
    "danish fungi",
)

_CC_URL = re.compile(
    r"creativecommons\.org/(publicdomain/zero|licenses/([a-z0-9-]+))/(\d+\.\d+)",
    re.IGNORECASE,
)


def normalize_license(raw_license: str | None, raw_url: str | None = None) -> str | None:
    """Normalize textual license label and deed URL into canonical license key.

    Returns canonical key (e.g. 'cc-by-4.0', 'cc-by-nc-4.0', 'cc0-1.0', 'pd_self')
    or None if rejected / invalid.
    """
    if raw_license is None:
        return None
    lic_text = str(raw_license).strip().casefold().replace(" ", "-")
    if not lic_text:
        return None

    # Check for forbidden tokens
    for token in _REJECT_TOKENS:
        if token in lic_text:
            return None

    # Check for pd_self
    if lic_text in ("pd-self", "pd_self"):
        return "pd_self"

    # Try extracting exact version from license_url if present
    if raw_url:
        match = _CC_URL.search(str(raw_url).strip())
        if match:
            kind = match.group(1).lower()
            version = match.group(3)
            if kind == "publicdomain/zero":
                return f"cc0-{version}"
            code = match.group(2).lower()
            if code in ("by", "by-nc", "by-sa"):
                return f"cc-{code}-{version}"

    # Fallback to direct mapping from label
    if lic_text in _ALLOWLIST_LICENSES:
        return lic_text

    return None


def is_allowed_license(license_id: str | None) -> bool:
    """Check if the normalized license id is allowed for training."""
    if not license_id:
        return False
    return license_id.casefold() in _ALLOWLIST_LICENSES


def is_ui_safe_license(license_id: str | None) -> bool:
    """Check if the license is allowed for display in the app UI."""
    if not license_id:
        return False
    info = _ALLOWLIST_LICENSES.get(license_id.casefold())
    return info[2] if info else False


def license_family(license_id: str | None) -> str | None:
    """Return the license family (cc0, cc-by, cc-by-nc, cc-by-sa, pd_self)."""
    if not license_id:
        return None
    info = _ALLOWLIST_LICENSES.get(license_id.casefold())
    return info[1] if info else None
