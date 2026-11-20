"""Tests for license policy and normalization."""

from __future__ import annotations

from grzybobranie_ml.license_policy import (
    is_allowed_license,
    is_ui_safe_license,
    license_family,
    normalize_license,
)


def test_normalize_allowed_licenses() -> None:
    assert normalize_license("CC0") == "cc0"
    assert is_allowed_license("cc0") is True
    assert normalize_license("CC0 1.0") == "cc0-1.0"
    assert normalize_license("CC BY") == "cc-by"
    assert normalize_license("CC BY 4.0") == "cc-by-4.0"
    assert normalize_license("CC BY-NC") == "cc-by-nc"
    assert normalize_license("CC BY-NC 4.0") == "cc-by-nc-4.0"
    assert normalize_license("CC BY-SA") == "cc-by-sa"
    assert normalize_license("CC BY-SA 4.0") == "cc-by-sa-4.0"
    assert normalize_license("pd-self") == "pd_self"
    assert normalize_license("pd_self") == "pd_self"


def test_normalize_with_license_url() -> None:
    url = "https://creativecommons.org/licenses/by/4.0/"
    assert normalize_license("CC BY", url) == "cc-by-4.0"

    nc_url = "https://creativecommons.org/licenses/by-nc/4.0/"
    assert normalize_license("CC BY-NC", nc_url) == "cc-by-nc-4.0"

    sa_url = "https://creativecommons.org/licenses/by-sa/3.0/"
    assert normalize_license("CC BY-SA", sa_url) == "cc-by-sa-3.0"

    zero_url = "https://creativecommons.org/publicdomain/zero/1.0/"
    assert normalize_license("CC0", zero_url) == "cc0-1.0"


def test_rejected_licenses() -> None:
    # NoDerivatives
    assert normalize_license("CC BY-ND") is None
    assert normalize_license("CC BY-ND 4.0") is None
    assert normalize_license("CC BY-NC-ND") is None

    # Danish Fungi
    assert normalize_license("DF20") is None
    assert normalize_license("df24") is None

    # Missing or placeholder
    assert normalize_license(None) is None
    assert normalize_license("") is None
    assert normalize_license("All Rights Reserved") is None


def test_ui_safe_classification() -> None:
    assert is_ui_safe_license("cc0-1.0") is True
    assert is_ui_safe_license("cc-by-4.0") is True
    assert is_ui_safe_license("cc-by") is True

    # Training only
    assert is_ui_safe_license("cc-by-nc-4.0") is False
    assert is_ui_safe_license("cc-by-sa-4.0") is False
    assert is_ui_safe_license("pd_self") is False
    assert is_ui_safe_license("unknown") is False


def test_license_family() -> None:
    assert license_family("cc0-1.0") == "cc0"
    assert license_family("cc-by-4.0") == "cc-by"
    assert license_family("cc-by-nc-4.0") == "cc-by-nc"
    assert license_family("cc-by-sa-3.0") == "cc-by-sa"
    assert license_family("pd_self") == "pd_self"
    assert license_family("invalid") is None
