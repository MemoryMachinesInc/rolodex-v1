"""Keep the suite hermetic against a configured machine.

``settings.resolve`` discovers ``configs/rolodex-v1.toml`` by walking up from the
working directory, and this checkout is exactly where a developer is told to put
one.
Without this fixture the suite would read whatever corpus that file points at,
and a test would pass or fail depending on whose laptop it ran on -- the same
hazard the explicit paths in ``test_build_profiles.py`` already guard against,
closed here so a future test cannot reopen it by forgetting.

The environment variables get the same treatment: an exported
``ROLODEX_V1_SOURCE_DOCS`` would otherwise reach into a run under test.
"""

from __future__ import annotations

import pytest

from rolodex_v1 import settings

LEAKY = (
    settings.ENV_ENTITIES_BUNDLE,
    settings.ENV_SOURCE_DOCS,
)


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable config discovery and path environment variables for every test.

    A test that wants a config file passes ``--config`` or sets
    ``ROLODEX_V1_CONFIG`` itself, which still wins -- this only removes what the
    machine would have contributed on its own.
    """
    monkeypatch.setenv(settings.ENV_CONFIG, "")
    for name in LEAKY:
        monkeypatch.delenv(name, raising=False)
