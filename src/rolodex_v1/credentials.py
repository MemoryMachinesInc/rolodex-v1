"""Read API credentials from the environment or the repo's ``.env.local``.

Ported from the predecessor's ``task_model/credentials.py``, with the lookup
pointed at this repo rather than a hardcoded ``/Users/blake/dev/rolodex``.

The environment wins over the file, so a run can be pointed at a different key
without editing anything. ``.env.local`` is gitignored and is the only place a
secret should sit; nothing here logs a value, and callers must not either --
a key in a log is a key in a bug report.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env.local"


def repo_root() -> Path:
    """The directory whose ``.env.local`` holds this project's credentials.

    Resolved from this file rather than the working directory, so a run started
    from a subdirectory finds the same file.
    """
    return Path(__file__).resolve().parent.parent.parent


def read_env_value(name: str, *, env_dir: Path | None = None) -> str:
    """Return one credential, from the environment or ``.env.local``.

    Missing is an error rather than an empty string: a request sent with no key
    fails at the far end with a message about authentication, several seconds
    and one confusing stack trace later.
    """
    value = os.environ.get(name)
    if value:
        return value

    env_path = (env_dir or repo_root()) / ENV_FILENAME
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, separator, raw_value = line.strip().partition("=")
            if separator and key == name:
                parsed = raw_value.strip().strip("\"'")
                if parsed:
                    return parsed

    raise RuntimeError(f"{name} must be set in the environment or {env_path}")
