"""Where the data lives, resolved from flags, environment and a config file.

Three locations are configurable -- the resolved-entities bundle, the source
documents and the output directory -- and all three default under a single
``data/`` directory. So the thing actually worth configuring is not each path
but the root they hang off: moving the corpus to an external disk is one
setting, not three.

AGENTS.md's rule is that a third corpus location graduates to a config file
rather than a third environment variable. This is that file. The existing
``ROLODEX_V1_*`` variables keep working -- a machine already set up does not
have to change -- but no new ones should be added; a new location belongs in
``LAYOUT`` below, where it costs nothing.

There is deliberately no memories location. Generation reads source documents
only; the memory regime was dropped rather than left configurable-but-unread,
because a path that nothing opens is a promise the code does not keep.

Precedence for each location, highest first:

1. the command-line flag
2. that location's ``ROLODEX_V1_*`` environment variable
3. that location's own key in ``[paths]``
4. ``[paths].data`` joined with the built-in name for that location
5. ``data/`` joined with the built-in name -- today's behaviour, unchanged on a
   machine with no config file

The file is ``configs/rolodex-v1.toml``, found by walking up from the working
directory; the bare ``rolodex-v1.toml`` beside it is still read, so a checkout
that predates the ``configs/`` directory is not silently ignored.

A relative path *in the config file* resolves against the config file's own
directory, not the working directory. That is what lets a run from a
subdirectory read the same corpus as a run from the repo root. It also means
**moving the config file changes what its relative paths mean** -- a `data`
directory beside the repo root is ``"../data"`` from inside ``configs/``. The
built-in default stays working-directory-relative because nothing has claimed a
root yet.

Only paths are configurable here. The knobs that change what a profile *is* --
model, context, variant, alias threshold -- stay on the command line, because
they are encoded in the output filename and a config file that quietly changed
one would break the promise that the name describes the artifact. ``[fetch]``,
the one other table, says how the two inputs are *filled* rather than what a
profile is -- see ``FetchSettings``.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rolodex_v1.memorymachines import ENVIRONMENTS

CONFIG_FILENAME = "rolodex-v1.toml"
CONFIG_DIR = "configs"

# Where to look at each level, in order. `configs/` is where these live; the
# bare name beside it stays supported because a config that is present but not
# found is the worst outcome this module has -- discovery would fall through to
# `data/` and report success against a corpus nobody chose. One extra stat per
# level buys that away.
CONFIG_RELATIVE_PATHS = (
    Path(CONFIG_DIR) / CONFIG_FILENAME,
    Path(CONFIG_FILENAME),
)

# Selects a config file explicitly. Set it to the empty string to disable
# discovery outright -- which is how the test suite stays hermetic on a machine
# that has a real config file in the checkout.
ENV_CONFIG = "ROLODEX_V1_CONFIG"

ENV_ENTITIES_BUNDLE = "ROLODEX_V1_ENTITIES_BUNDLE"
ENV_SOURCE_DOCS = "ROLODEX_V1_SOURCE_DOCS"

DEFAULT_DATA_ROOT = Path("data")

# What each location is called under the data root. A location listed here is
# settable by name, relocatable by root, and reported by --dry-run; adding one
# needs no new flag and no new environment variable.
LAYOUT = {
    "entities_bundle": "resolved_entities_bundle.json",
    "source_docs": "source_docs",
    # The output directory is the root itself, not a subdirectory of it.
    "out_dir": "",
}

# The environment variable that configures each location. Kept so an
# already-configured machine keeps working.
ENV_FOR = {
    "entities_bundle": ENV_ENTITIES_BUNDLE,
    "source_docs": ENV_SOURCE_DOCS,
}

CONFIGURABLE = frozenset({"data", *LAYOUT})

# What [fetch] may set. Deliberately no secret: a key or a refresh token in a
# config file is a key in whatever copies that file, and this one is only
# gitignored by convention. The file names the *variable* the credential lives
# in, and `credentials` reads it from the environment or .env.local.
#
# Nor any knob that changes what the bundle *is* -- a `case` filter, a smaller
# `top_k`, aliases off. The bundle lands under one fixed name, and a subset under
# that name is a different artifact nothing downstream can tell apart; see
# `memorymachines.fetch_bundle`.
FETCH_CONFIGURABLE = frozenset(
    {"environment", "base_url", "sources", "api_key_env", "refresh_token_env"}
)

#: Where each credential is read from unless [fetch] names another variable.
DEFAULT_API_KEY_ENV = "MM_API_KEY"
# The *name* of a variable, not a token.
DEFAULT_REFRESH_TOKEN_ENV = "MM_REFRESH_TOKEN"  # noqa: S105


@dataclass(frozen=True)
class FetchSettings:
    """How this machine fetches a corpus.

    There is no instance for a machine that cannot: ``fetch_table`` returns
    ``None`` for a config with no ``[fetch]`` table, and the fetch CLI says so
    rather than inventing a default environment and asking a production API for
    somebody's documents. ``environment`` has no default for the same reason --
    it is the one value that decides whose API is asked, so it is the one the
    user has to have written down.
    """

    environment: str
    base_url: str | None = None
    sources: tuple[str, ...] | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV
    refresh_token_env: str = DEFAULT_REFRESH_TOKEN_ENV


@dataclass(frozen=True)
class DataPaths:
    """Every configurable location, resolved, plus where that came from.

    ``fetch`` rides along because it comes out of the same parse: how the two
    inputs are filled is read from the file that says where they are, once.
    """

    entities_bundle: Path
    source_docs: Path
    out_dir: Path
    root: Path
    config_path: Path | None
    fetch: FetchSettings | None = None

    @property
    def origin(self) -> str:
        """What a dry run names when somebody asks why it reads there.

        A config file makes paths arrive from somewhere the command line does
        not show, so the report has to say which file was in effect -- or that
        none was, which is the more confusing case to debug without being told.
        """
        if self.config_path is None:
            return f"no {CONFIG_DIR}/{CONFIG_FILENAME} found; data root {self.root}"
        return f"{self.config_path}; data root {self.root}"


def find_config(start: Path | None = None) -> Path | None:
    """Return the nearest ``configs/rolodex-v1.toml`` at or above ``start``.

    Walking up from the working directory rather than from this module's
    location keeps the config with the *checkout* rather than the installed
    package, so an editable install and a wheel behave the same.

    Each level is checked completely before moving up, so a nested checkout's
    own config wins over an outer one regardless of which of the two names it
    uses. Nearest wins; between the two names at one level, ``configs/`` does.
    """
    directory = (start or Path.cwd()).resolve()
    for candidate in (directory, *directory.parents):
        for relative in CONFIG_RELATIVE_PATHS:
            config_path = candidate / relative
            if config_path.is_file():
                return config_path
    return None


def select_config(
    explicit: Path | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> Path | None:
    """Decide which config file is in effect, if any.

    A named file that does not exist is an error rather than a silent
    fall-through to discovery: you asked for that file, and reading a different
    corpus than the one you named is the expensive kind of surprise.
    """
    environ = os.environ if environ is None else environ
    if explicit is not None:
        path = explicit.expanduser()
        if not path.is_file():
            raise ValueError(f"config file {path} does not exist")
        return path

    from_env = environ.get(ENV_CONFIG)
    if from_env is not None:
        if not from_env.strip():
            return None
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise ValueError(f"${ENV_CONFIG} points at {path}, which does not exist")
        return path

    return find_config(start)


def load_config(path: Path) -> dict[str, Any]:
    """Parse the file once and reject tables nobody reads.

    The two table readers take the parsed document rather than the path, so a
    command that needs both ``[paths]`` and ``[fetch]`` opens the file once and
    cannot end up reading them from two different files.
    """
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    unknown_tables = sorted(set(document) - {"paths", "fetch"})
    if unknown_tables:
        raise ValueError(
            f"{path}: unknown table(s) {unknown_tables}; expected [paths] or [fetch]"
        )
    return document


def read_config(path: Path) -> dict[str, str]:
    """Parse and validate the ``[paths]`` table of the file at ``path``."""
    return paths_table(load_config(path), path)


def read_fetch(path: Path) -> FetchSettings | None:
    """Parse and validate the ``[fetch]`` table of the file at ``path``."""
    return fetch_table(load_config(path), path)


def paths_table(document: Mapping[str, Any], path: Path) -> dict[str, str]:
    """Validate the ``[paths]`` table.

    Unknown keys are rejected. A misspelled key would otherwise do nothing at
    all, and the run would read the default corpus while the file on disk looks
    like it says otherwise -- the failure would surface as a wrong number, not
    as an error. ``path`` is only for the messages.
    """
    paths = document.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"{path}: [paths] must be a table")

    unknown = sorted(set(paths) - CONFIGURABLE)
    if unknown:
        raise ValueError(
            f"{path}: unknown [paths] key(s) {unknown}; "
            f"expected any of {sorted(CONFIGURABLE)}"
        )
    for key, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}: [paths].{key} must be a non-empty string")
    return paths


def fetch_table(document: Mapping[str, Any], path: Path) -> FetchSettings | None:
    """Validate the ``[fetch]`` table, or return ``None`` when there is none.

    Unknown keys are rejected for the reason unknown ``[paths]`` keys are: a
    misspelled ``sources`` would quietly fetch all thirty source types instead
    of the two that were meant, and the failure would surface as a surprising
    bill of documents rather than as an error. ``path`` is only for the messages.
    """
    table = document.get("fetch")
    if table is None:
        return None
    if not isinstance(table, dict):
        raise ValueError(f"{path}: [fetch] must be a table")

    unknown = sorted(set(table) - FETCH_CONFIGURABLE)
    if unknown:
        raise ValueError(
            f"{path}: unknown [fetch] key(s) {unknown}; "
            f"expected any of {sorted(FETCH_CONFIGURABLE)}"
        )

    names = ", ".join(ENVIRONMENTS)
    if "environment" not in table:
        raise ValueError(
            f"{path}: [fetch].environment is required (one of {names}). It decides "
            "whose API is asked, so it is not defaulted."
        )
    environment = table["environment"]
    if environment not in ENVIRONMENTS:
        raise ValueError(
            f"{path}: [fetch].environment must be one of {names}, got {environment!r}"
        )
    base_url = table.get("base_url")
    if base_url is not None and (not isinstance(base_url, str) or not base_url.strip()):
        raise ValueError(f"{path}: [fetch].base_url must be a non-empty string")
    sources = table.get("sources")
    if sources is not None:
        if not isinstance(sources, list) or not all(
            isinstance(item, str) and item.strip() for item in sources
        ):
            raise ValueError(f"{path}: [fetch].sources must be a list of strings")
        sources = tuple(sources)

    return FetchSettings(
        environment=environment,
        base_url=base_url,
        sources=sources,
        api_key_env=table.get("api_key_env", DEFAULT_API_KEY_ENV),
        refresh_token_env=table.get("refresh_token_env", DEFAULT_REFRESH_TOKEN_ENV),
    )


def _config_relative(raw: str, base: Path) -> Path:
    """Resolve one config value against the config file's directory.

    The join is normalized because the config lives one level down, so the
    idiomatic ``data = "../data"`` would otherwise surface everywhere as
    ``configs/../data/source_docs`` -- in --dry-run, whose only job is to say
    where a run will read, and in every error message naming a missing input.

    Collapsing ``..`` lexically is safe here rather than in general: ``base`` is
    a directory that was found on disk, not a path assembled from user text, so
    there is no symlink between it and the ``..`` being cancelled.
    """
    path = Path(raw).expanduser()
    joined = path if path.is_absolute() else base / path
    return Path(os.path.normpath(joined))


def resolve(
    *,
    config: Path | None = None,
    flags: Mapping[str, Path | None] | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> DataPaths:
    """Apply the precedence above and return every location.

    ``flags`` carries ``None`` for a location the caller did not pass, which is
    why the argument parser defaults those to ``None`` rather than to a path: an
    argparse default is indistinguishable from a value the user typed, and the
    config file has to lose to a typed flag while beating an untyped one.
    """
    environ = os.environ if environ is None else environ
    flags = flags or {}

    config_path = select_config(config, environ, start)
    if config_path is None:
        keys: dict[str, str] = {}
        fetch = None
    else:
        document = load_config(config_path)
        keys = paths_table(document, config_path)
        fetch = fetch_table(document, config_path)
    base = config_path.parent if config_path is not None else Path()

    def configured(key: str) -> Path | None:
        raw = keys.get(key)
        return None if raw is None else _config_relative(raw, base)

    root = configured("data") or DEFAULT_DATA_ROOT

    def location(key: str) -> Path:
        flag = flags.get(key)
        if flag is not None:
            return flag
        from_env = environ.get(ENV_FOR[key], "") if key in ENV_FOR else ""
        if from_env.strip():
            return Path(from_env).expanduser()
        return configured(key) or root / LAYOUT[key]

    return DataPaths(
        entities_bundle=location("entities_bundle"),
        source_docs=location("source_docs"),
        out_dir=location("out_dir"),
        root=root,
        config_path=config_path,
        fetch=fetch,
    )
