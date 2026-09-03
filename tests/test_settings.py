"""What the path config guarantees.

The reason this module exists is that four locations all hang off one `data/`
directory, so moving the corpus should be one setting rather than four. The
tests below are mostly about precedence: a config file has to beat a default and
lose to a flag, or it is a trap rather than a convenience.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rolodex_v1 import settings


def write_config(directory: Path, body: str) -> Path:
    """Write a config at the canonical ``configs/`` location under `directory`."""
    path = directory / settings.CONFIG_DIR / settings.CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def write_bare_config(directory: Path, body: str) -> Path:
    """Write a config at the pre-``configs/`` location, beside the repo root."""
    path = directory / settings.CONFIG_FILENAME
    path.write_text(body, encoding="utf-8")
    return path


def test_no_config_leaves_today_s_defaults_alone(tmp_path: Path) -> None:
    """A machine with no config file must behave exactly as it did before."""
    paths = settings.resolve(environ={}, start=tmp_path)
    assert paths.config_path is None
    assert paths.out_dir == Path("data")
    assert paths.source_docs == Path("data/source_docs")


def test_the_data_root_moves_every_location_at_once(tmp_path: Path) -> None:
    """The whole point: relocating the corpus is one setting, not four."""
    write_config(tmp_path, '[paths]\ndata = "/corpora/obama"\n')
    paths = settings.resolve(environ={}, start=tmp_path)
    assert paths.root == Path("/corpora/obama")
    assert paths.source_docs == Path("/corpora/obama/source_docs")
    assert paths.entities_bundle == Path("/corpora/obama/resolved_entities_bundle.json")
    # The output directory is the root itself, not a subdirectory of it.
    assert paths.out_dir == Path("/corpora/obama")


def test_a_relative_root_resolves_against_the_config_not_the_cwd(
    tmp_path: Path,
) -> None:
    """Otherwise a run from a subdirectory reads a different corpus."""
    nested = tmp_path / "deep" / "deeper"
    nested.mkdir(parents=True)
    write_config(tmp_path, '[paths]\ndata = "corpus"\n')
    paths = settings.resolve(environ={}, start=nested)
    # Relative to the config file, which lives in configs/ -- not to tmp_path
    # and not to the working directory the run started from.
    assert paths.source_docs == tmp_path / "configs/corpus/source_docs"


def test_the_shipped_idiom_reaches_the_repo_root_data_directory(
    tmp_path: Path,
) -> None:
    """`../data` is what the example ships, so it has to mean what it says.

    Moving the config into `configs/` changed what every relative path in it
    resolves to. This pins the one the example tells people to write.
    """
    write_config(tmp_path, '[paths]\ndata = "../data"\n')
    paths = settings.resolve(environ={}, start=tmp_path)
    assert paths.source_docs == tmp_path / "data/source_docs"


def test_a_flag_beats_the_config_file(tmp_path: Path) -> None:
    """A typed flag is the most specific thing a person can say."""
    write_config(tmp_path, '[paths]\ndata = "/corpora/obama"\n')
    paths = settings.resolve(
        flags={"source_docs": Path("/typed/on/the/command/line")},
        environ={},
        start=tmp_path,
    )
    assert paths.source_docs == Path("/typed/on/the/command/line")
    # ... and only that location; the rest still come from the file.


def test_the_environment_beats_the_config_file(tmp_path: Path) -> None:
    """An already-configured machine keeps working after this file lands."""
    write_config(tmp_path, '[paths]\ndata = "/corpora/obama"\n')
    paths = settings.resolve(
        environ={settings.ENV_SOURCE_DOCS: "/from/env"},
        start=tmp_path,
    )
    assert paths.source_docs == Path("/from/env")


def test_one_location_can_sit_outside_the_root(tmp_path: Path) -> None:
    """Corpora do not always arrive on the same disk."""
    write_config(
        tmp_path,
        '[paths]\ndata = "/corpora/obama"\nsource_docs = "/elsewhere/docs"\n',
    )
    paths = settings.resolve(environ={}, start=tmp_path)
    assert paths.source_docs == Path("/elsewhere/docs")


def test_discovery_walks_up_from_the_working_directory(tmp_path: Path) -> None:
    """One file in configs/ has to serve runs started anywhere inside."""
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    expected = write_config(tmp_path, "[paths]\n")
    assert settings.find_config(nested) == expected


def test_a_config_beside_the_root_is_still_found(tmp_path: Path) -> None:
    """A checkout predating configs/ must not fall through to `data/`.

    Silently reading a different corpus and reporting success is the worst
    failure this module has; supporting the old name costs one stat per level.
    """
    expected = write_bare_config(tmp_path, "[paths]\n")
    assert settings.find_config(tmp_path) == expected


def test_configs_wins_over_the_bare_name_at_the_same_level(tmp_path: Path) -> None:
    """Two files, one answer -- and the canonical location is the one that wins."""
    write_bare_config(tmp_path, '[paths]\ndata = "/legacy"\n')
    expected = write_config(tmp_path, '[paths]\ndata = "/canonical"\n')
    assert settings.find_config(tmp_path) == expected
    assert settings.resolve(environ={}, start=tmp_path).root == Path("/canonical")


def test_the_nearest_checkout_wins_over_an_outer_one(tmp_path: Path) -> None:
    """A nested checkout's own config outranks one further up, either name."""
    inner = tmp_path / "inner"
    inner.mkdir()
    write_config(tmp_path, '[paths]\ndata = "/outer"\n')
    write_bare_config(inner, '[paths]\ndata = "/inner"\n')
    assert settings.resolve(environ={}, start=inner).root == Path("/inner")


def test_a_misspelled_key_is_an_error_not_a_no_op(tmp_path: Path) -> None:
    """Silently ignoring it would run the default corpus and report success."""
    config_path = write_config(tmp_path, '[paths]\nsource_doc = "/typo"\n')
    with pytest.raises(ValueError, match="unknown \\[paths\\] key"):
        settings.read_config(config_path)


def test_an_unknown_table_is_an_error(tmp_path: Path) -> None:
    """`[path]` or `[tool.rolodex]` must not look like it worked."""
    config_path = write_config(tmp_path, '[path]\ndata = "/x"\n')
    with pytest.raises(ValueError, match="unknown table"):
        settings.read_config(config_path)


def test_a_named_config_that_is_missing_is_an_error(tmp_path: Path) -> None:
    """Falling back to discovery would read a corpus you did not name."""
    with pytest.raises(ValueError, match="does not exist"):
        settings.select_config(tmp_path / "absent.toml", environ={})


def test_an_empty_config_env_var_disables_discovery(tmp_path: Path) -> None:
    """This is the switch that keeps the test suite off a real corpus."""
    write_config(tmp_path, '[paths]\ndata = "/corpora/obama"\n')
    assert settings.select_config(None, {settings.ENV_CONFIG: ""}, tmp_path) is None


def test_every_configurable_key_reaches_a_location(tmp_path: Path) -> None:
    """A key the parser accepts but the resolver ignores is worse than a typo."""
    body = "\n".join(
        [
            "[paths]",
            *(f'{key} = "/set/{key}"' for key in settings.LAYOUT),
        ]
    )
    write_config(tmp_path, body + "\n")
    paths = settings.resolve(environ={}, start=tmp_path)
    for key in settings.LAYOUT:
        assert getattr(paths, key) == Path(f"/set/{key}")


def test_the_origin_says_which_file_was_in_effect(tmp_path: Path) -> None:
    """A path arriving from an unmentioned file needs a one-line explanation."""
    # The config goes in a sibling rather than an ancestor of the bare
    # directory, because discovery walks *up*: a config written at tmp_path
    # would be found from anywhere beneath it, including "bare".
    configured = tmp_path / "configured"
    bare = tmp_path / "bare"
    configured.mkdir()
    bare.mkdir()

    config_path = write_config(configured, '[paths]\ndata = "/corpora/obama"\n')
    assert str(config_path) in settings.resolve(environ={}, start=configured).origin
    assert (
        "no configs/rolodex-v1.toml" in settings.resolve(environ={}, start=bare).origin
    )


def test_every_key_the_example_offers_is_actually_accepted(tmp_path: Path) -> None:
    """The example shipped a `memories_bundle` key that read_config now rejects.

    An example that documents a key the parser refuses is worse than no example:
    uncommenting it produces an error, not the corpus you asked for.
    """
    import re

    text = Path("configs/rolodex-v1.toml.example").read_text(encoding="utf-8")
    # Table headers are uncommented too, or every key below a commented-out
    # `# [fetch]` would be read as a [paths] key and the example would look
    # wrong for a reason it does not have.
    live = re.sub(r"^# (\w+\s*=|\[\w+\])", r"\1", text, flags=re.MULTILINE)
    config = tmp_path / "rolodex-v1.toml"
    config.write_text(live, encoding="utf-8")
    settings.read_config(config)  # raises on an unknown [paths] key
    settings.read_fetch(config)  # and on an unknown [fetch] key
