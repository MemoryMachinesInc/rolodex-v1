"""What the scaffold guarantees, so a green run means something on day one.

Each test below states the property it guards. Replace them as real behaviour
arrives, but keep the shape: a test names a property the code must hold, not a
line of code it must contain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rolodex_v1.build_profiles import Config, main


def config(tmp_path: Path, **overrides: object) -> Config:
    """Build a Config whose corpus paths are inside tmp_path unless overridden."""
    kwargs: dict[str, object] = {
        "model_code": "g5m",
        "variant": "v1",
        "context": "memory",
        "memories_bundle": tmp_path / "memories_bundle.json",
        "source_docs_dir": tmp_path / "source_docs",
        "out_dir": tmp_path,
    }
    kwargs.update(overrides)
    return Config(**kwargs)


def test_config_encodes_settings_in_the_filename(tmp_path: Path) -> None:
    """Two configs must not collide on disk."""
    a = config(tmp_path, model_code="g4m")
    b = config(tmp_path, model_code="g5m")
    assert a.out_path != b.out_path
    assert "g4m" in a.out_path.name


def test_the_two_context_regimes_do_not_collide(tmp_path: Path) -> None:
    """Memory and chunk runs are different profiles, not two attempts at one."""
    memory = config(tmp_path, context="memory")
    chunks = config(tmp_path, context="chunks")
    assert memory.out_path != chunks.out_path


def test_an_unknown_context_is_rejected(tmp_path: Path) -> None:
    """A typo would otherwise name a file after a regime that did not run."""
    with pytest.raises(ValueError, match="context must be one of"):
        config(tmp_path, context="memories")


def test_each_context_reads_its_own_corpus(tmp_path: Path) -> None:
    """The configured location, not a constant, decides what is read."""
    assert config(tmp_path, context="memory").evidence_path.name == (
        "memories_bundle.json"
    )
    assert config(tmp_path, context="chunks").evidence_path.name == "source_docs"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    """A --dry-run that touched the output clobbered real data once."""
    assert main(["--out-dir", str(tmp_path), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_dry_run_survives_an_unconfigured_corpus(tmp_path: Path) -> None:
    """It is the cheap check on a machine that has not been given the data."""
    exit_code = main(
        [
            "--out-dir",
            str(tmp_path),
            "--memories-bundle",
            str(tmp_path / "absent.json"),
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_a_missing_corpus_fails_instead_of_writing(tmp_path: Path) -> None:
    """Half a profile set is worse than none, and looks the same on disk."""
    out_dir = tmp_path / "out"
    exit_code = main(
        [
            "--out-dir",
            str(out_dir),
            "--memories-bundle",
            str(tmp_path / "absent.json"),
        ]
    )
    assert exit_code == 1
    assert not out_dir.exists()


def test_existing_output_is_not_rewritten(tmp_path: Path) -> None:
    """Re-running is a no-op, which is what makes the stage incremental."""
    cfg = config(tmp_path)
    cfg.out_path.write_bytes(b"sentinel")
    assert main(["--out-dir", str(tmp_path)]) == 0
    assert cfg.out_path.read_bytes() == b"sentinel"


def test_each_context_names_its_own_env_var(tmp_path: Path) -> None:
    """A missing-corpus message that names the wrong knob sends you nowhere."""
    assert config(tmp_path, context="memory").evidence_env.endswith("MEMORIES_BUNDLE")
    assert config(tmp_path, context="chunks").evidence_env.endswith("SOURCE_DOCS")
