"""Properties of the bundle reader that the generation path depends on.

The fixtures below are trimmed from the real Obama bundle, keeping the shapes
that actually bite: a probability split across two claimants, a "Last, First
<email>" alias, and a merge holding several people.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rolodex_v1.resolved_entities import clean_surface, load_bundle


def entity(entity_id: str, name: str, aliases: list[tuple[str, float]], **kw: object):
    return {
        "resolved_entity_id": entity_id,
        "canonical_name": name,
        "canonical_type": kw.get("canonical_type"),
        "case": kw.get("case", "participant_person"),
        "mention_count": kw.get("mention_count", 1),
        "memory_count": kw.get("memory_count", 1),
        "aliases": [{"alias": a, "probability": p} for a, p in aliases],
    }


def write_bundle(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"entities": records, "count": len(records)}))
    return path


def test_entities_are_keyed_by_id_not_name(tmp_path: Path) -> None:
    """160 canonical names in the real bundle are shared; a name key loses them."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "organization:proto:entity_organization:2296",
                "The White House",
                [("White House", 1.0)],
                case="entity_organization",
            ),
            entity(
                "proto:participant_person:4267",
                "The White House",
                [("The White House", 1.0)],
            ),
        ],
    )
    entities = load_bundle(path)
    assert len(entities) == 2
    assert set(entities) == {
        "organization:proto:entity_organization:2296",
        "proto:participant_person:4267",
    }


def test_provenance_prefix_does_not_change_handling(tmp_path: Path) -> None:
    """Human-reviewed and machine-resolved entities take the same path."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "proto:participant_person:20", "Josh Earnest", [("Josh Earnest", 1.0)]
            ),
            entity(
                "post_splink_review:participant_person:0a2f",
                "Katherine J. Hogan",
                [("Katherine J. Hogan", 1.0)],
            ),
        ],
    )
    entities = load_bundle(path)
    assert all(e.is_person for e in entities.values())
    assert all(len(e.surface_forms()) == 1 for e in entities.values())


def test_a_contested_alias_resolves_to_one_owner(tmp_path: Path) -> None:
    """Probability is P(alias refers to this entity); claimants sum to 1.0."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "org:1",
                "The White House",
                [("The White House", 0.9983)],
                case="entity_organization",
            ),
            entity("person:1", "The White House", [("The White House", 0.0017)]),
        ],
    )
    entities = load_bundle(path)
    assert entities["org:1"].surface_forms(0.5) == ("The White House",)
    assert entities["person:1"].surface_forms(0.5) == ()


def test_an_entity_can_lose_every_form_to_the_threshold(tmp_path: Path) -> None:
    """133 real entities do. The caller must report them, not silently skip."""
    path = write_bundle(tmp_path, [entity("org:bp", "BP", [("BP", 0.0427)])])
    assert load_bundle(path)["org:bp"].surface_forms(0.5) == ()
    assert load_bundle(path)["org:bp"].surface_forms(0.0) == ("BP",)


def test_forms_are_minimized(tmp_path: Path) -> None:
    """A longer form matches nowhere the shorter one does not."""
    path = write_bundle(
        tmp_path,
        [entity("p:1", "Obama", [("Obama", 1.0), ("Barack Obama", 1.0)])],
    )
    assert load_bundle(path)["p:1"].surface_forms() == ("Obama",)


def test_emails_survive_as_their_own_matchers(tmp_path: Path) -> None:
    """Headers address people who are never named in the body."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "p:1",
                "Josh Earnest",
                [("Earnest, Joshua R. <Joshua_R_Earnest@who.eop.gov>", 1.0)],
            )
        ],
    )
    resolved = load_bundle(path)["p:1"]
    assert resolved.emails() == ("joshua_r_earnest@who.eop.gov",)
    assert resolved.surface_forms() == ("Joshua R. Earnest",)


def test_a_swept_up_merge_is_measurable(tmp_path: Path) -> None:
    """The real bundle holds one 100-alias entity spanning 38 surnames."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "p:clean",
                "Josh Earnest",
                [("Josh Earnest", 1.0), ("Joshua R. Earnest", 1.0)],
            ),
            entity(
                "p:merged",
                "Nancy-Ann DeParle",
                [("Amy Hall", 1.0), ("Andrea Palm", 1.0), ("Barbara Smith", 1.0)],
            ),
        ],
    )
    entities = load_bundle(path)
    assert entities["p:clean"].surname_spread() == 1
    assert entities["p:merged"].surname_spread() == 3


def test_case_decides_personhood_when_type_is_null(tmp_path: Path) -> None:
    """Half the real bundle carries a null canonical_type; every record has a case."""
    path = write_bundle(
        tmp_path, [entity("p:1", "Someone", [("Someone", 1.0)], canonical_type=None)]
    )
    resolved = load_bundle(path)["p:1"]
    assert resolved.canonical_type is None
    assert resolved.is_person


def test_a_duplicate_entity_id_is_refused(tmp_path: Path) -> None:
    """Silently keeping the last one would drop a profile."""
    path = write_bundle(
        tmp_path,
        [entity("p:1", "A", [("A", 1.0)]), entity("p:1", "B", [("B", 1.0)])],
    )
    with pytest.raises(ValueError, match="duplicate resolved_entity_id"):
        load_bundle(path)


def test_a_non_bundle_is_refused(tmp_path: Path) -> None:
    """A wrong --entities-bundle should fail here, not at extraction time."""
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"profiles": []}))
    with pytest.raises(ValueError, match="not a resolved-entities bundle"):
        load_bundle(path)


def test_clean_surface_reorders_and_strips() -> None:
    """Aliases arrive as 'Last, First <email>' and must come out as prose."""
    assert clean_surface("Earnest, Joshua R. <j@who.eop.gov>") == "Joshua R. Earnest"
    assert clean_surface("Mrs. Obama") == "Mrs. Obama"


def test_an_address_only_entity_keeps_its_email(tmp_path: Path) -> None:
    """56 real entities are named by an address; they have no prose form at all."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "p:1",
                "politicoplaybook@politico.com",
                [("politicoplaybook@politico.com", 1.0)],
            )
        ],
    )
    resolved = load_bundle(path)["p:1"]
    assert resolved.surface_forms() == ()
    assert resolved.emails() == ("politicoplaybook@politico.com",)


def test_a_truncation_artifact_does_not_eat_the_real_names(tmp_path: Path) -> None:
    r"""The minimization bug, pinned.

    The real bundle holds ``axelro`` beside ``Axelrod``. Testing plain substring
    containment drops every real form, because ``axelro`` is lexically inside
    ``david axelrod`` -- and the entity is left searching for a fragment that
    matches nothing. 216 entities lost all their usable forms that way. The
    boundary-aware test keeps both, because ``\baxelro\b`` does not match
    ``Axelrod``.
    """
    path = write_bundle(
        tmp_path,
        [
            entity(
                "p:1",
                "axelrod_d",
                [("axelro", 1.0), ("Axelrod", 1.0), ("David Axelrod", 1.0)],
            )
        ],
    )
    forms = load_bundle(path)["p:1"].surface_forms()
    assert "Axelrod" in forms
    assert "axelro" in forms


def test_minimization_still_drops_a_genuinely_redundant_form(tmp_path: Path) -> None:
    """A form a kept pattern already matches inside is real redundancy."""
    path = write_bundle(
        tmp_path,
        [entity("p:1", "Obama", [("Obama", 1.0), ("Barack Obama", 1.0)])],
    )
    assert load_bundle(path)["p:1"].surface_forms() == ("Obama",)


def test_every_surface_form_and_email_becomes_a_pattern(tmp_path: Path) -> None:
    """Both identify, and neither outranks the other; there is no surname tier."""
    path = write_bundle(
        tmp_path,
        [
            entity(
                "p:1",
                "Peggy Maloney",
                [("Maloney, Peggy <peggy_a_maloney@oa.eop.gov>", 1.0)],
            )
        ],
    )
    resolved = load_bundle(path)["p:1"]
    patterns = resolved.mention_patterns()
    assert len(patterns) == len(resolved.surface_forms()) + len(resolved.emails())
    assert any(p.search("From: peggy_a_maloney@oa.eop.gov") for p in patterns)
