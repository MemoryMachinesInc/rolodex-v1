"""Properties of the agentic regime that must hold before a run costs anything.

Nothing here calls a model. The sandbox and the prompt seam are exactly the
parts whose failures are invisible in the output: a profile built from outside
the corpus, or built under two rules that contradict each other, looks like any
other profile.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rolodex_v1 import agentic, grounded_profile, prompt_rules
from rolodex_v1.evidence_pack import build_pack
from rolodex_v1.resolved_entities import Alias, ResolvedEntity

# The pack now searches for an entity's resolved surface forms, so these tests
# carry a minimal resolution rather than a name string.
JOSH = ResolvedEntity(
    entity_id="proto:participant_person:20",
    canonical_name="Josh Earnest",
    case="participant_person",
    canonical_type="person",
    mention_count=1,
    memory_count=1,
    aliases=(Alias(text="Josh Earnest", probability=1.0),),
)

# The eleven fields the tuning runs score. The addendum owns them; the base
# rules must not, or the model has to arbitrate between a tuned rule and an
# untuned one.
SCORED_FIELDS = (
    "age",
    "emails",
    "phones",
    "social_profiles",
    "city",
    "country",
    "height",
    "pronouns",
    "gender",
    "birthday",
    "current_positions",
)

ADDENDUM = Path(__file__).resolve().parent.parent / "prompts/extraction_addendum.jinja"


# Every schema field is required, so even a profile that supports almost nothing
# has to state that. This is what a mostly-abstaining response looks like.
NULL_FIELDS = (
    "first_name",
    "last_name",
    "age",
    "height",
    "pronouns",
    "gender",
    "social_profiles",
    "city",
    "country",
    "type",
    "relationship_to_user",
    "how_met",
    "introduced_by",
    "birthday",
    "spouse",
    "communication_style",
)
EMPTY_FIELDS = (
    "aliases",
    "languages",
    "emails",
    "phones",
    "current_positions",
    "previous_positions",
    "expertise",
    "industries",
    "education",
    "common_connections",
    "interests",
    "publications",
)


def blank_profile(full_name: str) -> dict:
    """A schema-valid profile asserting support for nothing but the name."""
    profile: dict = dict.fromkeys(NULL_FIELDS)
    for field in EMPTY_FIELDS:
        profile[field] = []
    profile["full_name"] = full_name
    profile["character_context"] = {"classification": "side_character", "bullets": []}
    return profile


def gate(*roots: Path):
    return agentic.make_permission_gate(list(roots))


def ask(gate_fn: Any, tool: str, tool_input: dict) -> Any:
    return asyncio.run(gate_fn(tool, tool_input, None))


def test_the_gate_denies_every_tool_that_is_not_read_only(tmp_path: Path) -> None:
    """A batch run has nobody to answer a prompt, so refusal must be the default."""
    can_use = gate(tmp_path)
    for tool in ("Bash", "Write", "Edit", "WebFetch", "Task"):
        assert ask(can_use, tool, {"file_path": str(tmp_path / "x")}).behavior == "deny"


def test_the_gate_allows_a_read_inside_the_sandbox(tmp_path: Path) -> None:
    can_use = gate(tmp_path)
    result = ask(can_use, "Read", {"file_path": str(tmp_path / "pack.md")})
    assert result.behavior == "allow"


def test_the_gate_denies_a_path_outside_the_sandbox(tmp_path: Path) -> None:
    """A profile drawn from anywhere else is not evidence from this corpus."""
    can_use = gate(tmp_path / "inside")
    (tmp_path / "inside").mkdir()
    result = ask(can_use, "Read", {"file_path": str(tmp_path / "outside.txt")})
    assert result.behavior == "deny"


def test_the_gate_denies_an_escape_through_a_parent_reference(tmp_path: Path) -> None:
    """`../` resolves before the check, or the sandbox is decorative."""
    inside = tmp_path / "inside"
    inside.mkdir()
    can_use = gate(inside)
    result = ask(can_use, "Grep", {"path": str(inside / ".." / "elsewhere")})
    assert result.behavior == "deny"


def test_a_relative_path_resolves_against_the_pack_directory(tmp_path: Path) -> None:
    """The agent's cwd is the pack directory, so a bare filename is inside it."""
    can_use = gate(tmp_path)
    assert ask(can_use, "Read", {"file_path": "pack.md"}).behavior == "allow"
    assert ask(can_use, "Read", {"file_path": "../escape"}).behavior == "deny"


def test_the_base_rules_leave_the_scored_fields_to_the_addendum() -> None:
    """The two must not both legislate a scored field.

    The predecessor's rule 4 allowed an age computed from a birth date, which
    the addendum exists to forbid. Two rules that disagree make the model pick.
    """
    for field in SCORED_FIELDS:
        assert f"`{field}`" not in agentic.BASE_RULES, (
            f"base rules restate `{field}`, which the tuned addendum owns"
        )


def test_the_addendum_covers_every_scored_field() -> None:
    """If it stopped covering one, the base rules do not cover it either."""
    addendum = agentic.load_addendum(ADDENDUM).lower()
    for field in SCORED_FIELDS:
        assert field.replace("_", " ") in addendum, f"addendum does not cover {field}"


def test_the_addendum_is_appended_after_the_base_rules() -> None:
    """Precedence is positional; reversing it would make the tuning meaningless."""
    prompt = agentic.build_system_prompt("ADDENDUM BODY")
    assert prompt.index(agentic.BASE_RULES) < prompt.index("ADDENDUM BODY")


def test_suppressing_the_addendum_leaves_only_the_base_rules() -> None:
    assert agentic.build_system_prompt(None) == agentic.BASE_RULES


def test_a_missing_addendum_is_an_error_not_an_empty_string(tmp_path: Path) -> None:
    """Degrading silently would file an untuned artifact under a tuned name."""
    with pytest.raises(OSError):
        agentic.load_addendum(tmp_path / "absent.jinja")
    (tmp_path / "blank.jinja").write_text("   \n")
    with pytest.raises(ValueError, match="empty"):
        agentic.load_addendum(tmp_path / "blank.jinja")


def test_no_owner_means_the_relationship_field_is_left_null(tmp_path: Path) -> None:
    """It is defined against the owner; the handoff found it meaningless without one."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Josh Earnest is the Press Secretary.")
    pack = build_pack(corpus, JOSH, budget_chars=90_000, max_doc_chars=4000)

    unowned = agentic.build_user_prompt(pack, tmp_path / "p.md", corpus, None)
    assert "was not supplied" in unowned

    owned = agentic.build_user_prompt(pack, tmp_path / "p.md", corpus, "Barack Obama")
    assert "Barack Obama" in owned

    own_profile = agentic.build_user_prompt(
        pack, tmp_path / "p.md", corpus, "Josh Earnest"
    )
    assert "owner's own profile" in own_profile


def test_build_profile_attaches_resolved_grounding(tmp_path: Path) -> None:
    """The end of the contract: a cited line comes back as canonical source text."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Josh Earnest is the Press Secretary.\n")
    pack = build_pack(corpus, JOSH, budget_chars=90_000, max_doc_chars=4000)
    line = next(iter(pack.context.chunks_by_line))

    response = {
        "profile": blank_profile("Josh Earnest"),
        "grounding": [
            {"path": "/full_name", "spans": [{"start_line": line, "end_line": line}]}
        ],
    }

    profile = grounded_profile.build_profile(response, pack)
    resolved = profile["_grounding"]["/full_name"]["resolved_grounding_spans"][0]
    assert "Josh Earnest" in resolved["text_quote"]["exact"]
    assert resolved["offset_unit"] == "unicode_code_point"
    assert profile["_sources"]["doc"]["source"] == "doc.txt"


def test_a_profile_that_does_not_match_the_schema_is_rejected(tmp_path: Path) -> None:
    """A coerced value is a scoring difference nobody can see in the output."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Josh Earnest spoke.\n")
    pack = build_pack(corpus, JOSH, budget_chars=90_000, max_doc_chars=4000)
    with pytest.raises(ValidationError):
        grounded_profile.build_profile({"profile": {"full_name": "Josh"}}, pack)


def test_the_base_rules_are_byte_for_byte_what_they_were() -> None:
    """The system prompt is the experiment's constant; a reword is a new arm.

    This digest is of the base rules as they read before the sentences shared
    with the in-context regime moved into ``prompt_rules``, so it also pins that
    the move changed nothing the model receives.
    """
    digest = hashlib.sha256(agentic.BASE_RULES.encode("utf-8")).hexdigest()
    assert digest == "ca4bffc8a0e472d7ed648e2b1c7b35cca0c16869eddbff77fb2dbb2c84bb8121"


def test_the_owner_sentences_are_the_in_context_regimes_own(tmp_path: Path) -> None:
    """The regimes are scored against each other, so this field is asked for once.

    Layout differs on purpose -- this prompt is wrapped prose, the in-context one
    is a single line -- so the property is on the words, not the whitespace.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Josh Earnest is the Press Secretary.")
    pack = build_pack(corpus, JOSH, budget_chars=90_000, max_doc_chars=4000)

    for owner in (None, "Barack Obama", "Josh Earnest"):
        rendered = agentic.build_user_prompt(pack, tmp_path / "p.md", corpus, owner)
        shared = prompt_rules.owner_context(pack.name, owner)
        assert " ".join(shared.split()) in " ".join(rendered.split())
