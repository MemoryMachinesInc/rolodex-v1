"""Properties the evidence pack must hold before anything is paid for.

The pack decides what a profile can possibly say. A name matched too loosely
puts another person's facts in front of the model; a pack that is not
reproducible makes two runs incomparable. Both failures are silent downstream.

Every pattern now comes from the entity's resolved surface forms, so these tests
build a ``ResolvedEntity`` rather than passing a name string. That is the point
of the change: nothing in this module derives what to search for.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from rolodex_v1.evidence_pack import (
    CUE_NEAR_NAME,
    CUE_WINDOW,
    DEFAULT_RECIPE,
    FIELD_CUES,
    NAME_WINDOW,
    build_pack,
    select_windows,
)
from rolodex_v1.grounding import CHUNK_LENGTH
from rolodex_v1.resolved_entities import Alias, ResolvedEntity, form_pattern


def person(name: str, *aliases: str) -> ResolvedEntity:
    """An entity whose surface forms are exactly the aliases given."""
    return ResolvedEntity(
        entity_id=f"proto:participant_person:{name}",
        canonical_name=name,
        case="participant_person",
        canonical_type="person",
        mention_count=1,
        memory_count=1,
        aliases=tuple(Alias(text=alias, probability=1.0) for alias in (name, *aliases)),
    )


def corpus(tmp_path: Path, documents: dict[str, str]) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir(exist_ok=True)
    for stem, text in documents.items():
        (directory / f"{stem}.txt").write_text(text, encoding="utf-8")
    return directory


def test_a_name_inside_a_longer_surname_is_not_a_mention() -> None:
    r"""A plain `\bMalley\b` matches after the apostrophe. This is trap 5."""
    pattern = form_pattern("Robert Malley")
    assert pattern.search("Robert Malley chaired the meeting")
    assert not pattern.search("Robert O'Malley chaired the meeting")
    assert not pattern.search("Robert O’Malley chaired the meeting")
    # The corpus carries mojibake where a curly apostrophe became a `?`.
    assert not pattern.search("Robert O?Malley chaired the meeting")


def test_a_possessive_still_counts_as_a_mention() -> None:
    """Guarding the leading edge must not cost the trailing one."""
    assert form_pattern("Vladimir Putin").search("Vladimir Putin's spokesman said")


def test_a_middle_initial_does_not_break_the_form() -> None:
    """The corpus writes both "Peggy Maloney" and "Peggy A. Maloney"."""
    pattern = form_pattern("Peggy Maloney")
    assert pattern.search("contact Peggy Maloney")
    assert pattern.search("contact Peggy A. Maloney")
    assert pattern.search("contact Peggy A Maloney")


def test_every_particle_of_a_form_is_kept() -> None:
    """First-plus-last would search for "Osama Laden", which never occurs."""
    pattern = form_pattern("Osama bin Laden")
    assert pattern.search("a tape from Osama bin Laden")
    assert not pattern.search("a tape from Osama Laden")


def test_a_bare_surname_is_not_a_mention() -> None:
    """The regression this whole change exists to prevent.

    Matching surnames reported the same 692 documents for Mike Allen and
    Jessica Allen. A document that only ever writes "Allen" now contributes to
    neither, because no surface form of either entity matches it.
    """
    document = "Allen confirmed the schedule at the briefing."
    entity = person("Mike Allen")
    assert select_windows(document, entity.mention_patterns()) == []


def test_an_email_form_identifies_a_document_that_only_writes_the_surname() -> None:
    """An address identifies as firmly as a full name.

    Press copy that only writes the surname often still carries the address, and
    the bundle keeps the address as its own alias.
    """
    document = "From: peggy_a_maloney@oa.eop.gov\nMaloney confirmed the booking."
    without = person("Peggy Maloney")
    assert select_windows(document, without.mention_patterns()) == []
    with_email = person("Peggy Maloney", "Maloney, Peggy <peggy_a_maloney@oa.eop.gov>")
    assert select_windows(document, with_email.mention_patterns()) != []


def test_addressed_lines_resolve_to_their_source_text(tmp_path: Path) -> None:
    """The whole contract: a numbered line is a real slice of a real document."""
    directory = corpus(
        tmp_path,
        {
            "doc-a": "SUBJECT: Briefing\nJosh Earnest is the Press Secretary.\n",
            "doc-b": "SUBJECT: Other\nJosh Earnest is at josh@who.eop.gov.\n",
        },
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=90_000, max_doc_chars=4000
    )
    assert pack.addressed_lines > 0
    for chunk in pack.context.chunks_by_line.values():
        document = pack.context.sources[chunk.source_id].document
        assert document[chunk.start_offset : chunk.end_offset] == chunk.text
        assert f"{chunk.line_number}: {chunk.text}" in pack.text


def test_the_pack_is_byte_identical_across_runs(tmp_path: Path) -> None:
    """Two runs of the same config must be comparable, so the pack cannot drift."""
    directory = corpus(
        tmp_path,
        {
            "b-doc": "Josh Earnest spoke. Josh Earnest spoke again.",
            "a-doc": "Josh Earnest spoke once.",
            "c-doc": "Nothing about anyone relevant.",
        },
    )
    entity = person("Josh Earnest")
    first = build_pack(directory, entity, budget_chars=90_000, max_doc_chars=4000)
    second = build_pack(directory, entity, budget_chars=90_000, max_doc_chars=4000)
    assert first.text == second.text


def test_the_densest_documents_are_kept_when_the_budget_bites(tmp_path: Path) -> None:
    """A truncated pack should hold the best evidence, not the alphabetically first."""
    directory = corpus(
        tmp_path,
        {
            "z-dense": "Josh Earnest. " * 40,
            "a-sparse": "Josh Earnest was mentioned once.",
        },
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=200, max_doc_chars=200
    )
    assert "z-dense" in pack.context.sources


def test_one_document_cannot_crowd_out_the_corpus(tmp_path: Path) -> None:
    """Breadth across sources beats depth in any one of them."""
    directory = corpus(
        tmp_path,
        {
            "huge": "Josh Earnest spoke.\n" * 500,
            "small": "Josh Earnest is the Press Secretary.",
        },
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=90_000, max_doc_chars=500
    )
    assert set(pack.context.sources) == {"huge", "small"}


def test_an_entity_nobody_mentions_yields_an_empty_pack(tmp_path: Path) -> None:
    """Empty means nothing citable, which is the signal not to spend on it."""
    directory = corpus(tmp_path, {"doc": "Josh Earnest spoke."})
    pack = build_pack(
        directory, person("Katherine Murtha"), budget_chars=90_000, max_doc_chars=4000
    )
    assert pack.is_empty
    assert pack.documents_matched == 0


def test_the_budget_measures_the_rendered_pack(tmp_path: Path) -> None:
    """Counting excerpts instead of the rendered text overran by ~7%.

    It is the same distinction that inflated a ~700K-token excerpt set into a
    1,168,995-token request, which is trap 1 in miniature.
    """
    directory = corpus(
        tmp_path,
        {
            f"doc{index:02d}": "Josh Earnest spoke at length.\n" * 20
            for index in range(30)
        },
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=5_000, max_doc_chars=1_000
    )
    sections = len(pack.text) - pack.text.index("## Source")
    assert sections <= 5_000


def test_line_numbers_stay_dense_when_a_document_is_skipped(tmp_path: Path) -> None:
    """A number the pack never prints is a number the model could cite at nothing."""
    directory = corpus(
        tmp_path,
        {f"doc{index:02d}": "Josh Earnest spoke.\n" * 30 for index in range(20)},
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=4_000, max_doc_chars=800
    )
    numbers = sorted(pack.context.chunks_by_line)
    assert numbers == list(range(1, len(numbers) + 1))


def test_a_form_with_no_usable_tokens_is_rejected() -> None:
    """Silently matching everything would be worse than failing."""
    with pytest.raises(ValueError, match="no usable token"):
        form_pattern("A. B.")


def test_an_entity_with_nothing_to_search_for_is_refused(tmp_path: Path) -> None:
    """build_pack must not scan a corpus for an entity it cannot identify."""
    directory = corpus(tmp_path, {"doc": "text"})
    empty = ResolvedEntity(
        entity_id="proto:participant_person:0",
        canonical_name="",
        case="participant_person",
        canonical_type=None,
        mention_count=0,
        memory_count=0,
        aliases=(),
    )
    with pytest.raises(ValueError, match="nothing to search for"):
        build_pack(directory, empty, budget_chars=1_000, max_doc_chars=100)


def test_a_cue_beyond_the_name_window_still_opens_one() -> None:
    """The cue pass is why a fact 500 characters from the name survives.

    ``NAME_WINDOW`` keeps 400 characters either side; ``CUE_NEAR_NAME`` lets a
    scored-field cue up to 600 away bring its own window. Without that pass the
    phone number below is simply not in the pack, and no profile built from it
    could ever carry the field.
    """
    filler = "padding text about nothing in particular. " * 12
    document = f"Josh Earnest chaired it.\n{filler[:480]}\nreach him at 202-456-1111.\n"
    windows = select_windows(document, person("Josh Earnest").mention_patterns())
    assert "202-456-1111" in "".join(document[start:end] for start, end in windows)


def test_a_cue_further_away_than_the_proximity_rule_is_ignored() -> None:
    """Otherwise one mention would drag in cues from the whole document."""
    filler = "padding text about nothing in particular. " * 40
    document = (
        f"Josh Earnest chaired it.\n{filler[:1200]}\nreach him at 202-456-1111.\n"
    )
    windows = select_windows(document, person("Josh Earnest").mention_patterns())
    assert "202-456-1111" not in "".join(document[start:end] for start, end in windows)


def test_cues_alone_select_nothing() -> None:
    """A cue is evidence about whoever is named nearby, and nobody is named here.

    The cue pass reads windows off ``hits``, so a document full of addresses and
    titles that never names the entity contributes no evidence at all.
    """
    document = "Dr Someone Else was appointed. Reach them at 202-456-1111.\n"
    assert select_windows(document, person("Josh Earnest").mention_patterns()) == []


def test_a_recipe_that_forbids_distant_cues_narrows_the_pack() -> None:
    """The recipe is wired, not decorative: changing it changes the evidence."""
    filler = "padding text about nothing in particular. " * 12
    document = f"Josh Earnest chaired it.\n{filler[:480]}\nreach him at 202-456-1111.\n"
    patterns = person("Josh Earnest").mention_patterns()
    narrow = replace(DEFAULT_RECIPE, cue_near_name=0)
    assert select_windows(document, patterns, narrow) != select_windows(
        document, patterns
    )


def test_the_pack_records_the_recipe_it_was_built_from(tmp_path: Path) -> None:
    """A knob that changes the evidence and is recorded nowhere is unfalsifiable.

    Tune a window and today's pack is a different artifact under an identical
    name. The recipe rides along so a profile can be checked against how its
    evidence was chosen.
    """
    directory = corpus(tmp_path, {"doc": "Josh Earnest is the Press Secretary."})
    pack = build_pack(
        directory,
        person("Josh Earnest"),
        budget_chars=90_000,
        max_doc_chars=4_000,
        min_probability=0.5,
    )
    assert pack.recipe.name_window == NAME_WINDOW
    assert pack.recipe.cue_window == CUE_WINDOW
    assert pack.recipe.cue_near_name == CUE_NEAR_NAME
    assert pack.recipe.chunk_length == CHUNK_LENGTH
    assert pack.recipe.budget_chars == 90_000
    assert pack.recipe.max_doc_chars == 4_000
    assert pack.recipe.min_alias_probability == 0.5


def test_the_recipe_digest_tracks_the_cue_regex() -> None:
    """The cue set is prose, so only a digest of it can be compared."""
    expected = hashlib.sha256(FIELD_CUES.pattern.encode("utf-8")).hexdigest()
    assert DEFAULT_RECIPE.field_cues_digest == f"sha256:{expected}"


def test_every_printed_line_number_addresses_the_line_it_prints(tmp_path: Path) -> None:
    """The numbering identity, end to end over a pack the budget truncated.

    The model reads ``N: text`` and cites ``N``; the resolver looks ``N`` up. If
    those two ever named different lines, every citation in the run would
    resolve to text the model was not looking at -- and the profile would still
    validate.
    """
    directory = corpus(
        tmp_path,
        {
            f"doc{index:02d}": f"Josh Earnest spoke {index}.\n" * 30
            for index in range(20)
        },
    )
    pack = build_pack(
        directory, person("Josh Earnest"), budget_chars=4_000, max_doc_chars=800
    )
    printed = {
        int(line.split(":", 1)[0]): line.split(": ", 1)[1]
        for line in pack.text.splitlines()
        if line[:1].isdigit()
    }
    assert printed
    assert set(printed) == set(pack.context.chunks_by_line)
    for number, text in printed.items():
        chunk = pack.context.chunks_by_line[number]
        assert chunk.text == text
        source = pack.context.sources[chunk.source_id]
        assert source.document[chunk.start_offset : chunk.end_offset] == text
