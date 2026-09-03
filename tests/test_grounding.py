"""Properties of span resolution, which is the part that makes a profile worth having.

A resolved span is a claim that the source says this. Every test here guards a
way that claim could be false while still looking well-formed on disk.
"""

from __future__ import annotations

import itertools

import pytest

from rolodex_v1.grounding import (
    AddressedChunk,
    DocumentSource,
    GroundingContext,
    LineAddressing,
    attach_profile_grounding,
    chunk_windows,
    iter_profile_leaf_paths,
    physical_line_ranges,
    resolve_grounding_spans,
    wrap_physical_line,
)


def context_from(
    documents: dict[str, str], selected: dict[str, list[int]] | None = None
):
    """Address the given physical lines of each document, in order.

    ``selected`` picks line indices per document, which is how a real evidence
    pack behaves -- it shows windows, not whole files.
    """
    chunks_by_line: dict[int, AddressedChunk] = {}
    sources: dict[str, DocumentSource] = {}
    line_number = 1
    for source_id, document in documents.items():
        ranges = physical_line_ranges(document)
        wanted = selected.get(source_id) if selected else None
        chunks: list[AddressedChunk] = []
        for index, (start, end) in enumerate(ranges):
            if wanted is not None and index not in wanted:
                continue
            chunk = AddressedChunk(
                line_number=line_number,
                source_id=source_id,
                start_offset=start,
                end_offset=end,
                text=document[start:end],
            )
            chunks_by_line[line_number] = chunk
            chunks.append(chunk)
            line_number += 1
        sources[source_id] = DocumentSource(
            source_id=source_id,
            source=f"{source_id}.txt",
            source_revision="sha256:test",
            document=document,
            chunks=tuple(chunks),
        )
    return GroundingContext("", chunks_by_line, sources)


def test_a_cited_line_resolves_to_exactly_its_source_text() -> None:
    context = context_from({"doc": "alpha\nbravo\ncharlie\n"})
    _, resolved = resolve_grounding_spans([{"start_line": 2, "end_line": 2}], context)
    assert len(resolved) == 1
    quote = resolved[0]["text_quote"]
    assert quote["exact"] == "bravo"
    assert quote["prefix"] == "alpha\n"
    assert quote["suffix"] == "\ncharlie\n"


def test_a_range_over_adjacent_lines_is_one_quote() -> None:
    """Consecutive lines really are contiguous source, so they stay one selector."""
    context = context_from({"doc": "alpha\nbravo\ncharlie\n"})
    _, resolved = resolve_grounding_spans([{"start_line": 1, "end_line": 3}], context)
    assert len(resolved) == 1
    assert resolved[0]["text_quote"]["exact"] == "alpha\nbravo\ncharlie"


def test_a_range_across_unselected_text_is_split() -> None:
    """A pack shows windows, so consecutive line numbers can straddle a gap.

    Slicing across it would quote text the model was never shown and did not
    cite -- an invented quote inside an otherwise valid-looking citation.
    """
    document = "alpha\nSECRET UNSELECTED MIDDLE\ncharlie\n"
    context = context_from({"doc": document}, selected={"doc": [0, 2]})
    _, resolved = resolve_grounding_spans([{"start_line": 1, "end_line": 2}], context)
    assert [span["text_quote"]["exact"] for span in resolved] == ["alpha", "charlie"]
    assert all("SECRET" not in span["text_quote"]["exact"] for span in resolved)


def test_a_range_across_two_sources_is_split() -> None:
    """A selector names one source revision, so it cannot span two documents."""
    context = context_from({"one": "alpha\n", "two": "bravo\n"})
    _, resolved = resolve_grounding_spans([{"start_line": 1, "end_line": 2}], context)
    assert [span["source_id"] for span in resolved] == ["one", "two"]


def test_a_reversed_range_is_read_as_a_transposition() -> None:
    """It is a typo about one line, not a claim about different evidence."""
    context = context_from({"doc": "alpha\nbravo\n"})
    _, resolved = resolve_grounding_spans([{"start_line": 2, "end_line": 1}], context)
    assert [span["text_quote"]["exact"] for span in resolved] == ["bravo"]


def test_an_unknown_line_is_rejected() -> None:
    """An invented line number must not silently resolve to nothing."""
    context = context_from({"doc": "alpha\n"})
    with pytest.raises(ValueError, match="unknown line"):
        resolve_grounding_spans([{"start_line": 99, "end_line": 99}], context)


def test_a_boolean_is_not_a_line_number() -> None:
    """A bool is an int in Python, so True would otherwise resolve to line 1."""
    context = context_from({"doc": "alpha\n"})
    with pytest.raises(ValueError, match="Invalid grounding span"):
        resolve_grounding_spans([{"start_line": True, "end_line": True}], context)


def test_an_empty_citation_is_rejected() -> None:
    context = context_from({"doc": "alpha\n"})
    with pytest.raises(ValueError, match="at least one span"):
        resolve_grounding_spans([], context)


def test_leaf_paths_skip_the_grounding_sidecars() -> None:
    """Grounding is not itself grounded, or attaching it would never terminate."""
    profile = {"full_name": "A", "age": None, "emails": [{"address": "a@b.c"}]}
    assert set(iter_profile_leaf_paths(profile)) == {"/full_name", "/emails/0/address"}
    assert iter_profile_leaf_paths({"_grounding": {"x": 1}}) == []


def test_a_parent_citation_grounds_its_children() -> None:
    """Citing /emails/0 is a citation of the address inside it."""
    context = context_from({"doc": "a@b.c\n"})
    profile = {"emails": [{"address": "a@b.c"}]}
    grounded = attach_profile_grounding(
        profile, {"/emails/0": [{"start_line": 1, "end_line": 1}]}, context
    )
    assert "/emails/0/address" in grounded["_grounding"]


def test_uncited_values_are_left_ungrounded_by_default() -> None:
    """How completely a model cites is a measurement, not a reason to fail."""
    context = context_from({"doc": "alpha\n"})
    profile = {"full_name": "alpha", "city": "Chicago"}
    grounded = attach_profile_grounding(
        profile, {"/full_name": [{"start_line": 1, "end_line": 1}]}, context
    )
    assert set(grounded["_grounding"]) == {"/full_name"}


def test_require_complete_rejects_a_partial_profile() -> None:
    """Half a grounded profile is an artifact whose coverage number is a lie."""
    context = context_from({"doc": "alpha\n"})
    profile = {"full_name": "alpha", "city": "Chicago"}
    with pytest.raises(ValueError, match="missing profile paths"):
        attach_profile_grounding(
            profile,
            {"/full_name": [{"start_line": 1, "end_line": 1}]},
            context,
            require_complete=True,
        )


def test_only_cited_sources_are_registered() -> None:
    """A pack spans hundreds of documents; uncited ones would bloat every profile."""
    context = context_from({"cited": "alpha\n", "uncited": "bravo\n"})
    grounded = attach_profile_grounding(
        {"full_name": "alpha"},
        {"/full_name": [{"start_line": 1, "end_line": 1}]},
        context,
    )
    assert set(grounded["_sources"]) == {"cited"}


def test_wrapping_tiles_a_long_line_exactly() -> None:
    """Gaps or overlaps in the tiling would make every offset after them wrong."""
    document = " ".join(f"word{index:03d}" for index in range(200))
    ranges = wrap_physical_line(document, 0, len(document), 200)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(document)
    for (_, previous_end), (next_start, _) in itertools.pairwise(ranges):
        assert previous_end == next_start
    assert "".join(document[start:end] for start, end in ranges) == document


def test_a_hard_token_longer_than_the_window_still_wraps() -> None:
    """A URL with no whitespace must not loop forever looking for a break."""
    document = "x" * 500
    ranges = wrap_physical_line(document, 0, len(document), 200)
    assert [end - start for start, end in ranges] == [200, 200, 100]


def test_empty_lines_take_no_line_number() -> None:
    """A cited blank line would be a citation of nothing."""
    assert physical_line_ranges("alpha\n\nbravo\n") == [(0, 5), (7, 12)]


def test_two_touching_ranges_become_one_quote() -> None:
    """The ``+ 1`` in the merge rule: ranges that abut are one passage.

    A model that cites line 1 and line 2 separately is quoting one sentence
    twice, not two pieces of evidence. Without the ``+ 1`` they would stay two
    ranges and the profile would carry the passage as two selectors.
    """
    context = context_from({"doc": "alpha\nbravo\ncharlie\n"})
    _, resolved = resolve_grounding_spans(
        [{"start_line": 1, "end_line": 1}, {"start_line": 2, "end_line": 2}], context
    )
    assert [span["text_quote"]["exact"] for span in resolved] == ["alpha\nbravo"]


def test_ranges_with_a_line_between_them_stay_separate() -> None:
    """Abutting is as far as the rule reaches; a skipped line is not covered.

    Merging across the gap would put line 2 inside a quote nothing cited.
    """
    context = context_from({"doc": "alpha\nbravo\ncharlie\n"})
    _, resolved = resolve_grounding_spans(
        [{"start_line": 1, "end_line": 1}, {"start_line": 3, "end_line": 3}], context
    )
    assert [span["text_quote"]["exact"] for span in resolved] == ["alpha", "charlie"]


def test_a_partly_unknown_range_narrows_to_the_lines_that_exist() -> None:
    """Pinning today's behaviour, which is quieter than it looks.

    Unknown lines are filtered out before the rejection check, so a range that
    overruns the pack resolves to whatever part of it was real and says nothing
    about the rest. Only a range where *every* line is unknown is rejected.
    """
    context = context_from({"doc": "alpha\nbravo\n"})
    _, resolved = resolve_grounding_spans([{"start_line": 1, "end_line": 5}], context)
    assert [span["text_quote"]["exact"] for span in resolved] == ["alpha\nbravo"]


def test_addressing_numbers_densely_from_one_across_documents() -> None:
    """One counter, one owner: numbering does not restart per document."""
    addressing = LineAddressing()
    first = addressing.address(chunk_windows("alpha\nbravo\n", [(0, 12)], "one"))
    addressing.commit(first)
    second = addressing.address(chunk_windows("charlie\n", [(0, 8)], "two"))
    addressing.commit(second)
    assert sorted(addressing.chunks_by_line) == [1, 2, 3]
    assert addressing.chunks_by_line[3].source_id == "two"


def test_uncommitted_lines_leave_no_hole_in_the_numbering() -> None:
    """A caller that addresses text and then drops it must not spend numbers.

    This is what a document skipped by the pack budget does. A number the pack
    never prints is a number the model could cite at nothing.
    """
    addressing = LineAddressing()
    dropped = addressing.address(chunk_windows("alpha\nbravo\n", [(0, 12)], "one"))
    assert [chunk.line_number for chunk in dropped.chunks] == [1, 2]
    kept = addressing.address(chunk_windows("charlie\n", [(0, 8)], "two"))
    addressing.commit(kept)
    assert sorted(addressing.chunks_by_line) == [1]
    assert addressing.chunks_by_line[1].source_id == "two"


def test_the_rendered_line_is_the_line_the_resolver_will_find() -> None:
    """The numbering identity, asserted at its single owner.

    Two renderings of a line that disagree would resolve a citation against
    text the model was not looking at when it cited that number.
    """
    addressing = LineAddressing()
    addressed = addressing.address(
        chunk_windows("alpha\nbravo\ncharlie\n", [(0, 20)], "doc")
    )
    addressing.commit(addressed)
    for rendered in addressed.rendered:
        number, _, text = rendered.partition(": ")
        assert addressing.chunks_by_line[int(number)].text == text
