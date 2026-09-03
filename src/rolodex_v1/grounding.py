"""Addressed source lines, and the resolution of cited lines back to source.

Ported from ``~/dev/rolodex/grounding.py``. The shape of the contract:

1. Evidence is rendered as *addressed lines* -- a global line counter over
   ≤``CHUNK_LENGTH``-character chunks, grouped under a per-source heading. Every
   chunk remembers the true character offsets it came from.
   :class:`LineAddressing` owns that counter, so "which line is line 47" is a
   question this module can answer on its own.
2. The model cites ``{start_line, end_line}`` ranges. Line numbers are the only
   thing it has to get right, which is the cheapest citation format that can
   still be checked.
3. Those ranges resolve deterministically into canonical selectors -- a source
   digest, code-point offsets, and a W3C-style text quote with 64 characters of
   context either side -- so a citation survives the document being moved,
   re-rendered, or modestly edited.

The offsets index Unicode code points in the *original* document, never the
numbered view. That distinction is the reason a resolved span can be checked
against a source revision at all.

Not ported: the memory-regime document renderer, and the two coverage
heuristics (exact-value search and sibling-path borrowing) that
``attach_profile_grounding`` used under ``require_complete=True``. Both patch up
a specific model's under-citation, both are untested against this repo's
regimes, and importing them would quietly redefine what "grounded" means here.
They are in the predecessor if a caller ever needs complete coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

# 200 characters is the wrap width the citation format assumes. Changing it
# renumbers every line, so a profile's spans only mean anything against the pack
# built with the same value.
CHUNK_LENGTH = 200
QUOTE_CONTEXT_LENGTH = 64


@dataclass(frozen=True)
class AddressedChunk:
    """One numbered line, and the exact source offsets it was taken from."""

    line_number: int
    source_id: str
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True)
class DocumentSource:
    """One source document, its digest, and the chunks addressed from it."""

    source_id: str
    source: str
    source_revision: str
    document: str
    chunks: tuple[AddressedChunk, ...]


@dataclass(frozen=True)
class GroundingContext:
    """The model-facing numbered text, and everything needed to resolve it."""

    addressed_text: str
    chunks_by_line: dict[int, AddressedChunk]
    sources: dict[str, DocumentSource]


def physical_line_ranges(document: str) -> list[tuple[int, int]]:
    """Offset ranges of each non-empty physical line, excluding its ending.

    Line endings are excluded so a resolved quote never begins or ends with a
    stray newline, and empty lines are dropped so they cannot consume a line
    number the model might then cite at nothing.
    """
    ranges: list[tuple[int, int]] = []
    offset = 0
    for physical_line in document.splitlines(keepends=True):
        end_offset = offset + len(physical_line.rstrip("\r\n"))
        if end_offset > offset:
            ranges.append((offset, end_offset))
        offset += len(physical_line)
    return ranges


def wrap_physical_line(
    document: str,
    start_offset: int,
    end_offset: int,
    max_length: int = CHUNK_LENGTH,
) -> list[tuple[int, int]]:
    """Split one long physical line into addressable chunks.

    Prefers the last whitespace inside the window so a chunk breaks between
    words; falls back to a hard cut when a single token is longer than the
    window. Either way the chunks tile the line exactly, with no gap and no
    overlap, which is what keeps offsets exact.
    """
    ranges: list[tuple[int, int]] = []
    cursor = start_offset
    while end_offset - cursor > max_length:
        candidate = document[cursor : cursor + max_length]
        whitespace = list(re.finditer(r"\s+", candidate))
        split_offset = whitespace[-1].end() if whitespace else max_length
        ranges.append((cursor, cursor + split_offset))
        cursor += split_offset
    if cursor < end_offset:
        ranges.append((cursor, end_offset))
    return ranges


def chunk_windows(
    document: str,
    windows: list[tuple[int, int]],
    source_id: str,
    max_length: int = CHUNK_LENGTH,
) -> list[AddressedChunk]:
    """Chunk the selected windows of a document, keeping true offsets.

    The chunks come back unnumbered (``line_number=0``): a caller selecting
    evidence may still discard them, and only :class:`LineAddressing` decides
    what a surviving chunk is called. Offsets are real from the start, which is
    what lets a cited line resolve back to canonical source text.
    """
    chunks: list[AddressedChunk] = []
    for line_start, line_end in physical_line_ranges(document):
        if not any(
            line_start < window_end and line_end > window_start
            for window_start, window_end in windows
        ):
            continue
        for chunk_start, chunk_end in wrap_physical_line(
            document, line_start, line_end, max_length
        ):
            chunks.append(
                AddressedChunk(
                    line_number=0,
                    source_id=source_id,
                    start_offset=chunk_start,
                    end_offset=chunk_end,
                    text=document[chunk_start:chunk_end],
                )
            )
    return chunks


def render_addressed_line(chunk: AddressedChunk) -> str:
    """The one rendering of a numbered line the model ever sees.

    Every producer of the model-facing text goes through here, so the string a
    caller prints and the chunk :func:`resolve_grounding_spans` looks up under
    that number cannot disagree about which line it is.
    """
    return f"{chunk.line_number}: {chunk.text}"


@dataclass(frozen=True)
class AddressedLines:
    """One document's chunks after numbering, and their rendered form."""

    chunks: tuple[AddressedChunk, ...]
    rendered: tuple[str, ...]


class LineAddressing:
    """The global line counter, and the only thing that assigns a line number.

    A caller hands over the chunks it selected and gets addressed text back; it
    never sees the counter. :meth:`address` numbers from the next free line but
    commits nothing, so a caller that then decides the text does not fit simply
    drops the result -- there is no tentative counter to rewind, and no way for
    a skipped document to leave a hole in the numbering.
    """

    def __init__(self) -> None:
        self.chunks_by_line: dict[int, AddressedChunk] = {}
        self._next_line = 1

    def address(self, chunks: list[AddressedChunk]) -> AddressedLines:
        numbered = tuple(
            replace(chunk, line_number=self._next_line + index)
            for index, chunk in enumerate(chunks)
        )
        return AddressedLines(
            chunks=numbered,
            rendered=tuple(render_addressed_line(chunk) for chunk in numbered),
        )

    def commit(self, addressed: AddressedLines) -> None:
        """Make those lines citable: after this they are what their numbers mean."""
        for chunk in addressed.chunks:
            self.chunks_by_line[chunk.line_number] = chunk
        self._next_line += len(addressed.chunks)


def _escape_pointer_segment(segment: object) -> str:
    return str(segment).replace("~", "~0").replace("/", "~1")


def iter_profile_leaf_paths(value: object, path: str = "") -> list[str]:
    """RFC 6901 pointers for every non-null primitive in a profile.

    This is the set of values that must each carry a citation. Keys beginning
    with ``_`` are skipped: the grounding sidecars are not themselves grounded.
    """
    paths: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).startswith("_"):
                continue
            paths.extend(
                iter_profile_leaf_paths(
                    nested, f"{path}/{_escape_pointer_segment(key)}"
                )
            )
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            paths.extend(iter_profile_leaf_paths(nested, f"{path}/{index}"))
    elif value is not None and path:
        paths.append(path)
    return paths


def _parse_span(span: object) -> tuple[int, int]:
    if not isinstance(span, dict):
        raise ValueError("Grounding spans must be JSON objects")
    start_line = span.get("start_line")
    end_line = span.get("end_line")
    # bool is an int in Python, and `True` as a line number would silently
    # resolve to line 1 rather than being rejected.
    if (
        isinstance(start_line, bool)
        or isinstance(end_line, bool)
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
        or start_line < 1
        or end_line < 1
    ):
        raise ValueError(f"Invalid grounding span: {span}")
    return start_line, end_line


def _normalize_spans(
    spans: object,
    context: GroundingContext,
) -> list[tuple[int, int]]:
    """Parse, repair, sort and merge the model's raw line ranges."""
    if not isinstance(spans, list) or not spans:
        raise ValueError("Every grounded value must cite at least one span")

    parsed: list[tuple[int, int]] = []
    for span in spans:
        start_line, end_line = _parse_span(span)
        if end_line < start_line:
            # A reversed range is a transposition, not a claim about different
            # evidence: keep whichever endpoint actually addresses a line.
            if start_line in context.chunks_by_line:
                end_line = start_line
            elif end_line in context.chunks_by_line:
                start_line = end_line
            else:
                raise ValueError(f"Invalid grounding span: {span}")
        parsed.append((start_line, end_line))

    parsed.sort()
    merged: list[tuple[int, int]] = []
    for start_line, end_line in parsed:
        if merged and start_line <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_line))
        else:
            merged.append((start_line, end_line))
    return merged


def _is_contiguous(document: str, previous_end: int, next_start: int) -> bool:
    """Whether two chunks abut in the source, ignoring the line ending.

    Adjacent addressed lines are always separated by the newline that
    ``physical_line_ranges`` trimmed, so exact adjacency is too strict. Anything
    more than whitespace between them is text the model was never shown.
    """
    return next_start >= previous_end and not document[previous_end:next_start].strip()


def resolve_grounding_spans(
    spans: object,
    context: GroundingContext,
) -> tuple[list[dict], list[dict]]:
    """Resolve cited line ranges into canonical source selectors.

    Returns the normalized line ranges alongside the resolved selectors.

    A cited range is split wherever the underlying text is not continuous --
    at a source-document boundary, and at a gap where intervening text was not
    addressed. Both matter for the same reason: a selector is a claim about
    what the evidence says, and slicing across a gap would quote text the model
    never saw and did not cite. An evidence pack built from selected windows is
    full of such gaps.
    """
    raw_spans: list[dict] = []
    resolved_spans: list[dict] = []

    for start_line, end_line in _normalize_spans(spans, context):
        try:
            cited = [
                context.chunks_by_line[line]
                for line in range(start_line, end_line + 1)
                if line in context.chunks_by_line
            ]
            if not cited:
                raise KeyError(start_line)
        except KeyError as exc:
            raise ValueError(
                f"Grounding span references unknown line {exc.args[0]}"
            ) from exc

        runs: list[list[AddressedChunk]] = []
        for chunk in cited:
            previous = runs[-1][-1] if runs else None
            if (
                previous is not None
                and previous.source_id == chunk.source_id
                and _is_contiguous(
                    context.sources[chunk.source_id].document,
                    previous.end_offset,
                    chunk.start_offset,
                )
            ):
                runs[-1].append(chunk)
            else:
                runs.append([chunk])

        for run in runs:
            source = context.sources[run[0].source_id]
            start_offset = run[0].start_offset
            end_offset = run[-1].end_offset
            raw_spans.append(
                {
                    "start_line": run[0].line_number,
                    "end_line": run[-1].line_number,
                }
            )
            resolved_spans.append(
                {
                    "source_id": source.source_id,
                    "source_revision": source.source_revision,
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "offset_unit": "unicode_code_point",
                    "text_quote": {
                        "exact": source.document[start_offset:end_offset],
                        "prefix": source.document[
                            max(0, start_offset - QUOTE_CONTEXT_LENGTH) : start_offset
                        ],
                        "suffix": source.document[
                            end_offset : end_offset + QUOTE_CONTEXT_LENGTH
                        ],
                    },
                }
            )

    return raw_spans, resolved_spans


def attach_profile_grounding(
    profile: dict,
    raw_grounding: dict[str, object],
    context: GroundingContext,
    *,
    require_complete: bool = False,
) -> dict:
    """Attach resolved citations to a profile as ``_grounding``/``_sources``.

    A citation on a parent pointer covers its descendants, so a model that cites
    ``/emails/0`` grounds ``/emails/0/address`` too.

    With ``require_complete=False`` an uncited or unresolvable value is left
    ungrounded and scored as such, which is what you want when measuring how
    well a model cites. With ``True`` the profile is rejected outright: half a
    grounded profile is not a partial success, it is an artifact whose coverage
    number is a lie.
    """
    if not all(isinstance(path, str) for path in raw_grounding):
        raise ValueError("Grounding keys must be JSON Pointer strings")

    expected_paths = set(iter_profile_leaf_paths(profile))
    spans_by_leaf: dict[str, object] = {}
    missing: set[str] = set()
    for path in expected_paths:
        candidate = path
        while True:
            if candidate in raw_grounding:
                spans_by_leaf[path] = raw_grounding[candidate]
                break
            if not candidate:
                missing.add(path)
                break
            candidate = candidate.rpartition("/")[0]

    resolved_grounding: dict[str, dict] = {}
    cited_source_ids: set[str] = set()
    for path in sorted(spans_by_leaf):
        try:
            raw_spans, resolved_spans = resolve_grounding_spans(
                spans_by_leaf[path], context
            )
        except ValueError:
            if require_complete:
                raise
            missing.add(path)
            continue
        cited_source_ids.update(span["source_id"] for span in resolved_spans)
        resolved_grounding[path] = {
            "grounding_spans": raw_spans,
            "resolved_grounding_spans": resolved_spans,
        }

    if require_complete and missing:
        raise ValueError(f"Grounding is missing profile paths: {sorted(missing)[:20]}")

    # Only cited sources are registered. The pack may span hundreds of
    # documents; carrying the ones nothing cites would bloat every profile with
    # text no value depends on.
    grounded = dict(profile)
    grounded["_grounding"] = resolved_grounding
    grounded["_sources"] = {
        source_id: {
            "source": context.sources[source_id].source,
            "source_revision": context.sources[source_id].source_revision,
            "document": context.sources[source_id].document,
            "lines": [
                {
                    "line_number": chunk.line_number,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "text": chunk.text,
                }
                for chunk in context.sources[source_id].chunks
            ],
        }
        for source_id in sorted(cited_source_ids)
    }
    return grounded
