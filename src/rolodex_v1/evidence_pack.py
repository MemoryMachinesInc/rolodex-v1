r"""Turn a source-document corpus into one addressed evidence pack per entity.

This is stage 1 of generation, deliberately separated from the agent that
consumes it: the pack is pure, deterministic and free, so it can be inspected
and tested without spending anything. ``--dry-run`` stops here.

What the pack is: verbatim passages from every document that mentions the
entity, numbered with ``grounding``'s addressing rules so each line carries the
true character offsets of its source. The model cites line numbers; those
citations resolve back into canonical source selectors. The pack is not a
summary, and nothing in it is paraphrased -- a summary could not be cited.

**Mentions are found only by the entity's resolved surface forms.** There is no
name parsing here and no bare-surname pass. The predecessor derived a full name
from a roster key and then also matched the surname alone, which reported the
same 692 documents for Mike Allen and Jessica Allen and built Jessica's profile
from Mike's traffic. Every pattern this module matches comes from
:meth:`rolodex_v1.resolved_entities.ResolvedEntity.mention_patterns`, and a
string claimed by two entities has already had its probability split between
them, so ownership is settled before retrieval starts.

**This module chooses evidence; it does not number it.** Wrapping, line numbers
and the rendered form of a numbered line belong to
:class:`rolodex_v1.grounding.LineAddressing`, which is also what resolves a
citation -- so the line the pack printed and the line the resolver finds cannot
drift apart. What decided the choosing is recorded on :class:`PackRecipe` and
carried on the pack, because a window size that changes the evidence and is
written down nowhere makes two identically-named packs incomparable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

from rolodex_v1.grounding import (
    CHUNK_LENGTH,
    AddressedChunk,
    DocumentSource,
    GroundingContext,
    LineAddressing,
    chunk_windows,
)
from rolodex_v1.resolved_entities import (
    DEFAULT_MIN_PROBABILITY,
    EMAIL_RE,
    ResolvedEntity,
)

# Corpus documents are plain text whose first line is a `SUBJECT:` banner.
SUBJECT_LINE = re.compile(r"^SUBJECT:\s*(.+)$", re.MULTILINE)

# Cues worth showing when they sit near a mention. Narrowed to the fields a
# profile is actually scored on -- a cue that opens a window for an unscored
# field spends pack budget that a scored one needed.
FIELD_CUES = re.compile(
    r"(?:" + EMAIL_RE.pattern + r"|(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}"
    r"|(?:linkedin\.com/in/|twitter\.com/|x\.com/)[\w-]+"
    r"|\b(?:he|him|his|she|her|hers|they|them|their)\b"
    r"|\b(?:Mr|Mrs|Ms|Dr|Secretary|Judge|Governor|Senator|Representative"
    r"|Director|Chief|President|Chairman|Chairwoman|Administrator|Ambassador"
    r"|Press Secretary|Deputy|Assistant|Counsel|Spokesman|Spokeswoman)\b"
    r"|\b(?:is|was|serves as|named|appointed|nominated|sworn in|stepped down"
    r"|resigned|former|current)\b"
    r"|\b(?:born|birthday|age \d{1,3}|\d{1,3}[- ]year[- ]old|years old)\b"
    r"|\b(?:based in|lives in|resident of|of Washington|D\.C\.)\b"
    r"|\b(?:university|college|law school|graduated|degree|B\.A\.|J\.D\.|M\.B\.A)\b"
    r")",
    re.IGNORECASE,
)

NAME_WINDOW = (400, 400)  # characters kept either side of a mention
CUE_WINDOW = (200, 260)  # characters kept either side of a nearby cue
CUE_NEAR_NAME = 600  # how close a cue must sit to a mention to count

# The cue regex is prose, not a number, so the recipe records a digest of it.
# Two packs whose digests differ were built from different evidence however
# identical the rest of the recipe reads.
FIELD_CUES_DIGEST = (
    "sha256:" + hashlib.sha256(FIELD_CUES.pattern.encode("utf-8")).hexdigest()
)


@dataclass(frozen=True)
class PackRecipe:
    """Every knob that decides which evidence a pack is built from.

    ``min_alias_probability`` is in the output filename because it changes the
    evidence; so does each of these, and none of them reaches the filename. The
    recipe is instead written onto every checkpoint and out to the profile
    record, so a profile can be checked against how its evidence was chosen
    rather than against the assumption that the constants never moved -- but two
    runs at different recipes still collide on one output path, and the second
    overwrites the first unless it is pointed somewhere else.

    The defaults are today's constants, so a pack built without naming a recipe
    is the pack this repo has always built.
    """

    name_window: tuple[int, int] = NAME_WINDOW
    cue_window: tuple[int, int] = CUE_WINDOW
    cue_near_name: int = CUE_NEAR_NAME
    chunk_length: int = CHUNK_LENGTH
    field_cues_digest: str = FIELD_CUES_DIGEST
    # Filled in per run: both arrive as arguments rather than constants, but
    # they change the evidence exactly as much as the constants above do.
    budget_chars: int = 0
    max_doc_chars: int = 0
    min_alias_probability: float = DEFAULT_MIN_PROBABILITY


DEFAULT_RECIPE = PackRecipe()


@dataclass(frozen=True)
class Candidate:
    """One document that names the entity, waiting on the pack budget.

    ``mentions`` is the sort key: densest first, so a pack truncated by the
    budget keeps the best evidence.
    """

    mentions: int
    path: Path
    document: str
    chunks: list[AddressedChunk]


@dataclass(frozen=True)
class EvidencePack:
    """One entity's pack: the model-facing text and how to resolve its lines."""

    name: str
    text: str
    context: GroundingContext
    documents_scanned: int
    documents_matched: int
    documents_included: int
    recipe: PackRecipe = DEFAULT_RECIPE

    @property
    def addressed_lines(self) -> int:
        return len(self.context.chunks_by_line)

    @property
    def is_empty(self) -> bool:
        """No addressed lines means nothing citable, so nothing worth paying for."""
        return not self.context.chunks_by_line


def select_windows(
    document: str,
    patterns: tuple[re.Pattern[str], ...],
    recipe: PackRecipe = DEFAULT_RECIPE,
) -> list[tuple[int, int]]:
    """Character ranges worth showing, or empty if no form names the entity.

    Every pattern here identifies. There is no weaker second tier that only
    counts once something stronger has fired, because that tier was the bare
    surname and it is gone: which entity a string belongs to was decided by the
    bundle's probabilities, not by what else the document happens to say.
    """
    hits: list[tuple[int, int]] = []
    for pattern in patterns:
        hits.extend(match.span() for match in pattern.finditer(document))
    if not hits:
        return []

    before, after = recipe.name_window
    windows = [
        (max(0, start - before), min(len(document), end + after)) for start, end in hits
    ]
    cue_before, cue_after = recipe.cue_window
    for match in FIELD_CUES.finditer(document):
        start, end = match.span()
        if any(abs(start - hit_start) <= recipe.cue_near_name for hit_start, _ in hits):
            windows.append(
                (max(0, start - cue_before), min(len(document), end + cue_after))
            )

    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_pack(
    corpus: Path,
    entity: ResolvedEntity,
    *,
    budget_chars: int,
    max_doc_chars: int,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    recipe: PackRecipe = DEFAULT_RECIPE,
) -> EvidencePack:
    """Scan every ``.txt`` in ``corpus`` for ``entity`` and build its pack.

    The entity is passed whole rather than as a name, because what to search for
    is a property of the resolution and nothing here should be re-deriving it
    from a string.

    The budget is in characters, not tokens, because the pack is written to disk
    and read by the agent as a file rather than rendered into one request whose
    size has to be predicted. That makes this regime immune to the token-budget
    trap that bit the single-request regimes -- but it also means the pack is
    only one part of what fills the agent's context, so leave room.

    It is measured against the *rendered* section, headings and line-number
    prefixes included, which is the same distinction that inflated a ~700K-token
    excerpt set into a 1,168,995-token request. Counting excerpts here overran
    a 90,000-character budget by about 7%.

    ``recipe`` carries the rest of what decided this evidence -- window sizes,
    the cue proximity, the wrap width, the cue regex digest -- and comes back on
    the pack, so a profile is checkable against how it was built rather than
    against the assumption that no constant has moved since.
    """
    recipe = replace(
        recipe,
        budget_chars=budget_chars,
        max_doc_chars=max_doc_chars,
        min_alias_probability=min_probability,
    )
    name = entity.canonical_name
    patterns = entity.mention_patterns(min_probability)
    if not patterns:
        raise ValueError(f"{entity.entity_id} has nothing to search for")
    paths = sorted(corpus.glob("*.txt"))

    candidates: list[Candidate] = []
    for path in paths:
        document = path.read_text(errors="replace")
        windows = select_windows(document, patterns, recipe)
        if not windows:
            continue
        chunks = chunk_windows(document, windows, path.stem, recipe.chunk_length)
        if chunks:
            mentions = sum(len(pattern.findall(document)) for pattern in patterns)
            candidates.append(Candidate(mentions, path, document, chunks))

    # The tiebreak on filename is what makes the pack byte-identical run to run.
    candidates.sort(key=lambda candidate: (-candidate.mentions, candidate.path.name))

    addressing = LineAddressing()
    sources: dict[str, DocumentSource] = {}
    sections: list[str] = []
    used_chars = 0

    for candidate in candidates:
        path = candidate.path
        document = candidate.document
        # Cap each document so one dense briefing cannot crowd out the rest of
        # the corpus: breadth across sources beats depth in any one of them.
        # Unlike the pack budget this counts excerpt text only, so a section
        # renders 15-25% larger. It is a shape control, not an exact ceiling.
        capped: list[AddressedChunk] = []
        doc_chars = 0
        for chunk in candidate.chunks:
            if capped and doc_chars + len(chunk.text) > recipe.max_doc_chars:
                break
            capped.append(chunk)
            doc_chars += len(chunk.text)

        subject = SUBJECT_LINE.search(document)
        lines = [
            f"## Source {path.stem}",
            f"<!-- file: {path.name} | mentions: {candidate.mentions}"
            + (f" | subject: {subject.group(1).strip()[:160]}" if subject else "")
            + " -->",
        ]
        # How a line is numbered, and how a numbered line reads, are
        # `LineAddressing`'s to know. This loop only decides which chunks are
        # offered and whether the rendered result fits.
        addressed = addressing.address(capped)
        lines.extend(addressed.rendered)

        section = "\n".join(lines)
        if used_chars and used_chars + len(section) > budget_chars:
            # Skip rather than stop: a later, smaller document may still fit.
            # Nothing was committed, so the numbers this section was offered go
            # to the next document and the pack stays densely numbered from 1.
            continue
        addressing.commit(addressed)

        sources[path.stem] = DocumentSource(
            source_id=path.stem,
            source=path.name,
            source_revision="sha256:"
            + hashlib.sha256(document.encode("utf-8")).hexdigest(),
            document=document,
            chunks=addressed.chunks,
        )
        sections.append(section)
        used_chars += len(section)

    header = "\n".join(
        [
            f"# Evidence pack for: {name}",
            "",
            f"- corpus documents scanned: {len(paths)}",
            f"- documents naming them: {len(candidates)}",
            f"- documents included below: {len(sources)}",
            "- Passages are verbatim corpus text. Non-consecutive line numbers",
            "  inside a source mean intervening text was not selected; Read the",
            "  file for it.",
            "- Cite these line numbers, and only these, in `grounding`.",
            "",
        ]
    )
    text = header + "\n\n".join(sections)
    return EvidencePack(
        name=name,
        text=text,
        context=GroundingContext(
            addressed_text=text,
            chunks_by_line=addressing.chunks_by_line,
            sources=sources,
        ),
        documents_scanned=len(paths),
        documents_matched=len(candidates),
        documents_included=len(sources),
        recipe=recipe,
    )
