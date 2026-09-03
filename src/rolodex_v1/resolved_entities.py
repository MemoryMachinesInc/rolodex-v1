"""Read a resolved-entities bundle: who to profile, and how they are named.

This replaces the three-file chain the predecessor used to recover a person's
surface forms -- ``annotations.json`` (label to entity id),
``merges_out.cleaned.json`` (which ids are one human) and a separate summary
bundle (id to canonical name). That chain was half human-curated, so it only
covered entities somebody had annotated, and it silently degraded to the bare
label when a lookup missed. One bundle carries all of it, for every entity, so
nothing here asks whether a resolution was made by a person or by a machine.

Provenance does leak into the data -- ``resolved_entity_id`` is prefixed
``proto:``, ``organization:proto:`` or ``post_splink_review:``, the last being
the human-reviewed pass. Read that prefix as an opaque part of the id. Branching
on it is what this module exists to avoid, and the sample bundle shows why it
would not help anyway: the worst merge in the corpus is a reviewed one.

Two properties of the schema drive the shape below.

**The id is the key, not the name.** 160 canonical names in the Obama bundle are
used by more than one entity (two distinct ``The White House`` entities, and so
on), so a name-keyed dict silently drops profiles.

**Alias probability is a distribution, not a confidence.** Where an alias string
is claimed by several entities, the claims sum to exactly 1.0 -- it is
``P(this alias refers to this entity)``. All 185 shared aliases in the sample
sum to 1.0, and every unshared alias is 1.0. So a threshold above 0.5 keeps at
most one owner per string, which is what stops one entity's chunks being
retrieved under another's name.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# An alias claimed by several entities splits its probability across them, so
# anything above 0.5 admits at most one owner per string.
DEFAULT_MIN_PROBABILITY = 0.5

# Aliases embed contact addresses ("Josh Earnest <j@who.eop.gov>"). They are
# unambiguous and appear in document headers that never spell the name out, so
# they are kept as matchers in their own right.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")

# Rejects a match that begins inside a longer name. A plain `\b` matches after
# an apostrophe -- and the corpus carries mojibake where a curly apostrophe was
# mangled into `?`, so `O?Malley` has to be guarded too. The trailing boundary
# stays a plain `\b` so possessives ("Putin's") still count as mentions.
_LEADING_GUARD = r"(?<![\w'\u2019?-])"

# Permits middle initials between name tokens: "Peggy Maloney" also matches
# "Peggy A. Maloney". A single letter cannot swallow a real word.
_TOKEN_GAP = r"\s+(?:[A-Za-z]\.?\s+)*"  # noqa: S105 -- a regex, not a secret

PERSON_CASE = "participant_person"
ORGANIZATION_CASE = "entity_organization"


def form_pattern(form: str) -> re.Pattern[str]:
    """Compile one surface form into the pattern that finds it in a document.

    Tokens shorter than two characters are dropped and the rest joined on a gap
    that tolerates a middle initial, so a form recovered from a resolution still
    matches the prose it was resolved from.
    """
    tokens = [token.rstrip(".") for token in form.split()]
    core = [re.escape(token) for token in tokens if len(token) > 1]
    if not core:
        raise ValueError(f"surface form has no usable token: {form!r}")
    return re.compile(_LEADING_GUARD + _TOKEN_GAP.join(core) + r"\b", re.IGNORECASE)


@dataclass(frozen=True)
class Alias:
    """One surface form and the probability that it refers to this entity."""

    text: str
    probability: float


@dataclass(frozen=True)
class ResolvedEntity:
    """One resolved entity, keyed by id because canonical names repeat."""

    entity_id: str
    canonical_name: str
    case: str
    canonical_type: str | None
    mention_count: int
    memory_count: int
    aliases: tuple[Alias, ...]

    @property
    def is_person(self) -> bool:
        """Whether this entity is a person.

        Read from ``case``, never ``canonical_type``: half the entities in the
        sample bundle carry a null type, and every one of them still has a case.
        """
        return self.case == PERSON_CASE

    def surface_forms(
        self,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
    ) -> tuple[str, ...]:
        r"""Every name this entity is written as, cleaned and minimized.

        A form is dropped when a shorter kept form already *matches inside it*
        under the same word-boundary rule retrieval uses. That qualifier is the
        whole correctness of this step. Testing plain substring containment
        instead -- which is what the predecessor did and what this module did
        first -- lets a truncation artifact eat every real name: the bundle
        holds ``axelro`` alongside ``David Axelrod``, ``axelro`` is
        lexically inside ``david axelrod``, and the entity was left searching
        for a fragment that matches nothing. That destroyed the usable forms of
        216 entities. ``\baxelro\b`` does not match ``Axelrod``, so the
        boundary-aware test keeps both.
        """
        forms: set[str] = set()
        for alias in self.aliases:
            if alias.probability < min_probability:
                continue
            cleaned = clean_surface(alias.text)
            if cleaned:
                forms.add(cleaned)

        minimal: list[str] = []
        for form in sorted(forms, key=lambda value: (len(value), value)):
            try:
                kept_patterns = [form_pattern(kept) for kept in minimal]
            except ValueError:  # pragma: no cover -- kept forms are valid
                kept_patterns = []
            if not any(pattern.search(form) for pattern in kept_patterns):
                minimal.append(form)
        return tuple(sorted(minimal))

    def emails(
        self,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
    ) -> tuple[str, ...]:
        """Contact addresses embedded in this entity's aliases, lowercased."""
        found = {
            match.lower()
            for alias in self.aliases
            if alias.probability >= min_probability
            for match in EMAIL_RE.findall(alias.text)
        }
        return tuple(sorted(found))

    def mention_patterns(
        self,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
    ) -> tuple[re.Pattern[str], ...]:
        """Every pattern that identifies this entity in a document.

        Surface forms and emails both identify, and neither outranks the other.
        There is deliberately no bare-surname pattern: a surname is what the
        predecessor matched on, and it reported the same 692 documents for Mike
        Allen and Jessica Allen. Disambiguation is the bundle's job now -- a
        string claimed by two entities splits its probability between them, so
        the threshold decides who owns it before retrieval ever runs.
        """
        patterns = []
        for form in self.surface_forms(min_probability):
            try:
                patterns.append(form_pattern(form))
            except ValueError:
                continue
        patterns.extend(
            re.compile(_LEADING_GUARD + re.escape(email) + r"\b", re.IGNORECASE)
            for email in self.emails(min_probability)
        )
        return tuple(patterns)

    def surname_spread(
        self,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
    ) -> int:
        """How many distinct final name tokens this entity's forms carry.

        A real person's surface forms share a surname. A merge that swept up
        several people does not, and the sample bundle contains one: a
        100-alias entity holding Amy Hall, Andrea Palm, Lauren Aronson and
        Barbara Smith under one name, every alias at probability 1.0. No
        threshold catches that, so it is measured instead.
        """
        surnames = {
            form.rsplit(" ", 1)[-1].casefold()
            for form in self.surface_forms(min_probability)
            if form
        }
        return len(surnames)


def clean_surface(text: str) -> str:
    """Reduce one alias to the name as it appears in prose.

    Drops any ``<email>``, reorders ``"Last, First"``, and strips accents.
    Punctuation is kept: resolved forms carry it (``Mrs. Obama``, ``Dr. Malley``)
    and the matcher is expected to tolerate it rather than the reader to
    discard it.
    """
    text = EMAIL_RE.sub(" ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    if "," in text:
        last, _, first = text.partition(",")
        text = f"{first} {last}"
    return re.sub(r"\s+", " ", text).strip(" \t\n\"'<>")


def _entity_from_record(record: dict[str, Any]) -> ResolvedEntity:
    """Build one entity, rejecting a record missing a field we key or match on."""
    for field in ("resolved_entity_id", "canonical_name", "case", "aliases"):
        if field not in record:
            raise ValueError(f"entity record is missing {field!r}")
    return ResolvedEntity(
        entity_id=record["resolved_entity_id"],
        canonical_name=record["canonical_name"],
        case=record["case"],
        canonical_type=record.get("canonical_type"),
        mention_count=record.get("mention_count", 0),
        memory_count=record.get("memory_count", 0),
        aliases=tuple(
            Alias(text=alias["alias"], probability=float(alias["probability"]))
            for alias in record["aliases"]
        ),
    )


def load_bundle(path: Path) -> dict[str, ResolvedEntity]:
    """Load every entity in a resolved-entities bundle, keyed by entity id.

    Nothing is filtered here -- not by case, not by type, and not by the
    provenance prefix on the id. Selection is the caller's decision, so a future
    bundle whose entities were resolved some other way needs no change here.
    """
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "entities" not in payload:
        raise ValueError(f"{path} is not a resolved-entities bundle")

    entities: dict[str, ResolvedEntity] = {}
    for record in payload["entities"]:
        entity = _entity_from_record(record)
        if entity.entity_id in entities:
            raise ValueError(f"duplicate resolved_entity_id: {entity.entity_id}")
        entities[entity.entity_id] = entity
    return entities
