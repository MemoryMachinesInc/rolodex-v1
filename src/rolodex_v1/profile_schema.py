"""The profile contract: what a model is asked to return, and how it is checked.

Ported from ``~/dev/rolodex/profile_schema.py``. Two things about it are
load-bearing and easy to undo by accident:

``extra="forbid"`` and ``strict=True``
    A profile is scored field by field against human labels. A coerced value --
    the string ``"42"`` arriving where an ``int`` was asked for, or an extra key
    riding along unnoticed -- is a scoring difference nobody can see in the
    output. Rejecting both at the boundary is cheaper than explaining a number
    later. It also means ``model_json_schema()`` emits
    ``additionalProperties: false`` throughout, which is what makes the schema
    usable as a strict structured-output contract.

Every field is required
    Not because every field is known, but so that "unsupported" is *stated*
    rather than left out. ``null`` for a scalar or nested object and ``[]`` for
    a collection is a claim the model makes and abstention scoring can read; a
    missing key is indistinguishable from the model forgetting.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Language(StrictModel):
    language: str
    proficiency: Literal["native", "fluent", "conversational", "basic"] | None


class Email(StrictModel):
    address: str


class Phone(StrictModel):
    number: str


class SocialProfiles(StrictModel):
    linkedin: str | None
    website: str | None


class CurrentPosition(StrictModel):
    title: str | None
    company: str | None
    department: str | None


class PreviousPosition(StrictModel):
    title: str | None
    company: str | None


class Education(StrictModel):
    institution: str
    degree: str | None
    field: str | None


class CharacterContext(StrictModel):
    classification: Literal["core_character", "side_character"]
    bullets: list[str]


class Spouse(StrictModel):
    name: str


class Publication(StrictModel):
    title: str
    venue: str | None
    year: int | None


# CAREFUL: pydantic emits a class docstring as the JSON Schema's `description`,
# so everything between the quotes below is prompt text that every regime's
# model reads. It is not a note for maintainers -- those go in comments like
# this one, where they cannot reach the model.
#
# This is the tuned runs' wording verbatim but for one clause: theirs said
# "requested from OpenAI", which is false for the agentic regime and carries no
# instruction either way. The second paragraph is the load-bearing part, and it
# is the reason every field is required.
class Biography(StrictModel):
    """Exact biography object requested from the model.

    Every key is required. Unsupported scalar and nested-object values are
    represented by ``null``; unsupported collections are represented by ``[]``.
    """

    full_name: str
    first_name: str | None
    last_name: str | None
    age: int | None
    height: str | None
    pronouns: str | None
    gender: str | None
    aliases: list[str]
    languages: list[Language]
    emails: list[Email]
    phones: list[Phone]
    social_profiles: SocialProfiles | None
    city: str | None
    country: str | None
    current_positions: list[CurrentPosition]
    previous_positions: list[PreviousPosition]
    expertise: list[str]
    industries: list[str]
    education: list[Education]
    type: (
        Literal[
            "colleague",
            "boss",
            "mentor",
            "client",
            "friend",
            "acquaintance",
            "other",
        ]
        | None
    )
    relationship_to_user: str | None
    character_context: CharacterContext
    how_met: str | None
    introduced_by: str | None
    common_connections: list[str]
    birthday: str | None
    spouse: Spouse | None
    interests: list[str]
    publications: list[Publication]
    communication_style: Literal["formal", "casual", "slow responder"] | None


class GroundingSpan(StrictModel):
    start_line: int
    end_line: int


class GroundingItem(StrictModel):
    path: str
    spans: list[GroundingSpan]


# A profile plus one citation per emitted primitive value. Grounding is a
# sibling array rather than a field on each value because the profile shape is
# fixed above and cannot carry per-leaf metadata; the two are stitched together
# by JSON Pointer in ``grounding.attach_profile_grounding``.
#
# Deliberately undocumented in the docstring sense: the tuned runs sent no
# `description` for this object, and adding one would put text in front of the
# model that the runs this repo compares against never saw.
class GroundedBiographyResponse(StrictModel):
    profile: Biography
    grounding: list[GroundingItem]


def structured_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON Schema to hand a structured-output request."""
    return model.model_json_schema()
