"""Prompt text both regimes must say identically.

The two regimes are scored against each other, so prompt wording that drifts
between them does not merely look untidy: it makes the comparison measure the
prompts rather than the regimes. This is the same move
:mod:`rolodex_v1.profile_schema` and :mod:`rolodex_v1.grounded_profile` already
make for the artifact -- held as a copy per regime, it drifts the first time
somebody fixes one of them.

Only text that is *already* word-for-word identical in both regimes lives here.
Everything each regime says in its own vocabulary stays where it is:

- The agentic prompt describes tools, a pack file and a corpus it may grep; the
  in-context prompt describes numbered source lines in a single request. The
  grounding contract is therefore two texts, not one, and deliberately so.
- The in-context rules legislate age, height, emails and positions. The agentic
  base rules deliberately do not -- the tuned addendum owns those fields there,
  and ``test_the_base_rules_leave_the_scored_fields_to_the_addendum`` fails if
  they creep back. So rules that are supersets on one side are not shared; only
  the sentence the two have in common is.

Sentences here are canonical single-space prose. Line breaks are presentation:
each regime wraps them the way its own prompt is laid out.
"""

from __future__ import annotations

# ``relationship_to_user`` is defined against the Rolodex owner, so all three
# cases below have to be stated the same way in both regimes or the field is
# being asked for differently in each.
OWNER_IS_ANOTHER_PERSON = (
    "The Rolodex owner is **{owner}**. Describe the direct, evidence-supported "
    "relationship between the owner and **{name}** in `relationship_to_user`. "
    "Do not list the owner as a common connection merely because the owner has "
    "a direct relationship with this person."
)

OWNER_IS_THE_SUBJECT = (
    "This is the Rolodex owner's own profile; set `relationship_to_user` to null."
)

OWNER_UNKNOWN = (
    "The Rolodex owner's identity was not supplied; set `relationship_to_user` to null."
)


def owner_context(name: str, profile_owner: str | None) -> str:
    """The owner sentence for one entity, as a single unwrapped paragraph."""
    if profile_owner and profile_owner.casefold() != name.casefold():
        return OWNER_IS_ANOTHER_PERSON.format(owner=profile_owner, name=name)
    if profile_owner:
        return OWNER_IS_THE_SUBJECT
    return OWNER_UNKNOWN


RULE_EVERY_FIELD_REQUIRED = (
    "Every schema field is required. Use null for an unsupported scalar or "
    "nested object and [] for an unsupported collection. Never invent facts or "
    "unsupported specificity."
)

RULE_ALIASES = "Put alternate names, handles, and name variants in `aliases`."

RULE_CONNECTIONS = "Include all explicitly supported people mentioned as connections."

RULE_DESCRIPTIVE_FIELDS = (
    "Make descriptive fields complete, grammatical, specific, direct, and "
    "concise while preserving accuracy."
)
