"""Properties of the prompt text the two regimes must state identically.

The regimes are scored against each other. Prompt wording that drifts between
them makes that comparison measure the prompts rather than the regimes, and
nothing in either output would show it -- which is why the shared sentences have
to be provably the ones both prompts render.
"""

from __future__ import annotations

from rolodex_v1 import agentic, in_context, prompt_rules


def normalised(text: str) -> str:
    """Text with layout removed: the regimes wrap differently on purpose."""
    return " ".join(text.split())


def test_the_shared_rules_appear_verbatim_in_both_regimes() -> None:
    """A copy per regime drifts the first time somebody fixes one of them."""
    shared = (
        prompt_rules.RULE_EVERY_FIELD_REQUIRED,
        prompt_rules.RULE_ALIASES,
        prompt_rules.RULE_CONNECTIONS,
        prompt_rules.RULE_DESCRIPTIVE_FIELDS,
    )
    for rule in shared:
        assert normalised(rule) in normalised(in_context.PROFILE_RULES)
        assert normalised(rule) in normalised(agentic.BASE_RULES)


def test_the_owner_context_covers_the_three_cases_once() -> None:
    """`relationship_to_user` is defined against the owner, or it is null."""
    assert "**Barack Obama**" in prompt_rules.owner_context("Josh", "Barack Obama")
    assert (
        prompt_rules.owner_context("Josh", "josh") == prompt_rules.OWNER_IS_THE_SUBJECT
    )
    assert prompt_rules.owner_context("Josh", None) == prompt_rules.OWNER_UNKNOWN


def test_the_shared_text_carries_no_regime_vocabulary() -> None:
    """What each regime says in its own words stays in its own module.

    The agentic prompt speaks of a pack file and tools it may grep; the
    in-context prompt speaks of numbered source lines in one request. Those
    differences are deliberate, and a sentence here that named either would be
    wrong in the other regime's prompt.
    """
    text = " ".join(
        value
        for name, value in vars(prompt_rules).items()
        if name.isupper() and isinstance(value, str)
    ).lower()
    for word in ("grep", "evidence pack", "numbered source", "corpus"):
        assert word not in text
