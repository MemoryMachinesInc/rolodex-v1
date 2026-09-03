"""Properties of the in-context regime that must hold before a request is sent.

Nothing here calls OpenAI. What is worth testing without spending is everything
that decides *whether* to spend: the token fit, the exact system message the fit
counts, and the strict validation of what comes back.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from rolodex_v1 import in_context, providers
from rolodex_v1.evidence_pack import build_pack
from rolodex_v1.resolved_entities import Alias, ResolvedEntity

JOSH = ResolvedEntity(
    entity_id="proto:participant_person:20",
    canonical_name="Josh Earnest",
    case="participant_person",
    canonical_type="person",
    mention_count=1,
    memory_count=1,
    aliases=(Alias(text="Josh Earnest", probability=1.0),),
)


def pack(tmp_path: Path, text: str = "Josh Earnest is the Press Secretary."):
    directory = tmp_path / "corpus"
    directory.mkdir(exist_ok=True)
    (directory / "doc.txt").write_text(text, encoding="utf-8")
    return build_pack(directory, JOSH, budget_chars=90_000, max_doc_chars=4_000)


def test_the_body_carries_strict_structured_outputs(tmp_path: Path) -> None:
    """A coerced value is a scoring difference nobody can see in the output."""
    body = in_context.chat_completion_body(pack(tmp_path))
    schema = body["response_format"]["json_schema"]
    assert body["response_format"]["type"] == "json_schema"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_the_fit_counts_the_system_message_that_will_be_sent(tmp_path: Path) -> None:
    """Trap 3: counting any other string undercounts, and the request lands over.

    The message in the body and the message the counter sees must be one object,
    not two strings that happen to match today.
    """
    body = in_context.chat_completion_body(pack(tmp_path))
    assert body["messages"][0]["content"] == in_context.SYSTEM_MESSAGE
    counted = in_context.count_message_tokens(body["messages"], body["model"])
    assert counted > 0


def test_the_budget_is_the_observed_limit_not_the_documented_one() -> None:
    """Trap 2: a 922,518-token request was rejected against a 1,050,000 window.

    The number lives on the provider now, because a self-hosted server's ceiling
    is its own; OpenAI's must still be the observed limit less headroom.
    """
    assert providers.OPENAI_OBSERVED_INPUT_LIMIT == 922_000
    assert providers.OPENAI.max_input_tokens < providers.OPENAI_OBSERVED_INPUT_LIMIT


def test_an_oversized_request_is_refused_rather_than_truncated(
    tmp_path: Path,
) -> None:
    """A silently shortened pack looks exactly like absent evidence."""
    tiny = dataclasses.replace(providers.OPENAI, max_input_tokens=10)
    with pytest.raises(ValueError, match="was not submitted"):
        in_context.chat_completion_body(pack(tmp_path), provider=tiny)


def test_an_unknown_effort_is_rejected(tmp_path: Path) -> None:
    """Low and high are separate arms; a typo would mislabel one as the other."""
    with pytest.raises(ValueError, match="effort must be one of"):
        in_context.chat_completion_body(pack(tmp_path), effort="highest")


def test_the_effort_reaches_the_request(tmp_path: Path) -> None:
    body = in_context.chat_completion_body(pack(tmp_path), effort="low")
    assert body["reasoning_effort"] == "low"


def test_the_addendum_may_not_replace_the_base_rules(tmp_path: Path) -> None:
    """It is appended under its own heading, after the grounding contract."""
    body = in_context.chat_completion_body(pack(tmp_path), addendum="EXTRA RULE")
    prompt = body["messages"][1]["content"]
    assert "Grounding Contract" in prompt
    assert prompt.index("Grounding Contract") < prompt.index("EXTRA RULE")
    assert "may not weaken evidence" in prompt


def test_no_addendum_leaves_no_empty_section(tmp_path: Path) -> None:
    body = in_context.chat_completion_body(pack(tmp_path), addendum=None)
    assert "Prompt Addendum" not in body["messages"][1]["content"]


def test_the_prompt_ends_with_the_addressed_pack(tmp_path: Path) -> None:
    """The model cites these line numbers, so they must be the last thing it reads."""
    built = pack(tmp_path)
    prompt = in_context.build_prompt(built)
    assert prompt.endswith(built.context.addressed_text)


def test_the_owner_field_is_nulled_when_no_owner_is_supplied(tmp_path: Path) -> None:
    """relationship_to_user is defined against the owner; invent neither."""
    assert "was not supplied" in in_context.build_prompt(pack(tmp_path))


def test_the_owners_own_profile_gets_a_null_relationship(tmp_path: Path) -> None:
    prompt = in_context.build_prompt(pack(tmp_path), profile_owner="Josh Earnest")
    assert "owner's own profile" in prompt


def test_a_response_that_does_not_match_the_schema_is_refused() -> None:
    """Strict validation, so "42" arriving where an int was asked for fails here."""
    with pytest.raises(ValueError, match="does not match"):
        in_context.validate_model_output(
            json.dumps({"profile": {}, "grounding": []}),
            in_context.GroundedBiographyResponse,
            "test",
        )


def test_the_prompt_text_is_byte_for_byte_what_it_was(tmp_path: Path) -> None:
    """The prompt is tuned; editing it is an experiment that needs a score.

    The digest covers everything above the numbered sources -- the rules, the
    grounding contract and the addendum section -- and deliberately stops there,
    because the pack below it is another module's artifact. These are the
    digests of the prompt as the tuned runs rendered it, taken before the four
    constant-returning helpers that built it were inlined.
    """
    built = pack(tmp_path, "x")
    for addendum, digest in (
        (None, "9e2db11f7a3eba614b87845ee1fd1b50b3616aac11673ad5144a65970a756a68"),
        (
            "EXTRA RULE",
            "fd5c745da25e096813048d952f42e7417b2e897881ce22d469d584311543069f",
        ),
    ):
        prompt = in_context.build_prompt(
            built, profile_owner="Barack Obama", addendum=addendum
        )
        prefix = prompt.split("## Numbered Sources")[0]
        assert hashlib.sha256(prefix.encode("utf-8")).hexdigest() == digest


def test_an_omitted_effort_is_the_providers_own(tmp_path: Path) -> None:
    """Not this module's: a default named here is one provider's, for all."""
    qwen = providers.qwen_provider("https://example.invalid/v1")
    body = in_context.chat_completion_body(pack(tmp_path), provider=qwen)
    assert body["chat_template_kwargs"] == {
        "reasoning_effort": providers.QWEN_DEFAULT_EFFORT
    }
    assert not hasattr(in_context, "DEFAULT_EFFORT")


def test_the_fit_uses_the_handed_providers_limit_not_a_global(tmp_path: Path) -> None:
    """A module-level ceiling would fit every provider against OpenAI's.

    Handing in a provider whose window is tiny must refuse a pack that OpenAI
    would accept; nothing in this module may consult a limit of its own.
    """
    built = pack(tmp_path)
    in_context.chat_completion_body(built)  # fits under OpenAI's ceiling
    tiny = dataclasses.replace(providers.OPENAI, name="tiny", max_input_tokens=10)
    with pytest.raises(ValueError, match="provider 'tiny'"):
        in_context.chat_completion_body(built, provider=tiny)


def test_every_retry_path_waits_on_one_backoff(tmp_path: Path) -> None:
    """Two handlers that compute their own delay can disagree about giving up."""
    assert in_context._backoff_seconds(0) == 1.0
    assert in_context._backoff_seconds(50) == 120.0


def test_only_the_last_tier_is_terminal() -> None:
    """A failure before the last tier escalates; on the last one it raises."""
    assert not in_context._tier_exhausted(providers.OPENAI, "flex")
    assert in_context._tier_exhausted(
        providers.OPENAI, providers.OPENAI.service_tiers[-1]
    )
    # A provider with no tiers sends no service_tier at all, and makes one pass.
    assert in_context._tier_exhausted(providers.OPENAI, None)
