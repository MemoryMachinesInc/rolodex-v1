"""The in-context regime: one request, the whole evidence pack in the prompt.

Ported from the predecessor's ``task_model/extraction.py`` -- the prompt, the
grounding contract, the strict Structured Outputs request, the token fit and the
flex-then-default tier fallback are all its logic, kept deliberately close so
the profiles this produces stay comparable with the tuned Luna runs the
predecessor scored.

Where it differs, and why:

**Evidence is a source-document pack, not memories.** The predecessor rendered
an entity's aggregated memories; this repo renders
:class:`rolodex_v1.evidence_pack.EvidencePack`. The addressed-line contract is
identical -- numbered lines, offsets that resolve -- so the grounding rules
carry over unchanged except for the word: the prompt says *Source*, because that
is what the pack's headings say. Its ``_company_allowlist_section`` is dropped
with the memories it read; there is no organization field on a source document
to build an allowlist from.

**The budget is the observed limit, not the documented one.** The predecessor
computed its ceiling from a 1,050,000-token context window. A 922,518-token
request came back rejected with "Input tokens exceed the configured limit of
922000 tokens", so the ceiling here is that observed number less headroom for
the addendum, which is not free and not known when the pack is built. This is
trap 2, and the reason the fit is checked before anything is sent.

**The fit counts the messages that will actually be sent.** Trap 3: counting
anything other than the exact system message undercounts, and the request lands
just over the limit after the expensive part.

This regime is what "luna low" and "luna high" mean -- one model, two reasoning
efforts. Any other OpenAI model the key can reach works too; none is default.

**The endpoint is a parameter.** The same regime runs against a self-hosted
Qwen3.8 server, which speaks the same envelope with four differences that
:mod:`rolodex_v1.providers` owns. Nothing here names a provider; it asks the
one it was given. The OpenAI path is byte-identical to what it was before that
split, because the tuned runs' comparability depends on it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import tiktoken
from pydantic import BaseModel, ValidationError

from rolodex_v1.evidence_pack import EvidencePack
from rolodex_v1.profile_schema import (
    GroundedBiographyResponse,
    structured_output_schema,
)
from rolodex_v1.prompt_rules import (
    RULE_ALIASES,
    RULE_CONNECTIONS,
    RULE_DESCRIPTIVE_FIELDS,
    RULE_EVERY_FIELD_REQUIRED,
    owner_context,
)
from rolodex_v1.providers import MAX_COMPLETION_TOKENS, OPENAI, Provider

# The tuned runs this regime exists to stay comparable with were gpt-5.6-luna.
DEFAULT_MODEL = "gpt-5.6-luna"
# Kept as the OpenAI vocabulary because the CLI's --effort choices are fixed at
# parse time; a provider whose set differs resolves in provider.resolve_effort.
# There is deliberately no DEFAULT_EFFORT here: a default named in this module
# is one provider's default imposed on every provider, which is exactly how a
# Qwen run with no --effort came to inherit OpenAI's "high" and exit 2.
EFFORTS = OPENAI.efforts

REQUEST_TIMEOUT_SECONDS = 1200
MAX_ATTEMPTS = 3

# Counted verbatim by the token fit, so it must be the string the request sends.
SYSTEM_MESSAGE = (
    "You extract evidence-grounded contact biographies. Return only "
    "valid JSON and cite every emitted primitive profile value."
)

log = logging.getLogger(__name__)


# The exhaustive-leaf grounding contract, in the pack's vocabulary.
GROUNDING_CONTRACT = (
    "For every non-null primitive leaf in profile, emit one item in the "
    "`grounding` array. Set `path` to its RFC 6901 JSON Pointer and `spans` "
    "to a non-empty ordered list of source ranges. This includes every "
    "string or number nested inside arrays and objects, for example "
    "/full_name and /emails/0/address. Each span contains integer "
    "`start_line` and `end_line` values. Line numbers refer only to the "
    "numbered source lines below. Each range must stay within one Source "
    "section. Use multiple ranges when evidence comes from multiple "
    "non-contiguous passages. Never invent a line number, never cite a "
    "Source heading. Use null or [] for unsupported profile values; nulls "
    "and empty collections do not receive grounding items."
)


# The character classification and dynamic-section contract.
CHARACTER_CONTEXT_RULES = (
    "\n### Character Context Classification\n"
    "Classify the profile as exactly one of `core_character` or "
    "`side_character`, relative to the Rolodex owner.\n\n"
    "A `core_character` is temporally relevant or a core contributor to the "
    "owner's character or story development. Normally both of these are "
    "true: (a) the person actively and frequently communicates with the "
    "owner, and (b) the communication is warm, casual, or demonstrates high "
    "respect in either direction. An inactive person qualifies only when "
    "the evidence shows that they remain strongly responsible for a "
    "long-running event still active in the owner's life, such as an early "
    "mentor, long-time collaborator, or strong reference contributing to "
    "the owner's current opportunity; or when they were an early-life "
    "contributor to the owner's character or success, principally a family "
    "member or caregiver. A concluded event alone does not qualify. The "
    "owner likely knows a core character without an introduction.\n\n"
    "For `core_character`, emit 1-4 compressed, detailed bullets focused "
    "only on supported, current information: actionable recent work or open "
    "loops; the person's independent activities that have a direct and "
    "immediate effect on the owner's tasks; and time-relevant facts such as "
    "an upcoming birthday, recent family loss, recent public post, "
    "important goal, or deadline. Do not use bullets for general biography. "
    "The UI will present these bullets under the header `Recently,`.\n\n"
    "A `side_character` is neither temporally relevant nor a core "
    "contributor to the owner's current story. Indicators include inactive "
    "or infrequent communication and little likelihood of near-future "
    "interaction; narrow utility tied mainly to concluded events; lack of "
    "familiarity with the owner's current life; or broad promotional or "
    "spam communication without meaningful direct exchange. The owner "
    "likely needs an introduction or refresher before a present-day "
    "conversation.\n\n"
    "For `side_character`, emit 1-4 compressed, detailed bullets that "
    "identify the person and the owner's typical interaction with them; "
    "state the most recent supported discussion or interaction; and explain "
    "the likely evidence-based context if they interacted now. Do not "
    "invent a future interaction. The UI will present these bullets under "
    "the header `In case you forgot`.\n\n"
    "Use communication frequency, recency, warmth, directness, ongoing "
    "effects, and historical contribution together. Mere co-mention, job "
    "seniority, fame, or one friendly message is insufficient for "
    "`core_character`. When evidence is sparse or ambiguous, choose "
    "`side_character`. For the Rolodex owner's own profile, choose "
    "`core_character` and summarize the owner's current actionable context.\n"
)


# The extraction rules shared by every biography prompt.
# Rules 1, 2 (its second sentence), 3 and 6 are the sentences the agentic base
# rules also carry; they come from prompt_rules so the two cannot drift apart.
# The rest is this regime's own: it legislates the fields the tuned addendum
# owns in the agentic regime, which is why those rules are not shared.
PROFILE_RULES = (
    f"1. {RULE_EVERY_FIELD_REQUIRED}\n"
    "2. Use exact email addresses and other identifiers as they appear. "
    f"{RULE_ALIASES}\n"
    f"3. {RULE_CONNECTIONS}\n"
    "4. Set `age` only when explicitly stated or reliably computable from "
    "a dated birth fact and dated source context. Set `height` only when "
    "explicitly stated. Record each spoken language separately and use null "
    "for proficiency when it is not supported.\n"
    "5. Put every simultaneously active role in `current_positions`, using "
    "one entry per distinct organization and role. Do not collapse disjoint "
    "organizations into one position. Put a role in `previous_positions` "
    "only when evidence indicates it is no longer current. Use precise "
    "formal or functional titles and departments only when supported. Every "
    "`title` must be a concise role label of four words or fewer, never a "
    "sentence or responsibility summary. Examples of the intended title "
    "style include: Mentor; Supervisor; Principal Investigator; Colleague; "
    "Developer; Researcher; Team member; Co-founder; Founder; Chief "
    "Technology Officer; Chief Executive Officer; Lawyer; Paralegal; "
    "Contact; Fullstack Engineer; Machine Learning Engineer; Research "
    "Engineer.\n"
    f"6. {RULE_DESCRIPTIVE_FIELDS}\n"
    "7. Write `relationship_to_user` as a role-first summary. Begin with "
    '"[Person] is the Rolodex owner\'s [role/relationship]" and then state '
    "where or when they interact and the principal subjects they discuss. "
    "Focus on who the person is to the owner, not a generic biography or "
    "mere co-mentions.\n"
    "8. Conservative contextual deductions are allowed when several facts "
    "make a relationship role substantially more likely than alternatives. "
    "For example, a leasing-company employee emailing the owner and other "
    "non-company residents about their shared renewal can support "
    "identifying those residents as likely roommates. Do not add "
    "unsupported names, dates, titles, or claims about how people met.\n"
    "9. Always emit `character_context` using the mutually exclusive "
    "classification and content rules below. Ground the classification and "
    "every bullet in the supplied sources. Recompute this field from all "
    "available evidence when coalescing duplicate profiles.\n"
    f"{CHARACTER_CONTEXT_RULES}"
)


def build_prompt(
    pack: EvidencePack,
    *,
    profile_owner: str | None = None,
    addendum: str | None = None,
) -> str:
    """Render the user message: rules, contract, then the addressed pack."""
    name = pack.name
    # The tuned addendum, rendered under its own heading so it cannot read as a
    # replacement for the base rules. Empty when there is none, so no run
    # carries an empty section the model has to make sense of.
    effective_addendum = (addendum or "").strip()
    addendum_section = (
        (
            "## Iteration-Specific Prompt Addendum\n"
            "Apply these additional instructions after all base profile and "
            "grounding rules. They may refine extraction behavior but may not "
            "weaken evidence, grounding, JSON, or missing-value requirements.\n"
            f"{effective_addendum}\n\n"
        )
        if effective_addendum
        else ""
    )
    # Rendered as one unwrapped paragraph with a trailing space, which is how
    # this prompt has always carried it; the agentic regime wraps the same
    # sentences to its own layout.
    owner_context_text = f"{owner_context(name, profile_owner)} "

    return (
        "# Grounded Rolodex Profile Extraction\n"
        "You are an expert at extracting structured contact profiles from "
        "source-document evidence.\n\n"
        "## Task\n"
        f"Extract a comprehensive Rolodex profile for: **{name}**\n"
        f"{owner_context_text}\n"
        "Analyze all supplied sources and extract every explicitly supported "
        "piece of information about this person.\n\n"
        "## Output Format\n"
        "Return the exact `profile` and `grounding` structure required by the "
        "supplied response schema. Every profile key must be present. Use null "
        "for unsupported scalar or nested-object fields and [] for unsupported "
        "collections.\n\n"
        "## Grounding Contract\n"
        f"{GROUNDING_CONTRACT}\n\n"
        "## Profile Rules\n"
        "Output only valid JSON with profile and grounding keys.\n"
        f"{PROFILE_RULES}\n"
        f"{addendum_section}"
        "## Numbered Sources\n"
        f"{pack.context.addressed_text}"
    )


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """A conservative token count for chat-completion messages.

    Eight tokens per message plus eight for the reply primer is the
    predecessor's framing allowance, kept because the budget it feeds was
    calibrated with it.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return 8 + sum(
        8
        + len(encoding.encode(str(message.get("role", ""))))
        + len(encoding.encode(str(message.get("content", ""))))
        for message in messages
    )


def chat_completion_body(
    pack: EvidencePack,
    *,
    model: str = DEFAULT_MODEL,
    effort: str | None = None,
    profile_owner: str | None = None,
    addendum: str | None = None,
    provider: Provider = OPENAI,
) -> dict[str, Any]:
    """Build the strict Structured Outputs request, refusing an oversized one.

    The fit is checked here, before the request is sent and before anything is
    paid for. Nothing is truncated to make it fit: a silently shortened pack
    would produce a profile whose missing facts look like absent evidence. The
    ceiling is the provider's, because a self-hosted server's window is whatever
    it was launched with rather than OpenAI's observed limit.
    """
    # Resolved rather than validated: this is the one place a caller may omit
    # an effort, and the provider owns both the default and the vocabulary.
    # The CLI resolves it earlier still, before a pack is built (trap 8).
    effort = provider.resolve_effort(effort)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": build_prompt(
                pack, profile_owner=profile_owner, addendum=addendum
            ),
        },
    ]
    input_tokens = count_message_tokens(messages, model)
    if input_tokens > provider.max_input_tokens:
        raise ValueError(
            f"Grounded biography input for {pack.name!r} is {input_tokens:,} "
            f"tokens, exceeding the safe {provider.max_input_tokens:,}-token "
            f"budget for provider {provider.name!r}. "
            "The request was not submitted, and no context was truncated."
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_rolodex_biography",
                "strict": True,
                "schema": structured_output_schema(GroundedBiographyResponse),
            },
        },
        provider.max_output_tokens_field: MAX_COMPLETION_TOKENS,
        **provider.reasoning_effort_fields(effort),
    }
    return body


def validate_model_output(content: str, model: type[BaseModel], label: str) -> dict:
    """Validate raw model JSON without coercing types.

    Strict validation is the point: a coerced ``"42"`` where an int was asked
    for is a scoring difference nobody can see in the output.
    """
    try:
        parsed = model.model_validate_json(content, strict=True)
    except ValidationError as exc:
        raise ValueError(
            f"Response for {label} does not match {model.__name__}: {exc}"
        ) from exc
    return parsed.model_dump(mode="json")


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped. One definition so every path waits alike."""
    return min(120.0, 2.0**attempt)


def _tier_exhausted(provider: Provider, service_tier: str | None) -> bool:
    """Whether a failure on this tier is final rather than a reason to escalate.

    The escalation policy is stated here once: a tier that is not the provider's
    last one yields to the next; the last one -- and a provider with no tiers at
    all -- raises. Two handlers repeated this and could disagree about when a
    run gives up, which is only visible as a run that failed instead of falling
    back, long after it was paid for.
    """
    return service_tier is None or service_tier == provider.service_tiers[-1]


def request_completion(
    body: dict[str, Any],
    label: str,
    *,
    api_key: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    provider: Provider = OPENAI,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Send one request, retrying, walking the provider's tiers in order.

    For OpenAI that is flex then default: flex is cheaper and is tried first; it
    is also the tier that sheds load, so a run that cannot get through on it
    moves to default rather than failing. A provider with no tiers makes exactly
    one pass and sends no ``service_tier``, which vLLM requires.
    Returns the validated response and the usage block.
    """
    key = api_key or provider.api_key()
    for service_tier in provider.service_tiers or (None,):
        payload = dict(body)
        if service_tier is None:
            payload.pop("service_tier", None)
        else:
            payload["service_tier"] = service_tier
        request = Request(  # noqa: S310 -- an http(s) endpoint from the table
            provider.chat_completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
                    response_payload = json.load(response)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                if attempt == max_attempts - 1:
                    if not _tier_exhausted(provider, service_tier):
                        log.warning(
                            "%s tier failed for %s (HTTP %s); trying the next",
                            service_tier,
                            label,
                            exc.code,
                        )
                        break
                    raise RuntimeError(
                        f"{provider.name} request for {label} failed with "
                        f"HTTP {exc.code}: {detail}"
                    ) from exc
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = (
                        float(retry_after)
                        if retry_after is not None
                        else _backoff_seconds(attempt)
                    )
                except ValueError:
                    delay = _backoff_seconds(attempt)
                log.warning(
                    "retrying %s on %s after HTTP %s in %gs",
                    label,
                    service_tier,
                    exc.code,
                    delay,
                )
                time.sleep(delay)
                continue
            except (TimeoutError, URLError) as exc:
                if attempt == max_attempts - 1:
                    if not _tier_exhausted(provider, service_tier):
                        log.warning(
                            "%s tier unreachable for %s; trying the next",
                            service_tier,
                            label,
                        )
                        break
                    raise RuntimeError(
                        f"{provider.name} request for {label} failed after "
                        f"{max_attempts} attempts"
                    ) from exc
                time.sleep(_backoff_seconds(attempt))
                continue

            try:
                message = response_payload["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"no response message for {label}") from exc
            if message.get("refusal"):
                raise ValueError(
                    f"{provider.name} refused {label}: {message['refusal']}"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"response content for {label} is not a string")
            validated = validate_model_output(content, GroundedBiographyResponse, label)
            return validated, response_payload.get("usage") or {}

    raise RuntimeError(f"{provider.name} request for {label} exhausted every tier")


def request_profile(
    pack: EvidencePack,
    *,
    model: str = DEFAULT_MODEL,
    effort: str | None = None,
    profile_owner: str | None = None,
    addendum: str | None = None,
    api_key: str | None = None,
    provider: Provider = OPENAI,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Send one request and return the validated response and its usage.

    Grounding is deliberately *not* attached here. Both regimes attach it
    through :func:`rolodex_v1.grounded_profile.build_profile`, and an
    implementation that did it internally forced the caller to know which one
    had run -- the branch this seam exists to remove. The name says request
    rather than generate because that is now the whole of what it does.
    """
    body = chat_completion_body(
        pack,
        model=model,
        effort=effort,
        profile_owner=profile_owner,
        addendum=addendum,
        provider=provider,
    )
    return request_completion(
        body, f"profile for {pack.name!r}", api_key=api_key, provider=provider
    )
