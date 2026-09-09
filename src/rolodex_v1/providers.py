"""Where an in-context request goes, and the dialect that endpoint speaks.

The in-context regime is one chat-completions request. OpenAI and a
self-hosted vLLM server agree on the envelope and disagree on four details that
this particular request happens to use, every one of which is a hard error
rather than a degradation:

- **Service tiers.** ``service_tier`` is an OpenAI concept. vLLM rejects the
  unknown field outright, so a provider that has no tiers must send none --
  which is also why the tier fallback loop is driven by this table rather than
  by a literal ``("flex", "default")``.
- **Reasoning effort.** OpenAI reads a top-level ``reasoning_effort``. Qwen3.8
  under vLLM reads it from ``chat_template_kwargs``, because the effort dial is
  implemented in the model's chat template.
- **The effort vocabulary is not shared.** OpenAI's top level is ``high``;
  Qwen3.8's is ``xhigh``. Accepting ``high`` for Qwen and quietly sending it
  would select a different amount of thinking than the caller asked for, so the
  valid set belongs to the provider and is validated against it.
- **The input ceiling.** OpenAI's is the observed 922,000-token rejection limit;
  a self-hosted server's is whatever ``--max-model-len`` it was launched with.
  Fitting against the wrong one either wastes a request or refuses a request
  that would have fit.

Adding a provider is adding an entry here. Nothing else in the codebase should
learn a provider's name: :mod:`rolodex_v1.in_context` takes a ``Provider`` and
asks it, and :mod:`rolodex_v1.build_profiles` resolves one from the model id.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from rolodex_v1.credentials import read_env_value

# The deployed OpenAI limit, observed by rejection rather than documented: a
# 922,518-token request came back "Input tokens exceed the configured limit of
# 922000 tokens". Headroom covers the prompt addendum, whose size is not known
# when the pack is built (the tuned one is ~2.2K tokens).
OPENAI_OBSERVED_INPUT_LIMIT = 922_000
REQUEST_HEADROOM_TOKENS = 22_000

# What the bundled Modal deployment launches vLLM with. Qwen3.8's native window
# is 262,144 tokens; the deployment applies the YaRN rope override to reach
# this, because an entity's pack can exceed the native window.
QWEN_MAX_MODEL_LEN = 960_000

# The completion reservation, defined here because the Qwen input budget is
# derived from it and the request body sends it: two copies would let the
# budget and the request drift apart silently.
MAX_COMPLETION_TOKENS = 32_768

# Qwen3.8's effort vocabulary, available without a base URL so the CLI can
# offer it in --choices before it knows whether a server was configured.
QWEN_EFFORTS = ("low", "medium", "xhigh")
QWEN_DEFAULT_EFFORT = "xhigh"

ENV_QWEN_BASE_URL = "ROLODEX_V1_QWEN_BASE_URL"


@dataclass(frozen=True)
class Provider:
    """One endpoint and the request dialect it accepts."""

    name: str
    chat_completions_url: str
    api_key_name: str
    #: Read when ``api_key_name`` is unset. Empty means there is no fallback.
    api_key_fallback_name: str
    efforts: tuple[str, ...]
    default_effort: str
    #: Tried in order. Empty means send no ``service_tier`` field at all.
    service_tiers: tuple[str, ...]
    #: ``"top-level"`` sends ``reasoning_effort``; ``"chat-template"`` nests it
    #: in ``chat_template_kwargs``.
    reasoning_effort_style: str
    max_output_tokens_field: str
    max_input_tokens: int
    #: Low and high estimate per entity, for the pre-flight price.
    cost_per_entity_usd: tuple[float, float]

    def api_key(self) -> str:
        """The credential for this endpoint, spec-named variable first.

        The fallback exists so a machine configured with the generic name keeps
        working; the error names both so a fresh one knows what to set.
        """
        try:
            return read_env_value(self.api_key_name)
        except RuntimeError:
            if not self.api_key_fallback_name:
                raise
        try:
            return read_env_value(self.api_key_fallback_name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{self.api_key_name} (or {self.api_key_fallback_name}) must be "
                f"set for provider {self.name!r}: {exc}"
            ) from exc

    def validate_effort(self, effort: str) -> None:
        """Reject an effort this provider does not implement.

        Named separately from the request builder so the CLI can fail before a
        pack is built rather than after.
        """
        if effort not in self.efforts:
            raise ValueError(
                f"effort must be one of {self.efforts} for provider "
                f"{self.name!r}, got {effort!r}"
            )

    def resolve_effort(self, effort: str | None) -> str:
        """The effort this run will actually buy.

        One interface for the whole decision: ``None`` means the caller asked
        for nothing and gets this provider's own default, and anything else is
        validated. A default fixed anywhere but here is a default fixed to one
        provider's vocabulary, which is how a Qwen run with no ``--effort``
        came to inherit OpenAI's ``high`` and be rejected for it.
        """
        if effort is None:
            return self.default_effort
        self.validate_effort(effort)
        return effort

    def reasoning_effort_fields(self, effort: str) -> dict[str, Any]:
        """The request fields that carry ``effort`` in this provider's dialect.

        The wire shape belongs beside the vocabulary that names it: OpenAI
        reads a top-level ``reasoning_effort``, while Qwen3.8's dial is
        implemented in the chat template and is read from there.
        """
        if self.reasoning_effort_style == "top-level":
            return {"reasoning_effort": effort}
        return {"chat_template_kwargs": {"reasoning_effort": effort}}


OPENAI = Provider(
    name="openai",
    chat_completions_url="https://api.openai.com/v1/chat/completions",
    # The spec's key for the extraction model, with the generic name kept as a
    # fallback so a machine configured before the split keeps working.
    api_key_name="OPENAI_API_KEY_SMALL",
    api_key_fallback_name="OPENAI_API_KEY",
    efforts=("low", "medium", "high"),
    default_effort="high",
    # Flex is cheaper and is tried first; it is also the tier that sheds load,
    # so a run that cannot get through on it moves to default rather than
    # failing.
    service_tiers=("flex", "default"),
    reasoning_effort_style="top-level",
    max_output_tokens_field="max_completion_tokens",
    max_input_tokens=OPENAI_OBSERVED_INPUT_LIMIT - REQUEST_HEADROOM_TOKENS,
    cost_per_entity_usd=(0.03, 0.15),
)


def qwen_provider(base_url: str | None = None) -> Provider:
    """Qwen3.8-27B behind an OpenAI-compatible vLLM server.

    The base URL is a property of the machine -- which server is up right now --
    rather than of the artifact, so it is not in the output filename: the model
    code already says which model produced the profiles. It comes from the flag,
    then the environment, and is required because there is no sensible default
    to guess at.

    Cost is quoted as zero because a self-hosted run is billed as GPU time by
    the hour, not per request. The run is not free; its cost simply is not
    attributable per entity, and pretending otherwise with a fabricated
    per-entity rate would put an invented number in the pre-flight estimate.
    """
    resolved = base_url or os.environ.get(ENV_QWEN_BASE_URL)
    if not resolved:
        raise RuntimeError(
            "A Qwen run needs the served endpoint: pass --qwen-base-url or set "
            f"{ENV_QWEN_BASE_URL} to the deployment's /v1 URL. "
            "See deploy/modal_qwen.py."
        )
    return Provider(
        name="qwen",
        chat_completions_url=resolved.rstrip("/") + "/chat/completions",
        api_key_name="QWEN_API_KEY",
        api_key_fallback_name="",
        # Qwen3.8's dial, which has no "high": the top of its range is xhigh.
        efforts=QWEN_EFFORTS,
        default_effort=QWEN_DEFAULT_EFFORT,
        service_tiers=(),
        reasoning_effort_style="chat-template",
        max_output_tokens_field="max_tokens",
        max_input_tokens=QWEN_MAX_MODEL_LEN - MAX_COMPLETION_TOKENS,
        cost_per_entity_usd=(0.0, 0.0),
    )


def provider_for(model: str, *, qwen_base_url: str | None = None) -> Provider:
    """Resolve the provider a model id is served by.

    Taking a model id means ``--model qwen3.8-27b`` is enough to reach the
    self-hosted server, with no second flag that could disagree with the first.
    """
    if model.lower().startswith("qwen"):
        return qwen_provider(qwen_base_url)
    return OPENAI
