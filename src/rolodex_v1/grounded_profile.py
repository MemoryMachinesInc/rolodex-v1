"""Turning one model response into one grounded profile.

This lives apart from either regime because both regimes must do it
*identically*. AGENTS.md makes schema parity load-bearing -- "a regime cannot
quietly grow its own shape" -- and the same argument covers the step just after
validation: a profile is only comparable across regimes if the grounding was
attached under the same contract. Held as one function, that cannot drift; held
as a copy per regime, it drifts the first time somebody fixes one of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rolodex_v1.grounding import attach_profile_grounding
from rolodex_v1.profile_schema import Biography

if TYPE_CHECKING:
    from rolodex_v1.evidence_pack import EvidencePack


def build_profile(response: dict[str, Any], pack: EvidencePack) -> dict[str, Any]:
    """Validate the response and attach resolved grounding.

    Coverage is not required: an uncited leaf is left ungrounded and scored as
    such. How completely a model cites is one of the things these regimes exist
    to measure, so failing the profile would discard the measurement.
    """
    profile = Biography.model_validate(response["profile"], strict=True).model_dump(
        mode="json"
    )
    spans_by_path = {
        item["path"]: item["spans"]
        for item in response.get("grounding") or []
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    return attach_profile_grounding(
        profile, spans_by_path, pack.context, require_complete=False
    )
