"""Schema parity with the runs this repo's numbers are compared against.

Every regime -- ``memory``, ``chunks``, ``agentic`` -- emits the same
``Biography``. Within this repo that is structural: there is one
``profile_schema`` module and everything imports it. What is *not* structural is
parity with the tuned Luna/Sol runs, whose scores are the baseline. Those ran in
a different repository against their own copy of this schema, so nothing but a
test stops the two drifting apart.

``fixtures/tuned_runs_grounded_schema.json`` is that copy's emitted schema,
taken verbatim from
``z-r-eval_prompt_and_autotune/src/tune_prompt/rolodex_v1/task_model/profile_schema.py``.
A failure here means a profile built today is not the same artifact as one that
was scored, and the comparison is invalid until somebody decides which side is
right.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rolodex_v1.profile_schema import (
    Biography,
    GroundedBiographyResponse,
    structured_output_schema,
)

TUNED_SCHEMA: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures/tuned_runs_grounded_schema.json").read_text(
        encoding="utf-8"
    )
)


def with_the_provider_clause_updated(value: Any) -> Any:
    """Apply the one intentional divergence to the tuned snapshot.

    Descriptions are prompt text -- pydantic emits a class docstring as the JSON
    Schema's ``description``, which every regime's model reads -- so they are
    compared, not stripped. Exactly one string may differ: the tuned runs said
    "requested from OpenAI", which is false under the agentic regime. Everything
    else the model is told must be identical, so the substitution is applied to
    theirs and the comparison is then exact.
    """
    if isinstance(value, dict):
        return {
            key: nested.replace("from OpenAI", "from the model")
            if key == "description" and isinstance(nested, str)
            else with_the_provider_clause_updated(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [with_the_provider_clause_updated(item) for item in value]
    return value


def test_the_contract_is_identical_to_the_tuned_runs() -> None:
    """Same fields, same types, same nullability, same required set.

    And the same descriptions: a reworded one is a reworded prompt.
    """
    mine = structured_output_schema(GroundedBiographyResponse)
    assert mine == with_the_provider_clause_updated(TUNED_SCHEMA)


def test_only_the_provider_clause_diverges_in_the_prompt_text() -> None:
    """A docstring is emitted as `description`, so it is text the model reads.

    The tuned runs said "requested from OpenAI", which is false for the agentic
    regime. Nothing else about what the model is told may differ, because
    everything else is instruction rather than provenance.
    """
    theirs = TUNED_SCHEMA["$defs"]["Biography"]["description"]
    mine = structured_output_schema(GroundedBiographyResponse)["$defs"]["Biography"][
        "description"
    ]
    assert theirs.replace("from OpenAI", "from the model") == mine


def test_the_response_object_carries_no_description() -> None:
    """The tuned runs sent none; adding one puts unseen text before the model."""
    assert "description" not in structured_output_schema(GroundedBiographyResponse)
    assert "description" not in TUNED_SCHEMA


def test_every_object_forbids_unknown_fields() -> None:
    """A key riding along unnoticed is a scoring difference nobody can see."""

    def objects(node: Any) -> list[dict]:
        found = []
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                found.append(node)
            for nested in node.values():
                found.extend(objects(nested))
        elif isinstance(node, list):
            for item in node:
                found.extend(objects(item))
        return found

    schema = structured_output_schema(GroundedBiographyResponse)
    found = objects(schema)
    assert found
    assert all(obj.get("additionalProperties") is False for obj in found)


def test_every_field_is_required() -> None:
    """An unsupported value must be stated, not left out and assumed."""
    schema = structured_output_schema(GroundedBiographyResponse)
    biography = schema["$defs"]["Biography"]
    assert set(biography["required"]) == set(biography["properties"])
    assert set(Biography.model_fields) == set(biography["properties"])
