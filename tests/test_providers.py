"""Each test names a property the provider split must hold.

The load-bearing one is the first: the OpenAI request is byte-for-byte what it
was before providers existed. The tuned Luna scores are the baseline every new
number is compared against, and a changed request means a profile built today is
not the artifact that was scored.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from rolodex_v1 import in_context, providers
from rolodex_v1.evidence_pack import build_pack
from rolodex_v1.profile_schema import (
    GroundedBiographyResponse,
    structured_output_schema,
)
from rolodex_v1.resolved_entities import Alias, ResolvedEntity

# Resolved from this file, not the working directory: AGENTS.md's rule is that a
# run started anywhere inside the checkout behaves the same, and these tests
# read the deployment as a file.
DEPLOYMENT = Path(__file__).resolve().parent.parent / "deploy" / "modal_qwen.py"

ENTITY = ResolvedEntity(
    entity_id="proto:participant_person:20",
    canonical_name="Josh Earnest",
    case="participant_person",
    canonical_type="person",
    mention_count=1,
    memory_count=1,
    aliases=(Alias(text="Josh Earnest", probability=1.0),),
)


def deployment_source() -> str:
    return DEPLOYMENT.read_text(encoding="utf-8")


def hf_overrides_literal() -> str:
    """The JSON string the deployment hands vLLM as one ``--hf-overrides`` argv.

    Reconstructed from the source rather than imported: modal is not a project
    dependency, so this module must never be imported.
    """
    deployment = deployment_source()
    start = deployment.index("HF_OVERRIDES = (\n") + len("HF_OVERRIDES = (\n")
    end = deployment.index("\n)\n", start)
    return "".join(
        line.strip().strip("'") for line in deployment[start:end].splitlines()
    )


def popen_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "Popen"
    ]


def pack(tmp_path: Path):
    """A one-document pack, enough to render a real request body."""
    directory = tmp_path / "corpus"
    directory.mkdir(exist_ok=True)
    (directory / "doc.txt").write_text(
        "Josh Earnest is the Press Secretary at The White House.",
        encoding="utf-8",
    )
    return build_pack(directory, ENTITY, budget_chars=90_000, max_doc_chars=4_000)


def test_the_openai_request_is_unchanged_by_the_provider_split(tmp_path: Path) -> None:
    """The tuned runs' comparability depends on this exact body."""
    body = in_context.chat_completion_body(pack(tmp_path), effort="high")
    assert body["reasoning_effort"] == "high"
    assert body["max_completion_tokens"] == in_context.MAX_COMPLETION_TOKENS
    assert "chat_template_kwargs" not in body
    assert "max_tokens" not in body
    assert body["response_format"]["json_schema"]["strict"] is True
    # The tier is set per attempt by request_completion, not baked in here.
    assert "service_tier" not in body


def test_qwen_omits_the_tier_and_nests_the_effort(tmp_path: Path) -> None:
    """The served vLLM rejects service_tier, and reads effort from the template."""
    qwen = providers.qwen_provider("https://example.modal.run/v1")
    body = in_context.chat_completion_body(
        pack(tmp_path), effort="xhigh", provider=qwen
    )
    assert "service_tier" not in body
    assert "reasoning_effort" not in body
    assert body["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
    assert body["max_tokens"] == in_context.MAX_COMPLETION_TOKENS
    assert "max_completion_tokens" not in body
    # Still the same strict schema: schema parity is not a provider's choice.
    assert body["response_format"]["json_schema"]["strict"] is True


def test_both_providers_send_the_identical_json_schema(tmp_path: Path) -> None:
    """A provider may vary the envelope around the schema, never the schema.

    ``strict is True`` on its own would pass while the body drifted, so compare
    the whole ``json_schema`` payload -- name, strict flag and schema -- between
    the two endpoints. Which server answered is not licence to emit a different
    artifact.
    """
    built = pack(tmp_path)
    qwen = providers.qwen_provider("https://example.modal.run/v1")
    openai_schema = in_context.chat_completion_body(built, effort="high")[
        "response_format"
    ]["json_schema"]
    qwen_schema = in_context.chat_completion_body(built, effort="xhigh", provider=qwen)[
        "response_format"
    ]["json_schema"]
    assert qwen_schema == openai_schema
    assert qwen_schema["schema"] == structured_output_schema(GroundedBiographyResponse)


def test_high_is_not_a_qwen_effort(tmp_path: Path) -> None:
    """Qwen's top of range is xhigh; silently sending high buys less thinking."""
    qwen = providers.qwen_provider("https://example.modal.run/v1")
    with pytest.raises(ValueError, match="effort must be one of"):
        in_context.chat_completion_body(pack(tmp_path), effort="high", provider=qwen)


def test_xhigh_is_not_an_openai_effort(tmp_path: Path) -> None:
    """And the reverse, so a mismatched pair cannot be mislabelled in a filename."""
    with pytest.raises(ValueError, match="effort must be one of"):
        in_context.chat_completion_body(pack(tmp_path), effort="xhigh")


def test_a_qwen_run_without_an_endpoint_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """There is no default server to guess at, and guessing wrong costs a run."""
    monkeypatch.delenv(providers.ENV_QWEN_BASE_URL, raising=False)
    with pytest.raises(RuntimeError, match="--qwen-base-url"):
        providers.provider_for("qwen3.8-27b")


def test_the_endpoint_comes_from_the_environment_when_the_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(providers.ENV_QWEN_BASE_URL, "https://env.example/v1")
    resolved = providers.provider_for("qwen3.8-27b")
    assert resolved.chat_completions_url == "https://env.example/v1/chat/completions"


def test_the_flag_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same precedence as every path in settings: what you typed wins."""
    monkeypatch.setenv(providers.ENV_QWEN_BASE_URL, "https://env.example/v1")
    resolved = providers.provider_for(
        "qwen3.8-27b", qwen_base_url="https://flag.example/v1"
    )
    assert resolved.chat_completions_url == "https://flag.example/v1/chat/completions"


def test_a_trailing_slash_does_not_double_up() -> None:
    resolved = providers.qwen_provider("https://example.modal.run/v1/")
    assert resolved.chat_completions_url == (
        "https://example.modal.run/v1/chat/completions"
    )


def test_an_openai_model_id_resolves_to_openai() -> None:
    assert providers.provider_for("gpt-5.6-luna") is providers.OPENAI


def test_the_client_ceiling_matches_what_the_deployment_serves() -> None:
    """Fitting against a ceiling the server does not have wastes the request.

    deploy/modal_qwen.py launches vLLM at QWEN_MAX_MODEL_LEN; the client budget
    is that less the completion reservation.
    """
    qwen = providers.qwen_provider("https://example.modal.run/v1")
    assert qwen.max_input_tokens == (
        providers.QWEN_MAX_MODEL_LEN - providers.MAX_COMPLETION_TOKENS
    )
    deployment = deployment_source()
    assert f"MAX_MODEL_LEN = {providers.QWEN_MAX_MODEL_LEN:_}" in deployment


def test_the_served_model_name_matches_what_a_run_would_send() -> None:
    """The server matches the request's `model` against --served-model-name."""
    deployment = deployment_source()
    assert 'SERVED_MODEL_NAME = "qwen3.8-27b"' in deployment
    assert providers.provider_for("qwen3.8-27b", qwen_base_url="https://x/v1")


def test_the_deployment_json_override_is_valid_json() -> None:
    """A malformed override fails 40 minutes into a cold start, not at deploy."""
    overrides = json.loads(hf_overrides_literal())
    rope = overrides["text_config"]["rope_parameters"]
    assert rope["rope_type"] == "yarn"
    # factor x native window must cover what the server is launched at.
    assert (
        rope["factor"] * rope["original_max_position_embeddings"]
        >= providers.QWEN_MAX_MODEL_LEN
    )


def test_the_deployment_launches_vllm_without_a_shell() -> None:
    """The override is one argv element or vLLM never starts.

    This cost a 45-minute cold start in the predecessor's deployment, which
    built the command with ``Popen(" ".join(cmd), shell=True)``. --hf-overrides
    carries JSON containing spaces, so the shell word-split it and stripped its
    quotes, and vLLM exited immediately with "unrecognized arguments". Modal
    then waited out its startup_timeout on a port that was never going to open,
    which reads like a slow boot rather than a crash.
    """
    tree = ast.parse(deployment_source())

    calls = popen_calls(tree)
    assert len(calls) == 1
    (launch,) = calls

    # No shell to word-split anything, on this call or any other.
    assert not any(
        keyword.arg == "shell" for call in calls for keyword in call.keywords
    )

    # The argv is a name bound to a list literal -- not a join, an f-string or a
    # bare string, each of which would hand execve one word-splittable blob.
    argv = launch.args[0]
    assert isinstance(argv, ast.Name)
    bindings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == argv.id
            for target in node.targets
        )
    ]
    assert len(bindings) == 1
    assert isinstance(bindings[0], ast.List)
    # The overrides go in as their own element, so execve sees them whole.
    assert any(
        isinstance(element, ast.Name) and element.id == "HF_OVERRIDES"
        for element in bindings[0].elts
    )

    # And the echoed line must be re-runnable rather than a lie about what ran:
    # shlex.join quotes the overrides argument, " ".join would not.
    # Narrowed in the loop rather than a comprehension: the isinstance inside a
    # comprehension's condition does not reach the element expression, so the
    # collected node stays an `expr` and `.value` is not known to exist.
    joins = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "join":
            continue
        joins += 1
        assert isinstance(node.func.value, ast.Name)
        assert node.func.value.id == "shlex"
    assert joins


def test_the_override_survives_being_passed_as_one_argument() -> None:
    """What vLLM would parse back out of the argv element the deployment sends."""
    literal = hf_overrides_literal()
    # Whole, it is JSON. Split on whitespace the way a shell would, it is not.
    assert json.loads(literal)
    with pytest.raises(json.JSONDecodeError):
        json.loads(literal.split(" ")[0])


def test_a_provider_defaults_to_its_own_effort_not_another_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect: a Qwen run with no --effort inherited OpenAI's "high".

    Qwen rejects it, so the run exited 2 before a pack was built, and
    QWEN_DEFAULT_EFFORT was a field nothing ever read.
    """
    monkeypatch.delenv(providers.ENV_QWEN_BASE_URL, raising=False)
    qwen = providers.qwen_provider("https://example.invalid/v1")
    assert qwen.resolve_effort(None) == providers.QWEN_DEFAULT_EFFORT
    assert providers.OPENAI.resolve_effort(None) == "high"


def test_resolving_an_effort_still_validates_one_that_was_asked_for() -> None:
    """Defaulting must not become a way past the vocabulary check (trap 8)."""
    qwen = providers.qwen_provider("https://example.invalid/v1")
    assert qwen.resolve_effort("xhigh") == "xhigh"
    with pytest.raises(ValueError, match="effort must be one of"):
        qwen.resolve_effort("high")


def test_the_wire_shape_belongs_to_the_provider() -> None:
    """OpenAI reads a top-level field; Qwen's dial lives in the chat template."""
    qwen = providers.qwen_provider("https://example.invalid/v1")
    assert providers.OPENAI.reasoning_effort_fields("low") == {
        "reasoning_effort": "low"
    }
    assert qwen.reasoning_effort_fields("low") == {
        "chat_template_kwargs": {"reasoning_effort": "low"}
    }
