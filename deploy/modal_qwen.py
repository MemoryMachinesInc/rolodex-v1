r"""Serve Qwen3.8-27B on Modal behind an OpenAI-compatible endpoint.

This is deliberately the *only* thing this repo knows about hosting a model. It
exposes vLLM's own OpenAI server rather than a bespoke handler, so
:mod:`rolodex_v1.in_context` needs no branch for it beyond the four dialect
differences recorded in :mod:`rolodex_v1.providers` -- it is the same request,
sent somewhere else.

It is not imported by the package and Modal is not a project dependency: this
file is run by the ``modal`` CLI, which brings its own environment. A machine
that never serves a local model never installs it.

**Context sizing is why this needs large GPUs, not the 27B of weights.**
Qwen3.8's native window is 262,144 tokens. An evidence pack for a
densely-mentioned entity exceeds that, so the YaRN rope override is applied to
reach ~1M and the server is launched at 960,000. At that length the KV cache,
not the parameters, decides the GPU count; FP8 weights and an FP8 KV cache
roughly halve both. Keep ``MAX_MODEL_LEN`` and
``rolodex_v1.providers.QWEN_MAX_MODEL_LEN`` in step, or the client will fit
requests against a ceiling the server does not have.

Deploy, then point a run at it::

    modal secret create qwen-serving-token QWEN_API_KEY=<a secret you choose>
    modal deploy deploy/modal_qwen.py

    export ROLODEX_V1_QWEN_BASE_URL=https://<workspace>--rolodex-qwen-serve.modal.run/v1
    export QWEN_API_KEY=<the same secret>
    uv run python -m rolodex_v1.build_profiles \\
        --regime in-context --model qwen3.8-27b --effort xhigh --limit 5

The first request pays a cold start: ~30GB of weights, then a very large KV
cache. ``scaledown_window`` keeps the container up between entities so a batch
pays that once rather than per profile.
"""

from __future__ import annotations

import os
import shlex
import subprocess

import modal

MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
SERVED_MODEL_NAME = "qwen3.8-27b"

# Must match rolodex_v1.providers.QWEN_MAX_MODEL_LEN.
MAX_MODEL_LEN = 960_000

# YaRN, as the vLLM recipe for this checkpoint specifies: nested under
# `text_config`, scaling the 262,144-token native window by 4.
HF_OVERRIDES = (
    '{"text_config": {"max_position_embeddings": 1010000, '
    '"rope_parameters": {"rope_type": "yarn", "factor": 4.0, '
    '"original_max_position_embeddings": 262144, "rope_theta": 10000000, '
    '"partial_rotary_factor": 0.25, "mrope_interleaved": true, '
    '"mrope_section": [11, 11, 10]}}}'
)

GPU_CONFIG = "H200:4"
N_GPU = 4
VLLM_PORT = 8000
MINUTES = 60

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.34.4")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            # vLLM refuses a max-model-len past the checkpoint's declared
            # maximum without this, which the YaRN override makes legitimate.
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        }
    )
)

# Weights survive a redeploy, so a restart is not another 30GB download.
hf_cache = modal.Volume.from_name("rolodex-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rolodex-vllm-cache", create_if_missing=True)

# The bearer token the client sends. Same name as the provider's api_key_name,
# so one secret serves both ends.
auth_token = modal.Secret.from_name("qwen-serving-token")

app = modal.App("rolodex-qwen")


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[auth_token],
    timeout=24 * 60 * MINUTES,
    scaledown_window=20 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=VLLM_PORT, startup_timeout=45 * MINUTES)
def serve() -> None:
    """Launch vLLM's OpenAI-compatible server and hand Modal the port."""
    command = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",  # noqa: S104 -- inside the container, Modal fronts it
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(N_GPU),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--hf-overrides",
        HF_OVERRIDES,
        # Halves KV-cache bytes per token, which is what gates the context
        # length at this window.
        "--kv-cache-dtype",
        "fp8",
        "--gpu-memory-utilization",
        "0.92",
        # A handful of ~800K-token prefills at once exhausts the cache; the
        # profiles are built one entity at a time anyway.
        "--max-num-seqs",
        "4",
        "--enable-chunked-prefill",
        "--api-key",
        os.environ["QWEN_API_KEY"],
    ]
    # shlex.join, not " ".join: the overrides argument contains spaces, so a
    # naively joined line is not the command that ran and would break if it
    # were pasted back into a shell -- which is exactly the bug that stopped
    # the predecessor's deployment from ever serving.
    print("launching:", shlex.join(command), flush=True)
    subprocess.Popen(command)  # noqa: S603 -- fixed argv, no shell
