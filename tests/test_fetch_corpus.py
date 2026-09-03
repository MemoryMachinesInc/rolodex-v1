"""Properties the corpus fetch must hold.

Every test here runs against an injected transport. Nothing reaches the network,
and nothing reads a credential: a suite that needed either would stop being run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rolodex_v1 import credentials, fetch_corpus, settings
from rolodex_v1.fetch_corpus import (
    convert_dump,
    fetch_source_docs,
    main,
    read_refresh_token,
    safe_name,
)
from rolodex_v1.memorymachines import (
    MAX_TOP_K,
    FetchError,
    FirebaseAuth,
    Response,
    document_text,
    fetch_bundle,
    list_items,
)


class Token:
    """A `Bearer` that never mints: the files routes under test need a header."""

    def bearer(self) -> str:
        return "bearer-token"


TOKEN = Token()


def responder(routes: dict[str, object], record: list[str] | None = None):
    """A transport answering by URL prefix, with 200 for anything matched."""

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        if record is not None:
            record.append(url)
        for prefix, payload in routes.items():
            if prefix in url:
                if isinstance(payload, Response):
                    return payload
                return Response(200, json.dumps(payload).encode())
        return Response(404, b'{"detail": "no route"}')

    return transport


def document(item_id: str, content: str) -> dict:
    return {
        "item_id": item_id,
        "source_type": "email",
        "content": content,
        "user_id": "u1",
    }


def test_a_document_becomes_its_content_and_nothing_else() -> None:
    """The pipeline reads text; item_id and user_id are addressing, not evidence."""
    assert document_text(document("a", "Josh Earnest wrote.")) == "Josh Earnest wrote."


@pytest.mark.parametrize(
    "payload",
    [{}, {"content": None}, {"content": ""}, {"content": "   "}, {"content": 3}],
)
def test_a_payload_with_no_usable_content_is_refused(payload: dict) -> None:
    """An empty .txt is evidence that exists, matches nothing, and hides a person."""
    with pytest.raises(FetchError, match="content"):
        document_text(payload)


def test_the_error_names_the_keys_the_payload_actually_had() -> None:
    """The next person needs to know what the shape changed *to*."""
    with pytest.raises(FetchError, match="body"):
        document_text({"item_id": "a", "body": "text under a new key"})


def test_a_truncated_bundle_is_refused_rather_than_written(tmp_path: Path) -> None:
    """A corpus missing people reports success at every later count."""
    transport = responder(
        {"/v1/entities/resolved": {"entities": [], "count": 10, "total_in_bundle": 99}}
    )
    with pytest.raises(FetchError, match="truncated"):
        fetch_bundle("https://api", "key", transport=transport)


@pytest.mark.parametrize("count,total", [("10", "99"), (10, "99"), (True, 1)])
def test_counts_that_cannot_be_compared_are_not_a_bundle(count, total) -> None:
    """A truncation check that shrugs at a count it cannot compare is no check."""
    transport = responder(
        {
            "/v1/entities/resolved": {
                "entities": [],
                "count": count,
                "total_in_bundle": total,
            }
        }
    )
    with pytest.raises(FetchError, match="not a resolved-entities bundle"):
        fetch_bundle("https://api", "key", transport=transport)


def test_a_bundle_keeps_only_the_three_keys_on_disk() -> None:
    """A fetched corpus and a hand-delivered one must be the same artifact."""
    transport = responder(
        {
            "/v1/entities/resolved": {
                "entities": [{"resolved_entity_id": "e1"}],
                "count": 1,
                "total_in_bundle": 1,
                "debug_timing_ms": 41,
            }
        }
    )
    bundle = fetch_bundle("https://api", "key", transport=transport)
    assert sorted(bundle) == ["count", "entities", "total_in_bundle"]


def test_the_bundle_request_asks_for_the_whole_thing_with_aliases() -> None:
    """An alias-less bundle gives retrieval nothing to match, so every pack is empty."""
    seen: list[str] = []
    transport = responder(
        {"/v1/entities/resolved": {"entities": [], "count": 0, "total_in_bundle": 0}},
        seen,
    )
    fetch_bundle("https://api", "key", transport=transport)
    assert f"top_k={MAX_TOP_K}" in seen[0]
    assert "include_aliases=true" in seen[0]


def test_listing_follows_has_more_rather_than_the_page_size() -> None:
    """A full page is not the same statement as "there is another page"."""
    pages = [
        {"item_ids": ["a", "b"], "has_more": True},
        {"item_ids": ["c"], "has_more": False},
    ]
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        calls.append(url)
        return Response(200, json.dumps(pages[len(calls) - 1]).encode())

    got = list(
        list_items("https://api", TOKEN, "email", transport=transport, page_limit=2)
    )
    assert got == ["a", "b", "c"]
    assert "offset=2" in calls[1]


def test_a_settled_rejection_is_not_retried() -> None:
    """Retrying a 403 three times only delays the message that says what to fix."""
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        calls.append(url)
        return Response(403, b'{"detail": "forbidden"}')

    with pytest.raises(FetchError, match="Firebase"):
        list(list_items("https://api", TOKEN, "email", transport=transport))
    assert len(calls) == 1


def test_a_files_route_403_says_which_credential_it_wanted() -> None:
    """The two halves take different credentials; the 403 must not be a riddle."""
    transport = responder({"/v1/files/list": Response(403, b"{}")})
    with pytest.raises(FetchError, match="Firebase token"):
        list(list_items("https://api", TOKEN, "email", transport=transport))


def test_documents_land_as_text_under_their_source_type(tmp_path: Path) -> None:
    """`.txt` at any depth is what the pack builder reads; JSON is invisible to it."""
    transport = responder(
        {
            "/v1/files/list": {"item_ids": ["m1"], "has_more": False},
            "/v1/files/download": document("m1", "Josh Earnest spoke."),
        }
    )
    written, skipped = fetch_source_docs(
        "https://api", TOKEN, tmp_path, sources=("email",), transport=transport
    )
    assert (written, skipped) == (1, 0)
    assert (tmp_path / "email" / "m1.txt").read_text() == "Josh Earnest spoke."


def test_an_already_downloaded_document_is_not_fetched_again(tmp_path: Path) -> None:
    """An interrupted fetch resumes; re-downloading thousands of files is the bug."""
    seen: list[str] = []
    transport = responder(
        {
            "/v1/files/list": {"item_ids": ["m1"], "has_more": False},
            "/v1/files/download": document("m1", "text"),
        },
        seen,
    )
    (tmp_path / "email").mkdir()
    (tmp_path / "email" / "m1.txt").write_text("already here")

    written, _ = fetch_source_docs(
        "https://api", TOKEN, tmp_path, sources=("email",), transport=transport
    )
    assert written == 0
    assert not any("download" in url for url in seen)
    assert (tmp_path / "email" / "m1.txt").read_text() == "already here"


def test_an_item_id_cannot_write_outside_the_corpus(tmp_path: Path) -> None:
    """Ids arrive from an API, so they are sanitized rather than trusted."""
    assert safe_name("../../etc/passwd") == ".._.._etc_passwd"
    transport = responder(
        {
            "/v1/files/list": {"item_ids": ["../escape"], "has_more": False},
            "/v1/files/download": document("../escape", "text"),
        }
    )
    fetch_source_docs(
        "https://api", TOKEN, tmp_path, sources=("email",), transport=transport
    )
    assert [path.name for path in (tmp_path / "email").iterdir()] == [".._escape.txt"]


def test_one_unreadable_document_does_not_strand_the_rest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A changed record in one source must not cost the thousands beside it."""
    dump = tmp_path / "dump"
    (dump / "email").mkdir(parents=True)
    (dump / "email" / "good.json").write_text(
        json.dumps(document("good", "Ada wrote."))
    )
    (dump / "email" / "shape.json").write_text(
        json.dumps({"item_id": "x", "body": "?"})
    )
    (dump / "email" / "broken.json").write_text("{not json")
    (dump / "email" / "list.json").write_text("[1, 2]")

    with caplog.at_level("WARNING"):
        written, skipped = convert_dump(dump, tmp_path / "corpus")

    assert (written, skipped) == (1, 3)
    assert (tmp_path / "corpus" / "email" / "good.txt").read_text() == "Ada wrote."
    # The message says what the payload was, not that a dict had no keys.
    assert "got JSON list" in caplog.text


def test_converting_a_dump_leaves_the_dump_alone(tmp_path: Path) -> None:
    """It is the only copy of what the API actually answered."""
    dump = tmp_path / "dump"
    (dump / "email").mkdir(parents=True)
    source = dump / "email" / "a.json"
    source.write_text(json.dumps(document("a", "Ada wrote.")))

    convert_dump(dump, tmp_path / "corpus")

    assert json.loads(source.read_text())["content"] == "Ada wrote."


def test_an_id_token_is_minted_once_and_reused() -> None:
    """Re-minting per request would spend a token exchange on every document."""
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        calls.append(url)
        return Response(200, json.dumps({"id_token": "abc", "user_id": "u"}).encode())

    auth = FirebaseAuth("refresh", "firebase-key", transport=transport, now=lambda: 0.0)
    assert auth.bearer() == "abc"
    assert auth.bearer() == "abc"
    assert len(calls) == 1


def test_an_expired_id_token_is_re_minted_mid_run() -> None:
    """A long download outlives an ID token, and the expiry must not read as a 401."""
    # mint, the expiry check, then the re-mint.
    clock = iter([0.0, 10_000.0, 10_000.0])
    calls: list[str] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        calls.append(url)
        return Response(200, json.dumps({"id_token": f"t{len(calls)}"}).encode())

    auth = FirebaseAuth("r", "k", transport=transport, now=lambda: next(clock))
    assert auth.bearer() == "t1"
    assert auth.bearer() == "t2"


def test_a_dead_refresh_token_says_what_to_do() -> None:
    """The 400 here reads like a server fault and is a finished credential."""
    transport = responder({"securetoken": Response(400, b'{"error": "TOKEN_EXPIRED"}')})
    auth = FirebaseAuth("r", "k", transport=transport, now=lambda: 0.0)
    with pytest.raises(FetchError, match="Sign in"):
        auth.bearer()


def test_a_config_without_fetch_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """Defaulting an environment asks a production API on the strength of nothing."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[paths]\ndata = "corpus"\n')
    assert main(["--config", str(config), "--dry-run"]) == 1


def test_a_fetch_dry_run_writes_nothing_and_names_both_credentials(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The cheap check before a machine pulls thousands of documents."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text(
        '[paths]\ndata = "corpus"\n[fetch]\nenvironment = "staging"\n'
        'sources = ["email"]\n'
    )
    with caplog.at_level("INFO"):
        assert main(["--config", str(config), "--dry-run"]) == 0
    logged = caplog.text
    assert "MM_API_KEY" in logged and "MM_REFRESH_TOKEN" in logged
    assert "api-staging.engramme.com" in logged
    assert not (tmp_path / "corpus").exists()


def test_an_unknown_fetch_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled `sources` would quietly fetch all thirty source types."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[fetch]\nsourcs = ["email"]\n')
    with pytest.raises(ValueError, match="unknown \\[fetch\\] key"):
        settings.read_fetch(config)


def test_an_unknown_environment_is_rejected(tmp_path: Path) -> None:
    """There is no Firebase key for an environment nobody deployed."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[fetch]\nenvironment = "production"\n')
    with pytest.raises(ValueError, match="prod, staging, dev"):
        settings.read_fetch(config)


def test_a_fetch_table_must_name_its_environment(tmp_path: Path) -> None:
    """A table that says only `sources` must not be aimed at production for it."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[fetch]\nsources = ["email"]\n')
    with pytest.raises(ValueError, match="environment is required"):
        settings.read_fetch(config)


def test_a_secret_cannot_be_put_in_the_config(tmp_path: Path) -> None:
    """A key in a config is a key in every copy of that config."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[fetch]\nenvironment = "dev"\napi_key = "sk-..."\n')
    with pytest.raises(ValueError, match="unknown \\[fetch\\] key"):
        settings.read_fetch(config)


def test_a_config_with_no_fetch_table_is_not_configured(tmp_path: Path) -> None:
    """Absent from the config means absent as a capability, not defaulted."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[paths]\ndata = "corpus"\n')
    assert settings.read_fetch(config) is None


def test_an_existing_bundle_is_left_alone_without_force(tmp_path: Path) -> None:
    """It is what every checkpointed entity id means; the skip needs no key."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[paths]\ndata = "corpus"\n[fetch]\nenvironment = "staging"\n')
    bundle = tmp_path / "corpus" / "resolved_entities_bundle.json"
    bundle.parent.mkdir()
    bundle.write_text("{}\n")

    assert main(["--config", str(config), "--only", "bundle"]) == 0
    assert bundle.read_text() == "{}\n"


def test_skipped_documents_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus quietly missing documents is what the exit code is for."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[paths]\ndata = "corpus"\n[fetch]\nenvironment = "staging"\n')
    monkeypatch.setattr(fetch_corpus, "read_refresh_token", lambda name: "token")
    monkeypatch.setattr(
        fetch_corpus, "fetch_source_docs", lambda *args, **kwargs: (5, 2)
    )
    assert main(["--config", str(config), "--only", "source-docs"]) == 1


def test_the_environment_beats_the_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The keychain is the only source that can stop and ask a human something."""
    monkeypatch.setenv("MM_REFRESH_TOKEN_TEST", "from-env")

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("the keychain was consulted with a token in hand")

    monkeypatch.setattr(fetch_corpus.subprocess, "run", explode)
    assert read_refresh_token("MM_REFRESH_TOKEN_TEST") == "from-env"


def test_a_silent_keychain_fails_fast_with_the_way_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hang here is indistinguishable from a slow download -- that is trap 9."""
    monkeypatch.delenv("MM_REFRESH_TOKEN_TEST", raising=False)
    monkeypatch.setattr(credentials, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(fetch_corpus, "TOKEN_FILE", tmp_path / "absent.txt")

    def timeout(*args, **kwargs):
        raise fetch_corpus.subprocess.TimeoutExpired(cmd="security", timeout=20)

    monkeypatch.setattr(fetch_corpus.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="consent dialog"):
        read_refresh_token("MM_REFRESH_TOKEN_TEST")


def test_the_token_file_is_read_before_the_keychain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dumper's own fallback, for a machine that cannot show a dialog."""
    monkeypatch.delenv("MM_REFRESH_TOKEN_TEST", raising=False)
    monkeypatch.setattr(credentials, "repo_root", lambda: tmp_path)
    token_file = tmp_path / "token.txt"
    token_file.write_text("from-file\n")
    monkeypatch.setattr(fetch_corpus, "TOKEN_FILE", token_file)
    assert read_refresh_token("MM_REFRESH_TOKEN_TEST") == "from-file"
