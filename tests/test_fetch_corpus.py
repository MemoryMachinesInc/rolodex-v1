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
    """A `Bearer` that mints nothing, and counts what it was asked for.

    The token exchange belongs to `memorome_takeout.firebase_token` and is
    tested there. What this repo has to hold is how its routes *use* a bearer:
    which requests carry one, and whether a rejection is worth a new one.
    """

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.tokens = list(tokens or ["bearer-token"])
        self.refreshes = 0

    def bearer(self) -> str:
        return self.tokens[0]

    def refresh(self) -> bool:
        self.refreshes += 1
        if len(self.tokens) == 1:
            # Nothing left to mint from -- what the provider answers for a
            # credential that cannot be renewed, and the retry must respect it.
            return False
        self.tokens.pop(0)
        return True


def rejecting(status: int, record: list[dict[str, str]]):
    """A transport that refuses everything, keeping the headers it was sent.

    The headers are what the rejection tests are actually about: whether the
    call came back with a different bearer than the one that was refused.
    """

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        record.append(headers)
        return Response(status, b'{"detail": "refused"}')

    return transport


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
        fetch_bundle("https://api", Token(), transport=transport)


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
        fetch_bundle("https://api", Token(), transport=transport)


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
    bundle = fetch_bundle("https://api", Token(), transport=transport)
    assert sorted(bundle) == ["count", "entities", "total_in_bundle"]


def test_the_bundle_request_asks_for_the_whole_thing_with_aliases() -> None:
    """An alias-less bundle gives retrieval nothing to match, so every pack is empty."""
    seen: list[str] = []
    transport = responder(
        {"/v1/entities/resolved": {"entities": [], "count": 0, "total_in_bundle": 0}},
        seen,
    )
    fetch_bundle("https://api", Token(), transport=transport)
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
        list_items("https://api", Token(), "email", transport=transport, page_limit=2)
    )
    assert got == ["a", "b", "c"]
    assert "offset=2" in calls[1]


def test_a_403_is_never_retried_and_never_re_minted() -> None:
    """A fresh token is the same account, so re-minting only repeats the refusal."""
    sent: list[dict[str, str]] = []
    auth = Token(["first", "second"])

    with pytest.raises(FetchError, match="allowlisted"):
        list(list_items("https://api", auth, "email", transport=rejecting(403, sent)))
    assert len(sent) == 1
    assert auth.refreshes == 0


def test_a_403_says_the_account_is_refused_rather_than_the_token() -> None:
    """Both routes take the same bearer now; a 403 is an allowlist, not a key."""
    transport = responder({"/v1/files/list": Response(403, b"{}")})
    with pytest.raises(FetchError, match="allowlisted"):
        list(list_items("https://api", Token(), "email", transport=transport))


def test_a_401_is_re_minted_once_and_the_new_token_is_the_one_retried() -> None:
    """A run that outlives its ID token must not die mid-corpus over it."""
    auth = Token(["stale", "fresh"])
    seen: list[str] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        seen.append(headers["Authorization"])
        if headers["Authorization"] == "Bearer stale":
            return Response(401, b'{"detail": "expired"}')
        return Response(
            200, json.dumps({"item_ids": ["m1"], "has_more": False}).encode()
        )

    got = list(list_items("https://api", auth, "email", transport=transport))
    assert got == ["m1"]
    assert seen == ["Bearer stale", "Bearer fresh"]
    assert auth.refreshes == 1


def test_a_401_that_survives_a_re_mint_is_not_retried_again() -> None:
    """A token the API keeps refusing must fail with the API's 401, not three."""
    auth = Token(["stale", "fresh"])
    sent: list[dict[str, str]] = []

    with pytest.raises(FetchError, match="401"):
        list(list_items("https://api", auth, "email", transport=rejecting(401, sent)))
    assert [headers["Authorization"] for headers in sent] == [
        "Bearer stale",
        "Bearer fresh",
    ]
    assert auth.refreshes == 1


def test_nothing_to_mint_from_lets_the_original_401_stand() -> None:
    """`refresh()` answering False means asking again is pointless, not urgent."""
    auth = Token(["only-one"])
    sent: list[dict[str, str]] = []

    with pytest.raises(FetchError, match="401"):
        list(list_items("https://api", auth, "email", transport=rejecting(401, sent)))
    assert len(sent) == 1
    assert auth.refreshes == 1


def test_documents_land_as_text_under_their_source_type(tmp_path: Path) -> None:
    """`.txt` at any depth is what the pack builder reads; JSON is invisible to it."""
    transport = responder(
        {
            "/v1/files/list": {"item_ids": ["m1"], "has_more": False},
            "/v1/files/download": document("m1", "Josh Earnest spoke."),
        }
    )
    written, skipped = fetch_source_docs(
        "https://api", Token(), tmp_path, sources=("email",), transport=transport
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
        "https://api", Token(), tmp_path, sources=("email",), transport=transport
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
        "https://api", Token(), tmp_path, sources=("email",), transport=transport
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


def test_the_bundle_route_takes_the_bearer_and_never_an_api_key() -> None:
    """The platform rejects mixed credentials, so it is one or the other."""
    seen: list[dict[str, str]] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        seen.append(headers)
        payload = {"entities": [], "count": 0, "total_in_bundle": 0}
        return Response(200, json.dumps(payload).encode())

    fetch_bundle("https://api", Token(["t"]), transport=transport)
    assert seen[0]["Authorization"] == "Bearer t"
    assert "x-api-key" not in {name.lower() for name in seen[0]}


def test_the_files_routes_carry_the_bearer_too() -> None:
    """One credential, both halves -- the whole point of the change."""
    seen: list[dict[str, str]] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        seen.append(headers)
        page = {"item_ids": [], "has_more": False}
        return Response(200, json.dumps(page).encode())

    list(list_items("https://api", Token(["t"]), "email", transport=transport))
    assert seen[0]["Authorization"] == "Bearer t"
    assert "x-api-key" not in {name.lower() for name in seen[0]}


def test_the_token_exchange_goes_through_the_injected_transport() -> None:
    """The seam that keeps minting off the network in this repo's tests.

    `memorome_takeout.firebase_token` does the exchange and owns its own tests
    for it; what is this repo's to prove is that `_token_transport` hands the
    provider *this* transport, so a test that mints reaches the fake rather
    than securetoken.googleapis.com.
    """
    seen: list[tuple[str, dict[str, str], bytes | None]] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        seen.append((url, headers, data))
        body = {"id_token": "minted", "expires_in": "3600"}
        return Response(200, json.dumps(body).encode())

    auth = FirebaseAuth(
        refresh_token="a-refresh-token",  # noqa: S106 - a fixture, not a token
        environment="staging",
        refresh_token_env="MM_REFRESH_TOKEN",  # noqa: S106 - a variable name
        transport=transport,
    )
    assert auth.bearer() == "minted"

    url, headers, data = seen[0]
    assert "securetoken.googleapis.com" in url
    assert headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert data is not None and b"grant_type=refresh_token" in data


def test_a_fetch_says_which_account_it_runs_as_and_never_the_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corpus pulled as the wrong account is the mistake worth one log line.

    The provenance comes from the provider -- an account plus the variable the
    credential was attributed to, which is what `refresh_token_env` is passed
    for. The token itself must never reach a log, here or anywhere.
    """

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        return Response(200, json.dumps({"id_token": "s3cret-token"}).encode())

    auth = FirebaseAuth(
        refresh_token="a-refresh-token",  # noqa: S106 - a fixture, not a token
        environment="staging",
        refresh_token_env="MM_REFRESH_TOKEN",  # noqa: S106 - a variable name
        transport=transport,
    )
    with caplog.at_level("INFO"):
        auth.bearer()
        auth.bearer()

    assert caplog.text.count("fetching as") == 1
    assert "MM_REFRESH_TOKEN" in caplog.text
    assert "s3cret-token" not in caplog.text


def test_the_supplied_refresh_token_is_used_instead_of_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This repo's own discovery stays in charge; the provider does not search.

    `read_refresh_token` looks in four places the provider has never heard of,
    so a stray ENGRAMME_REFRESH_TOKEN on the machine must not quietly become the
    credential a fetch runs as.
    """
    monkeypatch.setenv("ENGRAMME_REFRESH_TOKEN", "somebody-elses-token")
    seen: list[bytes | None] = []

    def transport(url: str, headers: dict[str, str], data: bytes | None = None):
        seen.append(data)
        return Response(200, json.dumps({"id_token": "minted"}).encode())

    auth = FirebaseAuth(
        refresh_token="the-one-we-found",  # noqa: S106 - a fixture, not a token
        environment="staging",
        transport=transport,
    )
    auth.bearer()
    assert seen[0] is not None and b"the-one-we-found" in seen[0]


def test_a_config_without_fetch_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """Defaulting an environment asks a production API on the strength of nothing."""
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[paths]\ndata = "corpus"\n')
    assert main(["--config", str(config), "--dry-run"]) == 1


def test_a_fetch_dry_run_writes_nothing_and_names_the_one_credential(
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
    assert "MM_REFRESH_TOKEN" in logged
    # There is no second credential to name any more, and naming one would send
    # somebody looking for a key the fetch no longer sends.
    assert "MM_API_KEY" not in logged
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


def test_a_vestigial_api_key_env_is_accepted_and_ignored(tmp_path: Path) -> None:
    """The configs that still name it are gitignored, so this repo cannot fix them.

    Rejecting the key would turn a credential change made here into a broken
    fetch on a laptop nobody remembers to update; reading it would send an
    x-api-key the platform now refuses alongside a bearer.
    """
    config = tmp_path / "rolodex-v1.toml"
    config.write_text('[fetch]\nenvironment = "dev"\napi_key_env = "MM_API_KEY"\n')
    fetch = settings.read_fetch(config)
    assert fetch is not None
    assert not hasattr(fetch, "api_key_env")


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
