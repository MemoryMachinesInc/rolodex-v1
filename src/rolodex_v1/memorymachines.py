"""Fetch a corpus from the memorymachines API: the bundle, and the documents.

Reverse-engineered from `fetch_resolved_entities_bundle.sh` and
`dump_all_source_docs.sh`, which remain the working record of what these
endpoints accept -- with one exception the scripts have not caught up to. The
bundle script sends a master ``x-api-key``, and that is history: **both halves
now take the same credential.** A Firebase refresh token is exchanged for an ID
token that lives about an hour, and that token rides on every request here as
``Authorization: Bearer``. API-key auth is being retired, so one refresh token
fetches a whole corpus, and ``--only bundle`` / ``--only source-docs`` are now
about which half you want rather than which of two credentials this machine
happens to hold.

**Never send both.** The platform rejects a request carrying an ``x-api-key``
and a bearer together, so no ``x-api-key`` header is built anywhere in this
module -- not as a fallback, not "just in case the route is old".

**The minting is not ours.** ``memorome_takeout.firebase_token`` owns the token
exchange, the expiry maths, the dead-credential classification and the account
provenance; `FirebaseAuth` below is the adapter that wires it to this module's
`Transport` and this module's `Bearer` protocol, and nothing else. That package
is a sibling checkout installed editable (see `pyproject.toml` and AGENTS.md);
the module imported from it is stdlib-only, so depending on it costs this
project no third-party dependency. What stays here is what is *this* repo's:
where the credential is found (`fetch_corpus.read_refresh_token`), which base
URL each environment answers on, and the retry policy the routes need.

An ID token is short-lived, so it is replaced twice over: on the deadline the
exchange itself reported, which is the provider's business, and -- when that
deadline or this machine's clock was wrong -- on the first 401 the API answers
with, which is `_get`'s. The first stops a thousand-document run from
rediscovering the expiry as a failed download and filing it as a missing
document; the second stops a wrong deadline from being fatal in the middle of
one.

Nothing here logs a credential, and callers must not either. The error messages
name the *variable*, never the value.

The transport is injected (``Transport``) so the retry policy, the pagination
and the JSON-to-text conversion are testable without a network or a token --
and, adapted through `_token_transport`, so is the token exchange behind it.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from memorome_takeout.firebase_token import IdTokenProvider, TokenTransport

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Environment:
    """One deployment of the API a corpus is fetched from.

    Only the base URL, now. The Firebase Web API key that used to sit beside it
    lives in ``memorome_takeout.firebase_token.FIREBASE_WEB_API_KEYS``, and the
    provider looks it up from the environment name itself -- a second copy here
    could only ever drift from the one doing the minting, and a key from the
    wrong Firebase project fails as a 400 that reads like a revoked credential.

    The names below must therefore be the names that module knows (`prod`,
    `staging`, `dev`); anything else raises `TokenError` at the first mint.
    """

    base_url: str


#: The base URLs are this repo's own: the corpus fetch talks to the public API
#: host, which is not the gateway URL the memorome takeout scripts use for the
#: same environment names.
ENVIRONMENTS = {
    "prod": Environment("https://api.engramme.com"),
    "staging": Environment("https://api-staging.engramme.com"),
    "dev": Environment("https://api-dev.engramme.com"),
}

#: VALID_SOURCE_TYPES from the API's app_platform.py, plus the legacy ``drive``
#: prefix its source probe still reads. Asking for a type the user has nothing
#: under costs one list request and returns zero items, so the default is the
#: whole vocabulary rather than a guess at which ones matter.
DEFAULT_SOURCES = (
    "text", "email", "pdf", "stream", "browser", "vscode", "calendar", "github",
    "slack", "asana", "claude_code", "cursor", "codex", "google_meets",
    "technical_docs", "weekly_updates", "gdocs", "tasks", "contacts", "youtube",
    "photos", "books", "fit", "ms-outlook", "ms-calendar", "ms-teams",
    "ms-onedrive", "ms-sharepoint", "plaud", "imessage", "whatsapp", "drive",
)  # fmt: skip

#: The server's own ceiling (MAX_RESOLVED_ENTITIES_TOP_K). Asking for the cap is
#: what makes the bundle whole -- a smaller number silently truncates it, and a
#: truncated bundle is a corpus missing people rather than a smaller one.
MAX_TOP_K = 50_000

#: The server's page ceiling for /v1/files/list.
PAGE_LIMIT = 10_000

#: Statuses that mean "ask again": the server is busy, or briefly broken. A 401
#: is handled separately because the fix is a new token rather than patience, and
#: a 403 is deliberately absent from both paths -- see `_hint`.
_RETRYABLE = frozenset({429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """An API call that cannot be retried into success."""


@dataclass(frozen=True)
class Response:
    status: int
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


#: url, headers, form-encoded body (None for GET) -> Response.
Transport = Callable[[str, dict[str, str], bytes | None], Response]


def urllib_transport(
    url: str, headers: dict[str, str], data: bytes | None = None
) -> Response:
    """The real transport: stdlib only, so a corpus fetch adds no dependency."""
    request = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=300) as handle:  # noqa: S310
            return Response(handle.status, handle.read())
    except urllib.error.HTTPError as error:
        # A 4xx carries the API's explanation in its body, which is the only
        # thing that distinguishes "wrong key" from "no bundle built yet".
        return Response(error.code, error.read())
    except urllib.error.URLError as error:
        raise FetchError(f"{url} unreachable: {error.reason}") from error


class Bearer(Protocol):
    """Anything that can put a token on a request, and replace a stale one.

    ``refresh`` is what the 401 retry needs, and is why this protocol is two
    methods rather than one: "I got you a new token, ask again" and "I have
    nothing left to mint from, let the API's answer stand" are different
    outcomes, and a retry that cannot tell them apart either loops or gives up
    while a perfectly good credential sits unused. False is the second.
    """

    def bearer(self) -> str: ...

    def refresh(self) -> bool: ...


def _token_transport(transport: Transport) -> TokenTransport:
    """Adapt this repo's Transport to the exchange's narrower one.

    The two shapes differ deliberately. This module's transport carries the
    headers because its routes need `Accept` and an `Authorization` that changes
    between attempts; the exchange's carries a timeout instead, because it is
    one POST with one fixed header and the thing worth injecting there is how
    long to wait. Adapting is three lines, and is what keeps a test's fake
    transport covering the token exchange as well as the corpus routes -- if
    this seam did not exist, every test that minted would reach the network.

    The timeout is received and dropped, because there is nowhere here to put
    it: a `Transport` takes no deadline, `urllib_transport` sets its own, and an
    injected one answers immediately. The parameter is named rather than hidden
    so the next reader sees which argument is being declined and can widen
    `Transport` if a real deadline ever matters on this side.
    """

    def send(url: str, body: bytes, timeout_seconds: int) -> tuple[int, bytes]:
        response = transport(
            url, {"Content-Type": "application/x-www-form-urlencoded"}, body
        )
        return response.status, response.body

    return send


class FirebaseAuth:
    """The `Bearer` this module's routes take, minted by the memorome provider.

    A deliberately thin adapter, and it should stay thin. The exchange, the
    replacement deadline, the difference between a revoked credential and a rate
    limit, and the account name a token belongs to are all
    ``memorome_takeout.firebase_token``'s -- this repo carried its own copy of
    every one of them once, alongside two other repos carrying theirs, which is
    the drift the dependency exists to end. What is left here is the wiring:
    this module's `Transport` in, this module's `Bearer` out.

    Credential *discovery* stays on this side, which is why the token is passed
    in rather than looked up: `fetch_corpus.read_refresh_token` searches the
    environment, `.env.local`, the desktop app's token file and finally the
    macOS keychain, a chain the provider knows nothing about. Handing it
    ``refresh_token=`` skips its own environment search entirely.

    ``refresh_token_env`` is passed for the same reason and read back by nobody
    here: it is the name the provider attributes the credential to when it
    describes itself or refuses to renew, so a message about a credential names
    the variable a human can go and set rather than "a supplied credential".

    Not a dataclass, unlike its neighbours here: a generated ``repr`` would put
    the refresh token into every log line, assertion failure and traceback that
    formatted this object. The token is handed to the provider and never stored
    on the adapter at all.
    """

    def __init__(
        self,
        *,
        refresh_token: str,
        environment: str,
        refresh_token_env: str | None = None,
        transport: Transport = urllib_transport,
    ) -> None:
        self._announced = False
        self._provider = IdTokenProvider(
            environment=environment,
            refresh_token=refresh_token,
            refresh_token_env=refresh_token_env,
            transport=_token_transport(transport),
        )

    def bearer(self) -> str:
        """A currently-valid ID token, minted on first use and as it ages out."""
        token = self._provider.token()
        if not self._announced:
            # Once per fetch, after the first mint rather than at construction:
            # `describe` reads the token's own claims, so before one exists
            # there is no account to name and asking for one would spend an
            # exchange on a run that may never make a request. It renders an
            # email or a uid plus the variable the credential came from -- the
            # provenance `refresh_token_env` is passed for, and never the token.
            self._announced = True
            log.info("fetching as %s", self._provider.describe())
        return token

    def refresh(self) -> bool:
        """Discard the held token and mint a replacement; see `Bearer`.

        False when the provider has nothing to mint from. It cannot happen with
        a credential this repo found -- `read_refresh_token` either returns a
        refresh token or raises -- but the retry asks rather than assumes,
        because the provider also accepts a bare ID token that cannot be renewed
        and answering "no" is how it says so.
        """
        return self._provider.refresh()


def _hint(status: int) -> str:
    """What each rejection actually means, in the API's own terms.

    Returned as the suffix an error message appends -- empty when the status has
    no story worth telling -- so every caller formats it the same way.
    """
    hint = {
        400: "a malformed request -- a bad top_k, or an empty parameter",
        401: (
            "the bearer token is expired, malformed, or was minted against a "
            "different Firebase project than this environment -- and re-minting "
            "it, which this call does once wherever the credential can be "
            "renewed, did not help"
        ),
        403: (
            "the account is refused, not the token: this user's email must be "
            "allowlisted for this route and the user must have technical "
            "access. A fresh token is the same account, so re-minting only "
            "repeats the rejection -- which is why a 403 is never retried"
        ),
        404: "no recall bundle for this user yet -- resolution has not been built",
        429: "gateway rate limit; retry shortly",
        503: "the allowlist lookup failed server-side; retry",
    }.get(status)
    return f" -- {hint}" if hint else ""


def _get(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    auth: Bearer,
    *,
    attempts: int = 3,
    pause: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Response:
    """GET with retries on the statuses that are worth retrying.

    A 429 or a 5xx is the server asking to be asked again; a 403 or a 404 is a
    settled answer and retrying it three times only delays the message that says
    what to fix.

    A 401 is neither. It says the *token* is stale, not that the request was
    wrong or that the account was refused, so the call re-mints once and asks
    again -- otherwise a run that outlives its ID token dies mid-corpus with a
    message about authentication for a credential that is perfectly good. That
    re-mint happens at most once: a token the API keeps refusing must fail with
    the API's own 401, not with three of them. It also costs no attempt and no
    backoff, because a stale credential is neither a failed try nor the server
    asking for patience.

    ``auth`` is required rather than optional because every route in this module
    is authenticated, and the Authorization header is built here, per attempt,
    rather than by the caller: a token replaced between two attempts has to be
    the one the second attempt sends. A 403 is pointedly absent from both paths.
    """
    last: Response | None = None
    remints_left = 1
    attempts_left = attempts
    while attempts_left:
        sent = dict(headers) | {"Authorization": f"Bearer {auth.bearer()}"}
        last = transport(url, sent, None)
        if last.status == 200:
            return last
        if last.status == 401 and remints_left and auth.refresh():
            # Costs no attempt and no backoff, and `remints_left` bounds it to
            # one. A `refresh` answering False falls through to the
            # settled-answer path below, so the API's own 401 is what the caller
            # sees rather than a retry loop chasing a credential that is gone.
            remints_left -= 1
            log.info("the API refused the ID token; minted a new one and retried")
            continue
        attempts_left -= 1
        if last.status not in _RETRYABLE:
            return last
        if attempts_left:
            sleep(pause * (attempts - attempts_left))
    assert last is not None
    return last


def _is_count(value: object) -> bool:
    """A JSON integer -- and not a bool, which Python would otherwise let through."""
    return isinstance(value, int) and not isinstance(value, bool)


def fetch_bundle(
    base_url: str,
    auth: Bearer,
    *,
    transport: Transport = urllib_transport,
) -> dict[str, Any]:
    """Return the whole resolved-entities bundle, validated into the shape on disk.

    Authenticated with the same Firebase bearer the files routes take. This
    route used to want a master ``x-api-key`` and no longer does, and the two
    are never sent together: the platform refuses a request carrying mixed
    credentials, so "send both and let the server decide" is not an option, and
    would not be one worth taking.

    The request takes no knobs, on purpose. ``top_k`` is pinned at the server's
    cap because anything smaller truncates. ``include_aliases`` is always on
    because a bundle without aliases has nothing for retrieval to search: the
    profile pipeline reads surface forms, not canonical names, so an alias-less
    bundle would produce empty packs for everybody rather than an obvious error.
    And there is no ``case`` filter: a subset of the people under the same
    filename is a different artifact rather than a smaller one, and nothing
    downstream could tell the two apart.
    """
    query = urllib.parse.urlencode({"top_k": MAX_TOP_K, "include_aliases": "true"})
    response = _get(
        transport,
        f"{base_url}/v1/entities/resolved?{query}",
        {"Accept": "application/json"},
        auth,
    )
    if response.status != 200:
        raise FetchError(
            f"GET /v1/entities/resolved returned HTTP {response.status}"
            f"{_hint(response.status)}"
        )
    bundle = response.json()
    if (
        not isinstance(bundle, dict)
        or not isinstance(bundle.get("entities"), list)
        or not _is_count(bundle.get("count"))
        or not _is_count(bundle.get("total_in_bundle"))
    ):
        # The counts are held to the same standard as the list: a truncation
        # check that shrugs at a count it cannot compare is no check at all.
        raise FetchError("the response is not a resolved-entities bundle")

    count, total = bundle["count"], bundle["total_in_bundle"]
    if count < total:
        # Silently keeping this would hand the pipeline a corpus that is
        # missing people while every count downstream reports success.
        raise FetchError(
            f"the bundle is truncated: {count} of {total} entities, and "
            f"{MAX_TOP_K} is the most the server returns in one request"
        )
    return {key: bundle[key] for key in ("entities", "count", "total_in_bundle")}


def list_items(
    base_url: str,
    auth: Bearer,
    source_type: str,
    *,
    transport: Transport = urllib_transport,
    page_limit: int = PAGE_LIMIT,
) -> Iterator[str]:
    """Yield every item id under one source type, following the pagination.

    ``has_more`` is trusted over the page size: a full page is not the same
    statement as "there is another page", and stopping on a short page would
    truncate a source type without saying so.
    """
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {"source_type": source_type, "limit": page_limit, "offset": offset}
        )
        response = _get(
            transport,
            f"{base_url}/v1/files/list?{query}",
            {"Accept": "application/json"},
            auth,
        )
        if response.status != 200:
            raise FetchError(
                f"GET /v1/files/list?source_type={source_type} returned HTTP "
                f"{response.status}{_hint(response.status)}"
            )
        payload = response.json()
        item_ids = payload.get("item_ids") or []
        yield from item_ids
        if not payload.get("has_more") or not item_ids:
            return
        offset += page_limit


def download_item(
    base_url: str,
    auth: Bearer,
    source_type: str,
    item_id: str,
    *,
    transport: Transport = urllib_transport,
) -> dict[str, Any]:
    """Return one document's decrypted payload."""
    query = urllib.parse.urlencode({"item_id": item_id, "source_type": source_type})
    response = _get(
        transport,
        f"{base_url}/v1/files/download?{query}",
        {"Accept": "application/json"},
        auth,
    )
    if response.status != 200:
        raise FetchError(
            f"downloading {source_type}/{item_id} returned HTTP "
            f"{response.status}{_hint(response.status)}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise FetchError(f"{source_type}/{item_id} is not a document object")
    return payload


def document_text(payload: dict[str, Any]) -> str:
    """The text this document contributes to the corpus.

    The download route answers with ``{item_id, source_type, content, user_id}``
    and the pipeline reads only ``content``; the rest is addressing. A payload
    whose ``content`` is missing, not a string, or blank is refused by name
    rather than written as an empty document -- an empty ``.txt`` is evidence
    that exists, matches nothing, and makes an entity look unmentioned.
    """
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        keys = sorted(payload)
        raise FetchError(
            f"no usable 'content' in the payload (keys: {keys}); "
            "the download shape has changed"
        )
    return content
