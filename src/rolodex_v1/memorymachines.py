"""Fetch a corpus from the memorymachines API: the bundle, and the documents.

Reverse-engineered from `fetch_resolved_entities_bundle.sh` and
`dump_all_source_docs.sh`, which are the working record of what these endpoints
accept. The two are kept in one module because they are one corpus, and split
into two credentials because the API splits them:

**The two halves do not share an authentication scheme, and cannot.** The bundle
route takes a master ``x-api-key``. The files routes go through the API's
``_require_firebase_principal_from_request`` and *reject* an API key with 403, so
they need a Firebase refresh token exchanged for a short-lived ID token. There is
no single credential that fetches a whole corpus, and a fetch configured with
only one of them does half the job -- which is why each half is asked for its own
credential at the point it is needed, and reports which one it wanted rather than
failing at the far end with somebody else's 403.

Nothing here logs a credential, and callers must not either. The error messages
name the *variable*, never the value.

The transport is injected (``Transport``) so the retry policy, the pagination
and the JSON-to-text conversion are testable without a network or a token.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

SECURE_TOKEN_URL = "https://securetoken.googleapis.com/v1/token"  # noqa: S105


@dataclass(frozen=True)
class Environment:
    """One deployment of the API, and the Firebase project that signs in to it."""

    base_url: str
    firebase_api_key: str


#: The Firebase Web API key must belong to the project that issued the refresh
#: token, or the exchange fails with a 400 that reads like a revoked credential.
#: The keys are not secrets: Google treats a Web API key as a public identifier
#: of a Firebase project, and every client shipped for it embeds the same one.
#: What authenticates is the refresh token, which is never in this repository.
ENVIRONMENTS = {
    "prod": Environment(
        "https://api.engramme.com",
        "AIzaSyB7DIVqzT72Pg9KAhJQCxNgBw7ZeTyLkzc",
    ),
    "staging": Environment(
        "https://api-staging.engramme.com",
        "AIzaSyAOPF6EQ_oSDUhFbRMqKlezxm7C8-d7i_s",
    ),
    "dev": Environment(
        "https://api-dev.engramme.com",
        "AIzaSyApDlbf3kensbIpgkjzH5X-ehHDqJohp5M",
    ),
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

#: An ID token lives an hour; re-mint well inside that, because the run between
#: two mints may be thousands of downloads long.
TOKEN_TTL_SECONDS = 2_700


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


def _hint(status: int) -> str:
    """What each rejection actually means, in the API's own terms.

    Returned as the suffix an error message appends -- empty when the status has
    no story worth telling -- so every caller formats it the same way.
    """
    hint = {
        400: "bad top_k or case, or an empty key",
        401: "the API key is invalid or unknown",
        403: (
            "the key must be a master key (no allowed_sources) and its user's "
            "email must be allowlisted; a files route rejects an API key "
            "outright and needs a Firebase token instead"
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
    *,
    attempts: int = 3,
    pause: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Response:
    """GET with retries on the statuses that are worth retrying.

    A 429 or a 5xx is the server asking to be asked again; a 401/403/404 is a
    settled answer and retrying it three times only delays the message that
    says what to fix.
    """
    last: Response | None = None
    for attempt in range(1, attempts + 1):
        last = transport(url, headers, None)
        if last.status == 200 or last.status not in (429, 500, 502, 503, 504):
            return last
        if attempt < attempts:
            sleep(pause * attempt)
    assert last is not None
    return last


class Bearer(Protocol):
    """Anything that can put a token on a request."""

    def bearer(self) -> str: ...


@dataclass
class FirebaseAuth:
    """A refresh token, and the short-lived ID tokens minted from it.

    The ID token is re-minted on a timer rather than on a 401, because the
    alternative is discovering the expiry in the middle of a long download and
    counting the failure as a missing document.
    """

    refresh_token: str
    firebase_api_key: str
    transport: Transport = urllib_transport
    now: Callable[[], float] = time.monotonic
    _token: str = field(default="", init=False)
    _minted_at: float = field(default=0.0, init=False)

    def bearer(self) -> str:
        if not self._token or self.now() - self._minted_at >= TOKEN_TTL_SECONDS:
            self._mint()
        return self._token

    def _mint(self) -> None:
        body = urllib.parse.urlencode(
            {"grant_type": "refresh_token", "refresh_token": self.refresh_token}
        ).encode()
        response = self.transport(
            f"{SECURE_TOKEN_URL}?key={self.firebase_api_key}",
            {"Content-Type": "application/x-www-form-urlencoded"},
            body,
        )
        if response.status != 200:
            # A 400 here means the credential is finished (expired, revoked,
            # user disabled) or the Firebase key belongs to another project.
            raise FetchError(
                f"token exchange failed (HTTP {response.status}): the refresh "
                "token is expired or revoked, or belongs to another Firebase "
                "project. Sign in to the Engramme desktop app again."
            )
        payload = response.json()
        token = payload.get("id_token")
        if not token:
            raise FetchError("token exchange returned no id_token")
        self._token = token
        self._minted_at = self.now()
        log.info("minted an ID token for uid %s", payload.get("user_id", "?"))


def _is_count(value: object) -> bool:
    """A JSON integer -- and not a bool, which Python would otherwise let through."""
    return isinstance(value, int) and not isinstance(value, bool)


def fetch_bundle(
    base_url: str,
    api_key: str,
    *,
    transport: Transport = urllib_transport,
) -> dict[str, Any]:
    """Return the whole resolved-entities bundle, validated into the shape on disk.

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
        {"x-api-key": api_key, "Accept": "application/json"},
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
            {
                "Authorization": f"Bearer {auth.bearer()}",
                "Accept": "application/json",
            },
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
        {"Authorization": f"Bearer {auth.bearer()}", "Accept": "application/json"},
    )
    if response.status != 200:
        raise FetchError(
            f"downloading {source_type}/{item_id} returned HTTP {response.status}"
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
