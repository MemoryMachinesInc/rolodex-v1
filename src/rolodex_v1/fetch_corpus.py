"""Download a corpus from the memorymachines API into the configured paths.

    uv run python -m rolodex_v1.fetch_corpus --dry-run

This is the alternative to pointing `[paths]` at an export somebody handed you:
the same two locations, filled from the API instead of by hand. It is a
*separate command* from `build_profiles` on purpose -- a run that spends money
per entity should not also be the thing that decides to pull thousands of
documents over the network, and a corpus that changes underneath a half-finished
profile set is a corpus nobody can explain afterwards.

**The two halves need two different credentials** (see `memorymachines`): an
``x-api-key`` master key for the bundle, and a Firebase refresh token for the
documents, because the files routes reject an API key outright. Either half can
be run alone (`--only bundle`, `--only source-docs`), which is what a machine
holding one of the two credentials should do.

Documents arrive as one JSON object per document and land as `.txt`, because
`.txt` at any depth is what the pack builder reads. The conversion is not a
separate pass over a staging directory: a document is converted as it arrives,
so an interrupted fetch leaves a directory of usable evidence rather than a
directory of JSON that looks like an empty corpus. `--from-dump` converts a
directory an earlier `dump_all_source_docs.sh` run already downloaded.

Both halves are incremental, like every script here. A document already on disk
is not downloaded again, so an interrupted fetch resumes. An existing bundle is
left alone unless `--force`: it is what every checkpointed entity id *means*, so
replacing it under a half-built profile set would detach the profiles already
bought from the ids that name them.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from rolodex_v1 import credentials, settings
from rolodex_v1.memorymachines import (
    DEFAULT_SOURCES,
    ENVIRONMENTS,
    Bearer,
    FetchError,
    FirebaseAuth,
    Transport,
    document_text,
    download_item,
    fetch_bundle,
    list_items,
    urllib_transport,
)

log = logging.getLogger(__name__)

#: Where the Engramme desktop app keeps the Firebase refresh token, and the
#: 0600-file fallback its own dump script reads on machines that refuse the
#: keychain.
KEYCHAIN_SERVICE = "ai.memorymachines.engramme"
KEYCHAIN_ACCOUNT = "engramme_refresh_token"  # noqa: S105 - an account name
TOKEN_FILE = Path("~/.engramme/engramme_refresh_token.txt")

#: The keychain item is ACL'd to the app's code signature, so reading it from
#: anything else raises a consent dialog. A shell with no way to show one waits
#: for it forever -- a hang that looks exactly like a slow download, which is
#: the failure mode AGENTS.md trap 9 already cost somebody 45 minutes. Bound it.
KEYCHAIN_TIMEOUT_SECONDS = 20


def read_refresh_token(env_name: str) -> str:
    """The Firebase refresh token, from the first place that has one.

    Environment and ``.env.local`` first, because they are the two a script can
    set. Then the file the desktop app's own dumper falls back to. Then the
    keychain, which is where the app actually keeps it -- last, because it is
    the only source that can stop and ask a human something.
    """
    try:
        return credentials.read_env_value(env_name)
    except RuntimeError as missing:
        token_file = TOKEN_FILE.expanduser()
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token
        return _read_keychain(env_name, missing)


def _read_keychain(env_name: str, missing: RuntimeError) -> str:
    """Ask the keychain, with a deadline and an explanation for every outcome."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=KEYCHAIN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Timeout means the consent dialog is waiting somewhere this process
        # cannot see. Say so, and name the two ways out that do not need one.
        raise RuntimeError(
            f"the keychain did not answer within {KEYCHAIN_TIMEOUT_SECONDS}s -- "
            "macOS is waiting on a consent dialog this process cannot show. "
            "Approve it once by running this in your own Terminal:\n"
            f"  security find-generic-password -s {KEYCHAIN_SERVICE} "
            f"-a {KEYCHAIN_ACCOUNT} -w\n"
            f"then set ${env_name}, or write the token to {TOKEN_FILE}."
        ) from None
    token = result.stdout.strip()
    if not token:
        raise RuntimeError(
            f"{missing}; the keychain item {KEYCHAIN_SERVICE}/{KEYCHAIN_ACCOUNT} "
            "is absent or was denied. Sign in to the Engramme desktop app, or "
            f"set ${env_name}."
        ) from None
    log.info("read the refresh token from the keychain")
    return token


#: An item id is a filename here, and ids arrive from an API. Anything outside
#: this set is replaced rather than trusted: a `/` or a `..` in an id would
#: otherwise write outside the corpus directory.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_name(item_id: str) -> str:
    """The filename one item id gets, matching the shell dump's sanitizer."""
    return _UNSAFE.sub("_", item_id)


def write_document(directory: Path, item_id: str, payload: dict) -> Path:
    """Write one document's text, and return where it landed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_name(item_id)}.txt"
    path.write_text(document_text(payload), encoding="utf-8")
    return path


def convert_dump(
    dump_dir: Path, out_dir: Path, *, force: bool = False
) -> tuple[int, int]:
    """Turn a downloaded `{source}/{id}.json` tree into `{source}/{id}.txt`.

    The dump is read, never emptied: it is the only copy of what the API
    actually answered, and a conversion that deletes its input cannot be re-run
    when the shape turns out to have been misread.

    Returns (written, skipped). A document whose payload has no usable content
    is counted as skipped and named in the log rather than aborting the pass --
    one changed record should not strand the thousands beside it, and the count
    is reported so "some were skipped" cannot pass for "all converted".
    """
    written = skipped = 0
    for path in sorted(dump_dir.rglob("*.json")):
        relative = path.relative_to(dump_dir)
        destination = (out_dir / relative).with_suffix(".txt")
        if destination.exists() and not force:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise FetchError(
                    f"expected a document object, got JSON {type(payload).__name__}"
                )
            text = document_text(payload)
        except (json.JSONDecodeError, FetchError) as error:
            log.warning("skipping %s: %s", relative, error)
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        written += 1
    return written, skipped


def fetch_source_docs(
    base_url: str,
    auth: Bearer,
    out_dir: Path,
    *,
    sources: tuple[str, ...],
    transport: Transport = urllib_transport,
    force: bool = False,
) -> tuple[int, int]:
    """Download every document under each source type as text.

    Returns (written, skipped). Already-present files are skipped without a
    request, so an interrupted fetch resumes instead of re-downloading -- the
    same property the profile checkpoints have, for the same reason.
    """
    written = skipped = 0
    for source in sources:
        directory = out_dir / source
        count = 0
        for item_id in list_items(base_url, auth, source, transport=transport):
            destination = directory / f"{safe_name(item_id)}.txt"
            if destination.exists() and not force:
                continue
            try:
                payload = download_item(
                    base_url, auth, source, item_id, transport=transport
                )
                write_document(directory, item_id, payload)
            except FetchError as error:
                log.warning("skipping %s/%s: %s", source, item_id, error)
                skipped += 1
                continue
            written += 1
            count += 1
            if count and count % 100 == 0:
                log.info("[%s] %d documents", source, count)
        if count:
            log.info("[%s] done: %d documents", source, count)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path config; the same file build_profiles reads",
    )
    parser.add_argument(
        "--only",
        choices=("bundle", "source-docs"),
        default=None,
        help=(
            "fetch one half. The two halves need different credentials, so a "
            "machine holding only one of them runs only its half."
        ),
    )
    parser.add_argument(
        "--from-dump",
        type=Path,
        default=None,
        help=(
            "convert a directory an earlier dump_all_source_docs.sh run wrote, "
            "instead of downloading. Reads the dump, never empties it."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite documents already on disk instead of skipping them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be fetched, where it would land, and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        paths = settings.resolve(config=args.config)
    except ValueError as error:
        log.error("%s", error)
        return 1

    if args.from_dump is not None:
        if not args.from_dump.is_dir():
            log.error("%s is not a directory", args.from_dump)
            return 1
        log.info("convert %s -> %s", args.from_dump, paths.source_docs)
        if args.dry_run:
            total = len(list(args.from_dump.rglob("*.json")))
            log.info("would convert %d JSON documents", total)
            return 0
        written, skipped = convert_dump(
            args.from_dump, paths.source_docs, force=args.force
        )
        log.info("converted %d documents, skipped %d", written, skipped)
        # A skipped document is one the corpus is missing. The exit code says
        # so even when the rest converted, or "some were skipped" in a log
        # nobody reads passes for "all converted".
        return 1 if skipped else 0

    fetch = paths.fetch
    if fetch is None:
        # Refused rather than defaulted: inventing an environment here would ask
        # a production API for somebody's documents on the strength of nothing
        # the user wrote down.
        log.error(
            "no [fetch] table in %s; add one (see configs/rolodex-v1.toml.example) "
            "or point [paths] at an export you already have",
            paths.config_path or "any config file",
        )
        return 1

    environment = ENVIRONMENTS[fetch.environment]
    base_url = (fetch.base_url or environment.base_url).rstrip("/")
    wants_bundle = args.only in (None, "bundle")
    wants_docs = args.only in (None, "source-docs")
    bundle_exists = paths.entities_bundle.exists()

    log.info("config   %s", paths.origin)
    log.info("base     %s (%s)", base_url, fetch.environment)
    if wants_bundle:
        log.info(
            "bundle   -> %s (%s; needs $%s)",
            paths.entities_bundle,
            "exists, skipped unless --force" if bundle_exists else "absent",
            fetch.api_key_env,
        )
    if wants_docs:
        sources = fetch.sources or DEFAULT_SOURCES
        log.info(
            "docs     -> %s (needs $%s, %d source types)",
            paths.source_docs,
            fetch.refresh_token_env,
            len(sources),
        )
    if args.dry_run:
        log.info("dry run: nothing fetched, nothing written")
        return 0

    status = 0
    if wants_bundle and bundle_exists and not args.force:
        log.info(
            "bundle   %s exists; skipped (--force replaces it)", paths.entities_bundle
        )
    elif wants_bundle:
        try:
            api_key = credentials.read_env_value(fetch.api_key_env)
            bundle = fetch_bundle(base_url, api_key)
        except RuntimeError as error:
            log.error("bundle: %s", error)
            status = 1
        else:
            paths.entities_bundle.parent.mkdir(parents=True, exist_ok=True)
            # Written the way the checked-in bundles are, so a fetched corpus
            # and a hand-delivered one are the same file byte for byte.
            paths.entities_bundle.write_text(
                json.dumps(bundle, indent=2) + "\n", encoding="utf-8"
            )
            log.info(
                "bundle   %d entities -> %s", bundle["count"], paths.entities_bundle
            )

    if wants_docs:
        try:
            refresh_token = read_refresh_token(fetch.refresh_token_env)
            auth = FirebaseAuth(
                refresh_token=refresh_token,
                firebase_api_key=environment.firebase_api_key,
            )
            written, skipped = fetch_source_docs(
                base_url,
                auth,
                paths.source_docs,
                sources=fetch.sources or DEFAULT_SOURCES,
                force=args.force,
            )
        except RuntimeError as error:
            log.error("source docs: %s", error)
            return 1
        log.info(
            "docs     %d written, %d skipped -> %s", written, skipped, paths.source_docs
        )
        if skipped:
            # Each skip was named as it happened; the exit code carries the fact
            # past the log, so a run that refused documents cannot report success.
            status = 1

    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
