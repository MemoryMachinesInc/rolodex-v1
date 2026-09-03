#!/usr/bin/env bash
# Fetch the resolved-entities bundle for one API key's user and write it as
# JSON in the same shape as rolodex-v1/data/<corpus>/resolved_entities_bundle.json:
#
#   {
#     "entities": [
#       {
#         "resolved_entity_id": "proto:participant_person:20",
#         "canonical_name": "Josh Earnest",
#         "canonical_type": "person" | null,
#         "case": "participant_person",
#         "mention_count": 19734,
#         "memory_count": 19734,
#         "aliases": [ { "alias": "...", "probability": 1.0 }, ... ]
#       }, ...
#     ],
#     "count": 9180,
#     "total_in_bundle": 9180
#   }
#
# Endpoint (memorymachines_api_v0/app_platform.py, list_resolved_entities):
#   GET /v1/entities/resolved?top_k=&case=&include_aliases=
#   header: x-api-key: <user api key>
#
# This is NOT the Firebase-bearer auth that dump_all_source_docs.sh uses.
# The handler requires:
#   * a MASTER api key (one with no allowed_sources restriction) -> else 403
#   * the key's platform user email to be allowlisted
#     (@engramme.com / @memorymachines.ai, see TRACE_RESPONSE_ALLOWLIST_*) -> else 403
#   * a built recall bundle for the user -> else 404 ("trigger a build")
# top_k is capped at MAX_RESOLVED_ENTITIES_TOP_K = 50000; we default to the cap
# so the file is the whole bundle (count == total_in_bundle).
#
# Usage:
#   MM_API_KEY=<key> ./fetch_resolved_entities_bundle.sh [output.json]
#   MM_API_KEY_FILE=~/.engramme/api_key ./fetch_resolved_entities_bundle.sh
#   MM_API_KEY_SECRET=user-api-key-blake ./fetch_resolved_entities_bundle.sh  # via gcloud
#   MM_ENV=staging MM_API_KEY=... ./fetch_resolved_entities_bundle.sh
#   MM_CASE=participant_person MM_API_KEY=... ./fetch_resolved_entities_bundle.sh
#   MM_TOP_K=500 MM_INCLUDE_ALIASES=false MM_API_KEY=... ./fetch_resolved_entities_bundle.sh

set -uo pipefail

# ── environment ─────────────────────────────────────────────────────────────
# Same bases as dump_all_source_docs.sh (api.engramme.com fronts the API
# Gateway). MM_BASE overrides, e.g. the raw gateway host
# https://memorymachines-gateway-prod-btf57kda.uc.gateway.dev
case "${MM_ENV:-prod}" in
  prod)    BASE="https://api.engramme.com" ;;
  staging) BASE="https://api-staging.engramme.com" ;;
  dev)     BASE="https://api-dev.engramme.com" ;;
  *) echo "error: MM_ENV must be prod|staging|dev" >&2; exit 1 ;;
esac
BASE="${MM_BASE:-$BASE}"

OUT_FILE="${1:-./resolved_entities_bundle.json}"
TOP_K="${MM_TOP_K:-50000}"                      # server max (MAX_RESOLVED_ENTITIES_TOP_K)
CASE_FILTER="${MM_CASE:-}"                      # participant_person|entity_organization|entity_program|project
INCLUDE_ALIASES="${MM_INCLUDE_ALIASES:-true}"   # the rolodex bundle shape has aliases
GCP_PROJECT="${MM_GCP_PROJECT:-}"               # only used with MM_API_KEY_SECRET

command -v jq >/dev/null || { echo "error: jq required (brew install jq)" >&2; exit 1; }

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || (( TOP_K < 1 || TOP_K > 50000 )); then
  echo "error: MM_TOP_K must be an integer in 1..50000 (got '$TOP_K')" >&2; exit 1
fi
case "$INCLUDE_ALIASES" in
  true|false) ;;
  *) echo "error: MM_INCLUDE_ALIASES must be true|false" >&2; exit 1 ;;
esac

# ── api key ─────────────────────────────────────────────────────────────────
read_api_key() {
  if [[ -n "${MM_API_KEY:-}" ]]; then
    printf '%s' "$MM_API_KEY"; return 0
  fi
  if [[ -n "${MM_API_KEY_FILE:-}" ]]; then
    [[ -s "$MM_API_KEY_FILE" ]] || { echo "error: MM_API_KEY_FILE '$MM_API_KEY_FILE' missing/empty" >&2; return 1; }
    tr -d '\r\n' < "$MM_API_KEY_FILE"; return 0
  fi
  # Keys live in Secret Manager as user-api-key-{username} (v0 CLAUDE.md).
  if [[ -n "${MM_API_KEY_SECRET:-}" ]]; then
    command -v gcloud >/dev/null || { echo "error: gcloud required for MM_API_KEY_SECRET" >&2; return 1; }
    local args=(secrets versions access latest --secret "$MM_API_KEY_SECRET")
    [[ -n "$GCP_PROJECT" ]] && args+=(--project "$GCP_PROJECT")
    gcloud "${args[@]}" | tr -d '\r\n'; return "${PIPESTATUS[0]}"
  fi
  return 1
}

if ! API_KEY=$(read_api_key) || [[ -z "$API_KEY" ]]; then
  echo "error: no API key. Set MM_API_KEY, MM_API_KEY_FILE, or MM_API_KEY_SECRET." >&2
  exit 1
fi

# ── request ─────────────────────────────────────────────────────────────────
QUERY="top_k=${TOP_K}&include_aliases=${INCLUDE_ALIASES}"
if [[ -n "$CASE_FILTER" ]]; then
  QUERY+="&case=$(jq -rn --arg v "$CASE_FILTER" '$v|@uri')"
fi
URL="$BASE/v1/entities/resolved?$QUERY"

echo "base : $BASE"
echo "query: $QUERY"
echo "out  : $OUT_FILE"

TMP_BODY=$(mktemp "${TMPDIR:-/tmp}/resolved_entities.XXXXXX")
trap 'rm -f "$TMP_BODY"' EXIT

# Full bundles with aliases run to tens of MB; the gateway deadline for this
# route is 30s but allow the transfer itself plenty of time.
status=$(curl -sS --compressed \
  --retry 3 --retry-delay 2 --retry-connrefused \
  --max-time 300 \
  -o "$TMP_BODY" -w '%{http_code}' \
  -H "x-api-key: $API_KEY" \
  -H 'Accept: application/json' \
  "$URL") || { echo "error: curl failed" >&2; exit 1; }

if [[ "$status" != "200" ]]; then
  echo "error: HTTP $status from $URL" >&2
  echo "  body: $(head -c 400 "$TMP_BODY")" >&2
  case "$status" in
    400) echo "  hint: bad top_k/case, or empty key" >&2 ;;
    401) echo "  hint: API key invalid or unknown" >&2 ;;
    403) echo "  hint: key must be a master key (no allowed_sources) AND its user email must be" >&2
         echo "        allowlisted (@engramme.com / @memorymachines.ai). Partial keys are rejected." >&2 ;;
    404) echo "  hint: no recall bundle for this user yet -- entity resolution has not been built." >&2 ;;
    429) echo "  hint: gateway rate limit; retry shortly" >&2 ;;
    503) echo "  hint: allowlist lookup failed server-side; retry" >&2 ;;
  esac
  exit 1
fi

# ── validate + write ────────────────────────────────────────────────────────
if ! jq -e 'type == "object" and (.entities | type == "array") and has("count") and has("total_in_bundle")' \
     "$TMP_BODY" >/dev/null 2>&1; then
  echo "error: response is not a resolved-entities bundle:" >&2
  head -c 400 "$TMP_BODY" >&2; echo >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_FILE")"
# Pretty-print with 2-space indent and \uXXXX-escaped non-ASCII, matching the
# checked-in rolodex bundles byte for byte.
jq -a --indent 2 '{entities, count, total_in_bundle}' "$TMP_BODY" > "$OUT_FILE"

count=$(jq -r '.count' "$OUT_FILE")
total=$(jq -r '.total_in_bundle' "$OUT_FILE")
with_aliases=$(jq -r '[.entities[] | select(has("aliases"))] | length' "$OUT_FILE")
bytes=$(wc -c < "$OUT_FILE" | tr -d ' ')

echo
echo "entities returned : $count"
echo "total in bundle   : $total"
echo "with aliases      : $with_aliases"
echo "bytes             : $bytes"
echo "written           : $OUT_FILE"

if [[ -z "$CASE_FILTER" ]] && (( count < total )); then
  echo "warning: bundle truncated by top_k=$TOP_K ($count of $total). Raise MM_TOP_K (max 50000)." >&2
fi
jq -r '.entities | group_by(.case) | map("  \(.[0].case): \(length)") | .[]' "$OUT_FILE"
