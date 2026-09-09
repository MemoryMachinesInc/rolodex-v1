#!/usr/bin/env bash
# Dump EVERY source document for the signed-in Engramme user.
#
# Auth chain (the same one engramme_desktop uses):
#   1. read the Firebase REFRESH token from the macOS keychain
#        service ai.memorymachines.engramme / account engramme_refresh_token
#        (fallback: ~/.engramme/engramme_refresh_token.txt)
#   2. exchange it at securetoken.googleapis.com for a short-lived ID TOKEN
#   3. send that ID token as `Authorization: Bearer <id_token>`
#
# The API endpoints only accept a Firebase ID token: both go through
# _require_firebase_principal_from_request, which 403s x-api-key.
#
#   GET /v1/files/list?source_type=&limit=&offset=   -> item_ids  (paginated)
#   GET /v1/files/download?item_id=&source_type=     -> decrypted content
#
# Usage:
#   ./dump_all_source_docs.sh [output_dir]
#   MM_ENV=staging ./dump_all_source_docs.sh
#   MM_SOURCES="email slack" ./dump_all_source_docs.sh
#   MM_REFRESH_TOKEN=<token> ./dump_all_source_docs.sh   # skip the keychain
#   MM_ID_TOKEN=<jwt> ./dump_all_source_docs.sh          # bring your own bearer

set -uo pipefail

# ── environment ─────────────────────────────────────────────────────────────
# Bases match engramme_desktop's MM_API_BASE (api.engramme.com fronts the same
# API Gateway through the HTTPS load balancer). The Firebase Web API key must
# belong to the project that ISSUED the refresh token, or securetoken rejects
# the exchange with a revoked-looking 400.
case "${MM_ENV:-prod}" in
  prod)
    BASE="https://api.engramme.com"
    FIREBASE_API_KEY="AIzaSyB7DIVqzT72Pg9KAhJQCxNgBw7ZeTyLkzc" ;;
  staging)
    BASE="https://api-staging.engramme.com"
    FIREBASE_API_KEY="AIzaSyAOPF6EQ_oSDUhFbRMqKlezxm7C8-d7i_s" ;;
  dev)
    BASE="https://api-dev.engramme.com"
    FIREBASE_API_KEY="AIzaSyApDlbf3kensbIpgkjzH5X-ehHDqJohp5M" ;;
  *)
    echo "error: MM_ENV must be prod|staging|dev" >&2; exit 1 ;;
esac
BASE="${MM_BASE:-$BASE}"

KEYCHAIN_SERVICE="ai.memorymachines.engramme"
KEYCHAIN_ACCOUNT="engramme_refresh_token"
REFRESH_TOKEN_FILE="$HOME/.engramme/engramme_refresh_token.txt"
SECURE_TOKEN_URL="https://securetoken.googleapis.com/v1/token"

OUT_DIR="${1:-./source_docs_dump}"
PAGE_LIMIT="${MM_PAGE_LIMIT:-10000}"   # server max
SLEEP="${MM_SLEEP:-0}"                 # gateway quota is 5000 req/min/project
TOKEN_TTL="${MM_TOKEN_TTL:-2700}"      # re-mint after 45 min; ID tokens live 1 h

command -v jq >/dev/null || { echo "error: jq required (brew install jq)" >&2; exit 1; }

# ── step 1: refresh token ───────────────────────────────────────────────────
read_refresh_token() {
  if [[ -n "${MM_REFRESH_TOKEN:-}" ]]; then
    printf '%s' "$MM_REFRESH_TOKEN"
    return 0
  fi
  # The keychain item is ACL'd to Engramme.app's code signature, so reading it
  # from a shell prompts for consent once ("Always Allow" makes it silent after).
  local from_keychain
  from_keychain=$(security find-generic-password \
    -s "$KEYCHAIN_SERVICE" -a "$KEYCHAIN_ACCOUNT" -w 2>/dev/null)
  if [[ -n "$from_keychain" ]]; then
    printf '%s' "$from_keychain"
    return 0
  fi
  # 0600-file fallback, used on machines that refuse the keychain.
  if [[ -s "$REFRESH_TOKEN_FILE" ]]; then
    tr -d '\r\n' < "$REFRESH_TOKEN_FILE"
    return 0
  fi
  return 1
}

# ── step 2: refresh token -> ID token ───────────────────────────────────────
ID_TOKEN=""
ID_TOKEN_MINTED_AT=0

mint_id_token() {
  if [[ -n "${MM_ID_TOKEN:-}" ]]; then
    ID_TOKEN="$MM_ID_TOKEN"
    ID_TOKEN_MINTED_AT=$SECONDS
    return 0
  fi

  local refresh_token
  if ! refresh_token=$(read_refresh_token) || [[ -z "$refresh_token" ]]; then
    echo "error: no refresh token found." >&2
    echo "  keychain: $KEYCHAIN_SERVICE / $KEYCHAIN_ACCOUNT" >&2
    echo "  file    : $REFRESH_TOKEN_FILE" >&2
    echo "  Sign in to the Engramme desktop app, or pass MM_REFRESH_TOKEN." >&2
    return 1
  fi

  local response status body
  response=$(curl -sS --max-time 20 -w '\n%{http_code}' \
    -X POST "$SECURE_TOKEN_URL?key=$FIREBASE_API_KEY" \
    --data-urlencode 'grant_type=refresh_token' \
    --data-urlencode "refresh_token=$refresh_token")
  status="${response##*$'\n'}"
  body="${response%$'\n'*}"

  if [[ "$status" != "200" ]]; then
    # 400 here means the credential is finished (TOKEN_EXPIRED,
    # INVALID_REFRESH_TOKEN, USER_DISABLED) or the key is from another project.
    echo "error: token exchange HTTP $status: $(printf '%s' "$body" | head -c 400)" >&2
    return 1
  fi

  ID_TOKEN=$(printf '%s' "$body" | jq -r '.id_token // empty')
  [[ -n "$ID_TOKEN" ]] || { echo "error: exchange returned no id_token" >&2; return 1; }
  ID_TOKEN_MINTED_AT=$SECONDS

  local expires_in user_id
  expires_in=$(printf '%s' "$body" | jq -r '.expires_in // "3600"')
  user_id=$(printf '%s' "$body" | jq -r '.user_id // "?"')
  echo "minted ID token for uid $user_id (expires in ${expires_in}s)"
}

ensure_fresh_token() {
  if (( SECONDS - ID_TOKEN_MINTED_AT >= TOKEN_TTL )); then
    echo "re-minting ID token..."
    mint_id_token || return 1
  fi
}

# ── the actual API call ─────────────────────────────────────────────────────
api_get() {  # api_get <path-with-query>  -> body + "\n" + http_code on stdout
  curl -sS --compressed \
    --retry 3 --retry-delay 2 --retry-connrefused \
    --max-time 120 \
    -w '\n%{http_code}' \
    -H "Authorization: Bearer $ID_TOKEN" \
    -H 'Accept: application/json' \
    "$BASE$1"
}

# ── source types ────────────────────────────────────────────────────────────
# VALID_SOURCE_TYPES from app_platform.py, plus the legacy "drive" GCS prefix
# that _get_searchable_sources still probes.
DEFAULT_SOURCES=(
  text email pdf stream browser vscode calendar github slack asana
  claude_code cursor codex google_meets technical_docs weekly_updates
  gdocs tasks contacts youtube photos books fit
  ms-outlook ms-calendar ms-teams ms-onedrive ms-sharepoint
  plaud imessage whatsapp drive
)
if [[ -n "${MM_SOURCES:-}" ]]; then
  read -r -a SOURCES <<< "$MM_SOURCES"
else
  SOURCES=("${DEFAULT_SOURCES[@]}")
fi

# ── run ─────────────────────────────────────────────────────────────────────
mint_id_token || exit 1

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.tsv"
[[ -s "$MANIFEST" ]] || printf 'source_type\titem_id\thttp_status\tbytes\tpath\n' > "$MANIFEST"

echo "base: $BASE"
echo "out : $OUT_DIR"
echo

total_docs=0
total_fail=0

for source in "${SOURCES[@]}"; do
  offset=0
  source_count=0
  src_dir="$OUT_DIR/$source"

  while :; do
    ensure_fresh_token || exit 1

    list_response=$(api_get "/v1/files/list?source_type=${source}&limit=${PAGE_LIMIT}&offset=${offset}")
    list_status="${list_response##*$'\n'}"
    list_body="${list_response%$'\n'*}"

    if [[ "$list_status" != "200" ]]; then
      echo "[$source] list HTTP $list_status offset=$offset: $(printf '%s' "$list_body" | head -c 300)" >&2
      break
    fi

    # bash 3.2 (macOS /bin/bash) has no mapfile
    item_ids=()
    while IFS= read -r line; do
      item_ids+=("$line")
    done < <(printf '%s' "$list_body" | jq -r '.item_ids[]? // empty')

    page_total=$(printf '%s' "$list_body" | jq -r '.total // 0')
    has_more=$(printf '%s' "$list_body" | jq -r '.has_more // false')

    if (( ${#item_ids[@]} == 0 )); then
      [[ "$offset" == "0" ]] && echo "[$source] 0 files"
      break
    fi

    echo "[$source] page offset=$offset  got=${#item_ids[@]}  total=$page_total"
    mkdir -p "$src_dir"

    for item_id in "${item_ids[@]}"; do
      safe_name=$(printf '%s' "$item_id" | tr -c 'A-Za-z0-9._-' '_')
      dest="$src_dir/${safe_name}.json"
      [[ -s "$dest" ]] && continue   # resumable: already downloaded

      ensure_fresh_token || exit 1

      encoded_item=$(jq -rn --arg v "$item_id" '$v|@uri')
      dl_response=$(api_get "/v1/files/download?item_id=${encoded_item}&source_type=${source}")
      dl_status="${dl_response##*$'\n'}"
      dl_body="${dl_response%$'\n'*}"

      if [[ "$dl_status" == "200" ]]; then
        printf '%s' "$dl_body" > "$dest"
        bytes=$(wc -c < "$dest" | tr -d ' ')
        source_count=$((source_count + 1))
        total_docs=$((total_docs + 1))
      else
        bytes=0
        dest="-"
        total_fail=$((total_fail + 1))
        echo "  [$source/$item_id] download HTTP $dl_status: $(printf '%s' "$dl_body" | head -c 200)" >&2
      fi

      printf '%s\t%s\t%s\t%s\t%s\n' "$source" "$item_id" "$dl_status" "$bytes" "$dest" >> "$MANIFEST"

      if (( source_count > 0 )) && (( source_count % 50 == 0 )); then
        echo "  [$source] $source_count downloaded (running total $total_docs)"
      fi
      [[ "$SLEEP" != "0" ]] && sleep "$SLEEP"
    done

    [[ "$has_more" == "true" ]] || break
    offset=$((offset + PAGE_LIMIT))
  done

  (( source_count > 0 )) && echo "[$source] done: $source_count docs"
done

echo
echo "docs written : $total_docs"
echo "failures     : $total_fail"
echo "output dir   : $OUT_DIR"
echo "manifest     : $MANIFEST"
