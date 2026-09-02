#!/usr/bin/env bash
# Download a paper PDF and extract its text.
#
#   ./fetch_paper.sh <PDF_URL> <name> [outdir]
#
# Prints "ok: <path>.txt" on the last line, or "FAILED: <reason>" and exits non-zero.
#
# Why a script rather than handing the URL to a fetch tool: a summarizer returns a
# paraphrase, and the whole point of a deep dive is the methodological detail a
# paraphrase drops — the stated assumptions, the exact setup, the scope of the claim.
# Overlap judgments made from summaries are the ones that turn out wrong.

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <PDF_URL> <name> [outdir]" >&2
  exit 2
fi

URL="$1"
NAME="$(printf '%s' "$2" | tr -cs '[:alnum:]._-' '_')"
OUTDIR="${3:-${CLAUDE_PROJECT_DIR:-$PWD}/papers}"
PDF="$OUTDIR/$NAME.pdf"
TXT="$OUTDIR/$NAME.txt"

mkdir -p "$OUTDIR" || { echo "FAILED: cannot create $OUTDIR"; exit 1; }

if [ -s "$TXT" ]; then
  echo "cached"
  echo "ok: $TXT"
  exit 0
fi

# arXiv abs pages are the most common input; convert rather than fail.
# (sed, not ${var/pat/rep}: backslashes are literal in the replacement half, which
# silently produced hostnames like "arxiv.org\/pdf\/".)
case "$URL" in
  *arxiv.org/abs/*) URL="$(printf '%s' "$URL" | sed 's#/abs/#/pdf/#')" ;;
esac

curl -sSL --max-time 120 --retry 2 --retry-delay 3 \
     -A "Mozilla/5.0 (compatible; paper-forge/1.0)" \
     -o "$PDF" "$URL" || { echo "FAILED: download error for $URL"; exit 1; }

if [ ! -s "$PDF" ]; then
  echo "FAILED: empty download from $URL"
  exit 1
fi

# A paywall or a Cloudflare page saved as .pdf is the most common silent failure:
# the file exists, has bytes, and contains no paper.
if ! head -c 5 "$PDF" | grep -q '%PDF'; then
  SNIP="$(head -c 200 "$PDF" | tr -d '\0' | tr '\n' ' ')"
  rm -f "$PDF"
  echo "FAILED: response was not a PDF (paywall, login wall, or dead link) — began: ${SNIP:0:120}"
  exit 1
fi

extract() {
  case "$1" in
    layout)     command -v pdftotext  >/dev/null 2>&1 && pdftotext -layout "$PDF" "$TXT" 2>/dev/null ;;
    plain)      command -v pdftotext  >/dev/null 2>&1 && pdftotext          "$PDF" "$TXT" 2>/dev/null ;;
    pdfplumber) python3 - "$PDF" "$TXT" <<'PY' 2>/dev/null
import sys
import pdfplumber
with pdfplumber.open(sys.argv[1]) as d, open(sys.argv[2], "w", encoding="utf-8") as fh:
    for p in d.pages:
        fh.write((p.extract_text() or "") + "\n")
PY
      ;;
    pymupdf)    python3 - "$PDF" "$TXT" <<'PY' 2>/dev/null
import sys
import fitz
doc = fitz.open(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as fh:
    for page in doc:
        fh.write(page.get_text() + "\n")
PY
      ;;
  esac
  [ -s "$TXT" ] && [ "$(wc -c < "$TXT")" -gt 1000 ]
}

for method in layout plain pdfplumber pymupdf; do
  if extract "$method"; then
    CHARS=$(wc -c < "$TXT" | tr -d ' ')
    echo "extracted with: $method (${CHARS} chars)"
    if grep -qiE '^\s*[0-9.]*\s*(method|approach|model|our method|methodology)\b' "$TXT"; then
      echo "method section: found"
    else
      echo "method section: NOT FOUND — the extraction may be partial; read before trusting it"
    fi
    echo "ok: $TXT"
    exit 0
  fi
done

rm -f "$TXT"
echo "FAILED: every extractor produced empty or near-empty text (scanned image PDF?)."
echo "        Fall back to the abstract/HTML version and record the limitation explicitly —"
echo "        do not report having read a paper you could not read."
exit 1
