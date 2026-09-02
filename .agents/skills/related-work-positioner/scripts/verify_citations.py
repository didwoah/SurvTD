#!/usr/bin/env python3
"""Resolve every entry in a .bib file against public bibliographic indexes.

A language model producing a related-work section will occasionally emit a
citation that reads perfectly and does not exist — right-looking authors, a
plausible venue, a title that was never written.  It is the single most damaging
failure mode in agent-assisted writing, and it is invisible to proofreading.
Every entry here is resolved against Crossref, OpenAlex, arXiv, and Semantic
Scholar; anything that resolves nowhere is reported as probably fabricated.

    python3 verify_citations.py refs.bib --out citations.verified.json

No API keys.  Exit 0 = everything resolved, 1 = at least one entry unresolved or
title-mismatched, 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONTACT = os.environ.get("LIT_SEARCH_MAILTO", "research-skills@example.org")
UA = f"paper-forge-citation-check/1.0 (mailto:{CONTACT})"
MATCH_THRESHOLD = 0.82

_last: dict[str, float] = {}
MIN_INTERVAL = {"crossref": 0.3, "openalex": 0.2, "arxiv": 4.0, "s2": 3.0}


def _get(url: str, host: str, retries: int = 1) -> bytes | None:
    for attempt in range(retries + 1):
        wait = MIN_INTERVAL.get(host, 1.0) - (time.time() - _last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(4)
                continue
            return None
        except Exception:  # noqa: BLE001
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    return None


# --------------------------------------------------------------------------- #
# bib parsing — deliberately tolerant; a malformed entry is still worth checking
# --------------------------------------------------------------------------- #

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\s*\}\s*(?=@|\Z)", re.S)
FIELD_RE = re.compile(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)", re.S)


def strip_braces(v: str) -> str:
    v = v.strip().rstrip(",").strip()
    while len(v) >= 2 and ((v[0] == "{" and v[-1] == "}") or (v[0] == '"' and v[-1] == '"')):
        v = v[1:-1].strip()
    return re.sub(r"\s+", " ", v.replace("{", "").replace("}", "")).strip()


def parse_bib(text: str) -> list[dict]:
    out = []
    for kind, key, body in ENTRY_RE.findall(text):
        fields = {k.lower(): strip_braces(v) for k, v in FIELD_RE.findall(body)}
        out.append({"key": key.strip(), "type": kind.lower(), **fields})
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def similarity(a: str, b: str) -> float:
    """Token-level Jaccard — robust to the subtitle and punctuation drift between indexes."""
    ta, tb = set(norm(a).split()), set(norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# --------------------------------------------------------------------------- #
# resolvers, cheapest and most authoritative first
# --------------------------------------------------------------------------- #


def by_doi(doi: str) -> dict | None:
    raw = _get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", "crossref")
    if not raw:
        return None
    try:
        m = json.loads(raw)["message"]
    except Exception:  # noqa: BLE001
        return None
    titles = m.get("title") or []
    return {
        "source": "crossref/doi",
        "title": titles[0] if titles else "",
        "year": ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0],
        "venue": (m.get("container-title") or [""])[0],
        "url": m.get("URL", ""),
    }


def by_arxiv(aid: str) -> dict | None:
    aid = re.sub(r"^arxiv[:/]", "", aid.strip(), flags=re.I)
    raw = _get(f"http://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid)}", "arxiv")
    if not raw:
        return None
    m = re.search(r"<title>(.*?)</title>", raw.decode("utf-8", "replace"), re.S)
    titles = re.findall(r"<title>(.*?)</title>", raw.decode("utf-8", "replace"), re.S)
    # the first <title> is the feed's own; the entry title is the second
    if len(titles) < 2:
        return None
    published = re.search(r"<published>(\d{4})", raw.decode("utf-8", "replace"))
    return {
        "source": "arxiv/id",
        "title": re.sub(r"\s+", " ", titles[1]).strip(),
        "year": int(published.group(1)) if published else None,
        "venue": "arXiv preprint",
        "url": f"https://arxiv.org/abs/{aid}",
    }


def _rank(cands: list[dict], title: str, want_year: int | None) -> dict | None:
    """Best title match, but a landmark paper is re-indexed for decades under the same
    title, so a year the .bib supplies breaks the tie before similarity does."""
    scored = [(similarity(title, c["title"]), c) for c in cands if c.get("title")]
    scored = [(s, c) for s, c in scored if s >= MATCH_THRESHOLD]
    if not scored:
        return None
    if want_year:
        near = [(s, c) for s, c in scored if c.get("year") and abs(int(c["year"]) - want_year) <= 1]
        if near:
            scored = near
    scored.sort(key=lambda sc: -sc[0])
    return scored[0][1]


def by_title(title: str, want_year: int | None = None) -> dict | None:
    q = urllib.parse.urlencode({"search": title, "per-page": 8, "mailto": CONTACT})
    raw = _get(f"https://api.openalex.org/works?{q}", "openalex")
    if raw:
        try:
            cands = []
            for w in json.loads(raw).get("results", []):
                loc = (w.get("primary_location") or {}).get("source") or {}
                cands.append(
                    {
                        "source": "openalex/title",
                        "title": w.get("title") or w.get("display_name") or "",
                        "year": w.get("publication_year"),
                        "venue": loc.get("display_name", ""),
                        "url": (w.get("primary_location") or {}).get("landing_page_url") or w.get("id", ""),
                    }
                )
            hit = _rank(cands, title, want_year)
            if hit:
                return hit
        except Exception:  # noqa: BLE001
            pass
    q = urllib.parse.urlencode({"query": title, "limit": 8, "fields": "title,year,venue,url"})
    raw = _get(f"https://api.semanticscholar.org/graph/v1/paper/search?{q}", "s2")
    if raw:
        try:
            cands = [
                {
                    "source": "s2/title",
                    "title": p.get("title") or "",
                    "year": p.get("year"),
                    "venue": p.get("venue", ""),
                    "url": p.get("url", ""),
                }
                for p in (json.loads(raw).get("data", []) or [])
            ]
            hit = _rank(cands, title, want_year)
            if hit:
                return hit
        except Exception:  # noqa: BLE001
            pass
    return None


def resolve(entry: dict) -> dict:
    title = entry.get("title", "")
    result = {"key": entry["key"], "title": title, "status": "unresolved", "matched": None, "notes": []}

    for field in ("doi",):
        if entry.get(field):
            hit = by_doi(entry[field])
            if hit:
                result["matched"] = hit
                break
    if not result["matched"]:
        aid = entry.get("eprint") or entry.get("archiveprefix_id") or ""
        if not aid and "arxiv" in (entry.get("journal", "") + entry.get("note", "")).lower():
            m = re.search(r"(\d{4}\.\d{4,5})", entry.get("journal", "") + entry.get("note", ""))
            aid = m.group(1) if m else ""
        if aid:
            hit = by_arxiv(aid)
            if hit:
                result["matched"] = hit
    if not result["matched"] and title:
        want_year = None
        if str(entry.get("year", "")).strip().isdigit():
            want_year = int(str(entry["year"]).strip())
        hit = by_title(title, want_year)
        if hit:
            result["matched"] = hit

    if not result["matched"]:
        result["status"] = "unresolved"
        result["notes"].append(
            "no index returned this work by DOI, arXiv id, or title — treat as fabricated until proven otherwise"
        )
        return result

    sim = similarity(title, result["matched"]["title"]) if title else 1.0
    result["title_similarity"] = round(sim, 3)
    if sim >= MATCH_THRESHOLD:
        result["status"] = "verified"
    else:
        result["status"] = "mismatch"
        result["notes"].append(
            f"resolved record is titled {result['matched']['title']!r}; the .bib says {title!r} — "
            "a chimeric citation (real paper, wrong metadata) or the wrong paper entirely"
        )
    for field, key in (("year", "year"), ("journal", "venue"), ("booktitle", "venue")):
        want, got = entry.get(field), result["matched"].get(key)
        if want and got and norm(str(want)) not in norm(str(got)) and field == "year":
            result["notes"].append(f"year mismatch: .bib says {want}, index says {got}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bib")
    ap.add_argument("--out", default=None, help="write citations.verified.json here")
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    args = ap.parse_args()

    try:
        with open(args.bib, encoding="utf-8") as fh:
            entries = parse_bib(fh.read())
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read {args.bib}: {exc}", file=sys.stderr)
        return 2
    if not entries:
        print(f"error: no @entries found in {args.bib}", file=sys.stderr)
        return 2

    print(f"verify_citations: {len(entries)} entr(ies) in {args.bib}\n")
    results = []
    for e in entries:
        res = resolve(e)
        results.append(res)
        mark = {"verified": "ok   ", "mismatch": "WARN ", "unresolved": "FAIL "}[res["status"]]
        print(f"{mark} {res['key']:<28} {res['status']:<11} {(res['title'] or '')[:60]}")
        for n in res["notes"]:
            print(f"      ↳ {n}")

    bad = [r for r in results if r["status"] != "verified"]
    payload = {
        "schema": "paper-forge/citations-verified@1",
        "bib": os.path.abspath(args.bib),
        "n_entries": len(results),
        "n_verified": len(results) - len(bad),
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")

    print(f"\n{len(results) - len(bad)}/{len(results)} verified.")
    if bad:
        print("Unresolved or mismatched entries must be fixed or removed before submission.")
    return 0 if (args.report_only or not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
