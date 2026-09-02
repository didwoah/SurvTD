#!/usr/bin/env python3
"""Multi-source literature retrieval with zero dependencies and zero API keys.

Sources: arXiv, OpenAlex, Semantic Scholar, Crossref, OpenReview (API2), DBLP.
Everything runs on the Python standard library so the skill installs by copy.

Typical use (one call per retrieval window, see SKILL.md Phase 0):

    python3 lit_search.py \
        --query "verifier-free credit assignment long context RL" \
        --query "phase-level advantage estimation GRPO" \
        --sources arxiv,s2,openreview --from 2026-03 --limit 30 \
        --out phase0/recent.json

    python3 lit_search.py --query ... \
        --sources openalex,s2,crossref --from 2024-09 --to 2026-03 \
        --published-only --limit 30 --out phase0/published.json

Records are deduplicated across queries and sources by DOI, then arXiv id, then
normalized title.  Each surviving record keeps every query that reached it in
``from_query`` so a downstream yield report can tell which query earned its slot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

UA = "research-ideation-skills/1.0 (https://github.com/; mailto:{mail})"
CONTACT = os.environ.get("LIT_SEARCH_MAILTO", "research-skills@example.org")

# Minimum seconds between two requests to the same host.  arXiv rate-limits
# aggressively and answers HTTP 429 with an empty body, which silently zeroes a
# connector, so it gets the widest interval.
MIN_INTERVAL = {
    "arxiv": float(os.environ.get("ARXIV_MIN_INTERVAL", "4.0")),
    # Unauthenticated Semantic Scholar throttles hard and answers 429; with a key it
    # tolerates roughly an order of magnitude more, so the interval follows the key.
    "s2": 1.0 if os.environ.get("S2_API_KEY") else 6.0,
    "openalex": 0.2,
    "crossref": 0.3,
    "openreview": 1.0,
    "dblp": 1.5,
}
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _throttle(source: str) -> None:
    interval = MIN_INTERVAL.get(source, 1.0)
    with _lock:
        prev = _last_call.get(source, 0.0)
        wait = interval - (time.time() - prev)
        if wait > 0:
            time.sleep(wait)
        _last_call[source] = time.time()


def _get(url: str, source: str, retries: int = 2) -> bytes | None:
    """GET with throttling and bounded retry.  Returns None on definitive failure."""
    headers = {"User-Agent": UA.format(mail=CONTACT), "Accept": "application/json"}
    if source == "s2" and os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = os.environ["S2_API_KEY"]
    for attempt in range(retries + 1):
        _throttle(source)
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 429/5xx are transient; anything else will not improve on retry.
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep((10 if exc.code == 429 else 5) * (attempt + 1))
                continue
            hint = " (set S2_API_KEY to raise the keyless rate limit)" if source == "s2" and exc.code == 429 else ""
            _warn(f"{source}: HTTP {exc.code} for {url[:110]}{hint}")
            return None
        except Exception as exc:  # noqa: BLE001 - network layer, report and move on
            if attempt < retries:
                time.sleep(3)
                continue
            _warn(f"{source}: {type(exc).__name__}: {exc}")
            return None
    return None


def _warn(msg: str) -> None:
    print(f"warn: {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# normalization helpers
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def norm_title(title: str) -> str:
    return _WS.sub(" ", _NON_ALNUM.sub(" ", (title or "").lower())).strip()


def clean(text: str | None, limit: int = 1800) -> str:
    if not text:
        return ""
    text = _WS.sub(" ", str(text)).strip()
    return text[:limit]


def month_key(value: str | None) -> str:
    """Coerce assorted date shapes into YYYY-MM (empty string when unknown)."""
    if not value:
        return ""
    m = re.match(r"(\d{4})[-/]?(\d{2})?", str(value))
    if not m:
        return ""
    return f"{m.group(1)}-{m.group(2)}" if m.group(2) else m.group(1)


def in_window(ym: str, lo: str | None, hi: str | None) -> bool:
    if not ym:
        return not lo  # undated records only survive an open-ended lower bound
    if lo and ym < lo:
        return False
    if hi and ym > hi:
        return False
    return True


def record(**kw) -> dict:
    base = {
        "paper_id": "",
        "title": "",
        "abstract": "",
        "year_month": "",
        "venue": "",
        "authors": [],
        "url": "",
        "doi": "",
        "arxiv_id": "",
        "citation_count": None,
        "source": "",
        "from_query": [],
        "is_preprint": False,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# connectors
# --------------------------------------------------------------------------- #


def _arxiv_daterange(lo: str | None, hi: str | None) -> str:
    """arXiv wants submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]."""
    if not lo and not hi:
        return ""
    start = f"{lo.replace('-', '')}010000" if lo else "199101010000"
    if hi:
        y, m = hi.split("-")[0], (hi.split("-") + ["12"])[1]
        end = f"{y}{m}312359"
    else:
        end = datetime.now(timezone.utc).strftime("%Y%m%d2359")
    return f"submittedDate:[{start} TO {end}]"


def _arxiv_fetch(search_query: str, limit: int) -> list | None:
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit * 3, 120),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    raw = _get(f"http://export.arxiv.org/api/query?{params}", "arxiv")
    if not raw:
        return None
    try:
        return list(ET.fromstring(raw).findall("{http://www.w3.org/2005/Atom}entry"))
    except ET.ParseError as exc:
        _warn(f"arxiv: unparseable feed ({exc})")
        return None


def search_arxiv(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    # A bare `all:<words>` query ORs the terms, and sorting that by date returns the
    # newest paper containing ANY of them — which is how a search for long-context RL
    # comes back with germanium spin relaxation. Quote the phrase, constrain the date
    # server-side, and rank by relevance instead.
    window = _arxiv_daterange(lo, hi)
    terms = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", query)
    # Longest words are the most discriminative ones a query owns; short ones
    # ("deep", "model", "data") are carried by every paper in the field.
    core = sorted(terms, key=len, reverse=True)[:3]

    # A recall ladder, strictest first. A single AND-of-every-term fallback is barely
    # weaker than the phrase itself, so a phrase miss used to return nothing at all.
    ladder = [f'all:"{query}"']
    if terms:
        ladder.append(" AND ".join(f"all:{t}" for t in terms))
    if len(core) > 1 and len(core) < len(terms):
        ladder.append(" AND ".join(f"all:{t}" for t in core))
    if len(terms) > 1:
        ladder.append(" OR ".join(f"all:{t}" for t in terms))

    entries = None
    for i, q in enumerate(ladder):
        entries = _arxiv_fetch(f"{q} AND {window}" if window else q, limit)
        if entries:
            if i:
                _warn(f"arxiv: phrase query missed; recovered at ladder rung {i + 1}/{len(ladder)}")
            break
    if not entries:
        return []

    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in entries:
        eid = (entry.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        published = entry.findtext("a:published", "", ns)
        ym = month_key(published)
        if not in_window(ym, lo, hi):
            continue
        doi = entry.findtext("{http://arxiv.org/schemas/atom}doi", "", ns) or ""
        out.append(
            record(
                paper_id=f"arxiv:{eid}",
                title=clean(entry.findtext("a:title", "", ns), 400),
                abstract=clean(entry.findtext("a:summary", "", ns)),
                year_month=ym,
                venue="arXiv preprint",
                authors=[
                    clean(a.findtext("a:name", "", ns), 80)
                    for a in entry.findall("a:author", ns)[:12]
                ],
                url=f"https://arxiv.org/abs/{eid}",
                doi=doi,
                arxiv_id=eid,
                source="arxiv",
                is_preprint=True,
            )
        )
        if len(out) >= limit:
            break
    return out


def search_openalex(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    # The top-level `search` param matches loosely across 250M works, which is how a
    # long-context-RL query returns valvular heart disease guidelines. The
    # title_and_abstract.search FILTER is the stricter entry point and confines the
    # match to where a paper's actual subject lives.
    filters = [
        "type:article",
        f"title_and_abstract.search:{query.replace(',', ' ')}",
    ]
    if lo:
        filters.append(f"from_publication_date:{lo}-01")
    if hi:
        filters.append(f"to_publication_date:{hi}-28")
    params = urllib.parse.urlencode(
        {
            "filter": ",".join(filters),
            "per-page": min(limit, 50),
            "sort": "relevance_score:desc",
            "mailto": CONTACT,
        }
    )
    raw = _get(f"https://api.openalex.org/works?{params}", "openalex")
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for w in payload.get("results", []):
        # OpenAlex ships abstracts as an inverted index; rebuild reading order.
        inv = w.get("abstract_inverted_index") or {}
        abstract = ""
        if inv:
            slots: list[tuple[int, str]] = []
            for word, positions in inv.items():
                slots.extend((p, word) for p in positions)
            abstract = " ".join(w for _, w in sorted(slots))
        loc = (w.get("primary_location") or {}).get("source") or {}
        out.append(
            record(
                paper_id=f"openalex:{(w.get('id') or '').rsplit('/', 1)[-1]}",
                title=clean(w.get("title") or w.get("display_name"), 400),
                abstract=clean(abstract),
                year_month=month_key(w.get("publication_date")),
                venue=clean(loc.get("display_name") or "", 160),
                authors=[
                    clean((a.get("author") or {}).get("display_name"), 80)
                    for a in (w.get("authorships") or [])[:12]
                ],
                url=(w.get("primary_location") or {}).get("landing_page_url")
                or w.get("id", ""),
                doi=(w.get("doi") or "").replace("https://doi.org/", ""),
                citation_count=w.get("cited_by_count"),
                source="openalex",
                is_preprint=(loc.get("type") == "repository"),
            )
        )
    return out


def search_s2(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    fields = "title,abstract,year,publicationDate,venue,authors,externalIds,url,citationCount,publicationTypes"
    params = {"query": query, "limit": min(limit, 100), "fields": fields}
    if lo or hi:
        params["publicationDateOrYear"] = f"{lo or ''}:{hi or ''}"
    raw = _get(
        "https://api.semanticscholar.org/graph/v1/paper/search?"
        + urllib.parse.urlencode(params),
        "s2",
    )
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    out = []
    for p in payload.get("data", []) or []:
        ext = p.get("externalIds") or {}
        ym = month_key(p.get("publicationDate")) or str(p.get("year") or "")
        if not in_window(ym, lo, hi):
            continue
        out.append(
            record(
                paper_id=f"s2:{p.get('paperId', '')}",
                title=clean(p.get("title"), 400),
                abstract=clean(p.get("abstract")),
                year_month=ym,
                venue=clean(p.get("venue"), 160) or "arXiv preprint",
                authors=[clean(a.get("name"), 80) for a in (p.get("authors") or [])[:12]],
                url=p.get("url", ""),
                doi=ext.get("DOI", "") or "",
                arxiv_id=ext.get("ArXiv", "") or "",
                citation_count=p.get("citationCount"),
                source="s2",
                is_preprint=not clean(p.get("venue")),
            )
        )
    return out


def search_crossref(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    params = {"query.bibliographic": query, "rows": min(limit, 50), "mailto": CONTACT}
    if lo:
        params["filter"] = f"from-pub-date:{lo}-01"
        if hi:
            params["filter"] += f",until-pub-date:{hi}-28"
    raw = _get("https://api.crossref.org/works?" + urllib.parse.urlencode(params), "crossref")
    if not raw:
        return []
    try:
        items = json.loads(raw).get("message", {}).get("items", [])
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        parts = (it.get("issued") or {}).get("date-parts") or [[]]
        dp = parts[0] if parts and parts[0] else []
        ym = f"{dp[0]}-{dp[1]:02d}" if len(dp) >= 2 else (str(dp[0]) if dp else "")
        titles = it.get("title") or []
        out.append(
            record(
                paper_id=f"doi:{it.get('DOI', '')}",
                title=clean(titles[0] if titles else "", 400),
                abstract=clean(re.sub(r"<[^>]+>", " ", it.get("abstract", "") or "")),
                year_month=ym,
                venue=clean((it.get("container-title") or [""])[0], 160),
                authors=[
                    clean(f"{a.get('given', '')} {a.get('family', '')}".strip(), 80)
                    for a in (it.get("author") or [])[:12]
                ],
                url=it.get("URL", ""),
                doi=it.get("DOI", "") or "",
                citation_count=it.get("is-referenced-by-count"),
                source="crossref",
            )
        )
    return out


def search_openreview(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    params = urllib.parse.urlencode(
        {"term": query, "content": "all", "group": "all", "source": "all", "limit": min(limit, 50)}
    )
    raw = _get(f"https://api2.openreview.net/notes/search?{params}", "openreview")
    if not raw:
        return []
    try:
        notes = json.loads(raw).get("notes", [])
    except json.JSONDecodeError:
        return []
    out = []
    for n in notes:
        c = n.get("content") or {}

        def val(key: str) -> str:
            v = c.get(key)
            return v.get("value", "") if isinstance(v, dict) else (v or "")

        ts = n.get("cdate") or n.get("tcdate")
        ym = (
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m")
            if ts
            else ""
        )
        if not in_window(ym, lo, hi):
            continue
        out.append(
            record(
                paper_id=f"openreview:{n.get('id', '')}",
                title=clean(val("title"), 400),
                abstract=clean(val("abstract")),
                year_month=ym,
                venue=clean(str(n.get("invitations") or n.get("invitation") or ""), 160),
                authors=[clean(a, 80) for a in (c.get("authors", {}) or {}).get("value", [])[:12]]
                if isinstance(c.get("authors"), dict)
                else [],
                url=f"https://openreview.net/forum?id={n.get('id', '')}",
                source="openreview",
                is_preprint=True,
            )
        )
    return out


def search_dblp(query: str, limit: int, lo: str | None, hi: str | None) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "h": min(limit, 50)})
    raw = _get(f"https://dblp.org/search/publ/api?{params}", "dblp")
    if not raw:
        return []
    try:
        hits = json.loads(raw).get("result", {}).get("hits", {}).get("hit", [])
    except json.JSONDecodeError:
        return []
    out = []
    for h in hits:
        info = h.get("info", {})
        ym = str(info.get("year", "") or "")
        if not in_window(ym, lo, hi):
            continue
        authors = (info.get("authors") or {}).get("author", [])
        if isinstance(authors, dict):
            authors = [authors]
        out.append(
            record(
                paper_id=f"dblp:{h.get('@id', '')}",
                title=clean(info.get("title"), 400),
                year_month=ym,
                venue=clean(info.get("venue"), 160),
                authors=[clean(a.get("text") if isinstance(a, dict) else a, 80) for a in authors[:12]],
                url=info.get("ee") or info.get("url", ""),
                doi=info.get("doi", "") or "",
                source="dblp",
            )
        )
    return out


CONNECTORS = {
    "arxiv": search_arxiv,
    "openalex": search_openalex,
    "s2": search_s2,
    "crossref": search_crossref,
    "openreview": search_openreview,
    "dblp": search_dblp,
}

# Surveys dilute a gap corpus: they restate a field instead of leaving a residue.
# They are demoted rather than dropped, because a survey is still a citeable map.
SURVEY_RE = re.compile(r"\b(survey|a review of|systematic review|tutorial|overview of)\b", re.I)

STOPWORDS = {
    "the", "and", "for", "with", "via", "using", "from", "into", "onto", "over", "under",
    "based", "toward", "towards", "through", "across", "about", "that", "this", "these",
    "their", "than", "then", "when", "where", "which", "while", "have", "has", "are",
    "was", "were", "been", "its", "our", "new", "novel", "approach", "method", "methods",
    "framework", "model", "models", "learning", "neural", "deep",
}


def content_terms(query: str) -> set[str]:
    """The query's discriminative words — 'learning' and 'model' are dropped because
    every ML paper contains them and they cannot separate on-topic from off-topic."""
    return {t for t in re.findall(r"[a-z][a-z0-9-]{2,}", query.lower()) if t not in STOPWORDS}


def term_overlap(rec: dict, terms: set[str]) -> float:
    if not terms:
        return 1.0
    hay = f"{rec.get('title', '')} {rec.get('abstract', '')}".lower()
    return sum(1 for t in terms if t in hay) / len(terms)


def dedup(records: list[dict]) -> list[dict]:
    """Collapse duplicates by DOI, then arXiv id, then normalized title.

    Later records enrich the surviving one (a Semantic Scholar hit often carries
    the venue and citation count an arXiv hit lacks) instead of replacing it.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for r in records:
        keys = []
        if r.get("doi"):
            keys.append(f"doi:{r['doi'].lower()}")
        if r.get("arxiv_id"):
            keys.append(f"arxiv:{r['arxiv_id'].split('v')[0]}")
        nt = norm_title(r.get("title", ""))
        if nt:
            keys.append(f"title:{nt}")
        hit = next((k for k in keys if k in merged), None)
        if hit is None:
            key = keys[0] if keys else r["paper_id"]
            for k in keys:
                merged[k] = r
            merged.setdefault(key, r)
            order.append(key)
            continue
        kept = merged[hit]
        for field in ("abstract", "venue", "doi", "arxiv_id", "url", "year_month"):
            if not kept.get(field) and r.get(field):
                kept[field] = r[field]
        if kept.get("citation_count") is None:
            kept["citation_count"] = r.get("citation_count")
        for q in r.get("from_query", []):
            if q not in kept["from_query"]:
                kept["from_query"].append(q)
        if r["source"] not in kept["source"].split("+"):
            kept["source"] = f"{kept['source']}+{r['source']}"
        for k in keys:
            merged.setdefault(k, kept)
    seen: set[int] = set()
    out = []
    for key in order:
        rec = merged[key]
        if id(rec) in seen:
            continue
        seen.add(id(rec))
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", action="append", required=True, help="repeatable; one per search angle")
    ap.add_argument("--sources", default="arxiv,openalex,s2", help="comma list of: " + ",".join(CONNECTORS))
    ap.add_argument("--from", dest="lo", default=None, metavar="YYYY-MM", help="lower bound, inclusive")
    ap.add_argument("--to", dest="hi", default=None, metavar="YYYY-MM", help="upper bound, inclusive")
    ap.add_argument("--limit", type=int, default=25, help="cap per (query, source) pair")
    ap.add_argument("--published-only", action="store_true", help="drop records flagged as preprints")
    ap.add_argument("--min-overlap", type=float, default=0.34, metavar="F",
                    help="drop records sharing less than this fraction of the query's content "
                         "terms (default 0.34; set 0 to keep everything a connector returned)")
    ap.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    ap.add_argument("--markdown", default=None, help="also write a review table here")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in CONNECTORS]
    if unknown:
        print(f"error: unknown source(s) {unknown}; known: {list(CONNECTORS)}", file=sys.stderr)
        return 2

    collected: list[dict] = []
    stats: dict[str, int] = {}
    for query in args.query:
        for src in sources:
            hits = CONNECTORS[src](query, args.limit, args.lo, args.hi)
            for h in hits:
                h["from_query"] = [query]
            stats[f"{src}"] = stats.get(src, 0) + len(hits)
            collected.extend(hits)

    records = dedup(collected)
    if args.published_only:
        records = [r for r in records if not r["is_preprint"]]

    # Source-agnostic precision gate. Every connector's lexical matching is weak in its
    # own way, so rather than tuning six query dialects, score each surviving record
    # against the query that actually reached it and drop the ones that share almost
    # nothing with it.
    term_sets = {q: content_terms(q) for q in args.query}
    dropped = []
    kept = []
    for r in records:
        best = max((term_overlap(r, term_sets[q]) for q in r["from_query"] if q in term_sets), default=1.0)
        r["query_overlap"] = round(best, 3)
        (kept if best >= args.min_overlap else dropped).append(r)
    records = kept

    # Keep surveys, but sink them so per-source caps downstream are not spent on them.
    records.sort(key=lambda r: (bool(SURVEY_RE.search(r["title"])), r["year_month"] == "", -r["query_overlap"]))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "queries": args.query,
        "sources": sources,
        "window": {"from": args.lo, "to": args.hi},
        "per_source_hits": stats,
        "n_raw": len(collected),
        "n_deduped": len(records) + len(dropped),
        "n_kept": len(records),
        "min_overlap": args.min_overlap,
        "dropped_off_topic": [
            {"title": d["title"], "source": d["source"], "query_overlap": d["query_overlap"]} for d in dropped
        ],
        "records": records,
    }

    blob = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(blob)
        print(f"ok: {len(records)} on-topic records "
              f"(raw {len(collected)} → deduped {len(records) + len(dropped)} → "
              f"kept {len(records)}, dropped {len(dropped)} below overlap {args.min_overlap}) -> {args.out}")
        print(f"    per-source: {stats}")
    else:
        print(blob)

    if args.markdown:
        lines = ["| paper_id | date | venue | title | cites |", "|---|---|---|---|---|"]
        for r in records:
            title = r["title"].replace("|", "/")
            lines.append(
                f"| `{r['paper_id']}` | {r['year_month'] or '?'} | {r['venue'] or '?'} "
                f"| {title} | {r['citation_count'] if r['citation_count'] is not None else '-'} |"
            )
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".", exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"ok: table -> {args.markdown}")

    if not records:
        print("warn: zero records — widen the window, or check connector reachability", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
