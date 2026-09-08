#!/usr/bin/env python3
"""
Automated Academic Prose Quality & Anti-AI Linting Tool.
Validates markdown draft sections against top-tier conference standards.
"""

import sys
import re
import os
import json
import argparse
from typing import List, Dict, Tuple

BANNED_CLICHES = [
    r"\bdelv\w*\b",
    r"\bpivotal\w*\b",
    r"\btestament\b",
    r"\bcrucial\b",
    r"\btapestr\w*\b",
    r"\brevolutioniz\w*\b",
    r"\bunleash\w*\b",
    r"\bmeticulous\w*\b",
    r"\bit is worth noting that\b",
    r"\bin this paper,? we propose\b",
    r"\bplays an? (?:important|pivotal|crucial) role\b",
    r"\bgame-?changer\b",
    r"\bgroundbreaking\b",
    r"\bbeacon\b",
    r"\bcornerstone\b",
    r"\bfoster\w*\b",
    r"\bseamlessly\b",
    r"\bunprecedented\b"
]

def split_into_sentences(text: str) -> List[str]:
    # Protect common abbreviations and numbers from sentence split
    protected = text
    abbrevs = [
        ("et al.", "et al<DOT>"),
        ("e.g.", "e<DOT>g<DOT>"),
        ("i.e.", "i<DOT>e<DOT>"),
        ("Fig.", "Fig<DOT>"),
        ("Tab.", "Tab<DOT>"),
        ("Sec.", "Sec<DOT>"),
        ("Eq.", "Eq<DOT>"),
        ("vs.", "vs<DOT>"),
        ("approx.", "approx<DOT>"),
    ]
    for orig, rep in abbrevs:
        protected = re.sub(re.escape(orig), rep, protected, flags=re.IGNORECASE)
    
    # Protect decimal numbers like 0.9538
    protected = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL_DOT>\2', protected)

    # Split on sentence terminals
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)

    cleaned_sentences = []
    for s in raw_sentences:
        # Restore protected tokens
        s_restored = s.replace("<DOT>", ".").replace("<DECIMAL_DOT>", ".")
        s_clean = s_restored.strip()
        if len(s_clean) > 5 and not s_clean.startswith('#'):
            cleaned_sentences.append(s_clean)
    return cleaned_sentences

def analyze_file(filepath: str) -> Dict[str, any]:
    if not os.path.exists(filepath):
        return {"error": f"File {filepath} not found."}

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Banned clichés scan
    cliche_violations = []
    for pattern in BANNED_CLICHES:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            line_no = text[:m.start()].count('\n') + 1
            cliche_violations.append({
                "line": line_no,
                "matched": m.group(0),
                "pattern": pattern
            })

    # 2. Sentence length variance & cadence
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and not p.startswith('#')]
    sentences = split_into_sentences(text)

    word_counts = [len(s.split()) for s in sentences]
    avg_len = sum(word_counts) / max(len(word_counts), 1)

    # Flag monotony: 4+ consecutive sentences with nearly identical length (within 3 words)
    monotony_runs = []
    for i in range(len(word_counts) - 3):
        window = word_counts[i:i+4]
        if max(window) - min(window) <= 3 and min(window) > 10:
            monotony_runs.append((i, window))

    return {
        "file": filepath,
        "total_paragraphs": len(paragraphs),
        "total_sentences": len(sentences),
        "avg_sentence_length": round(avg_len, 1),
        "cliche_violations": cliche_violations,
        "monotony_count": len(monotony_runs),
        "passed": len(cliche_violations) == 0
    }

def main():
    parser = argparse.ArgumentParser(description="Lint academic markdown draft prose.")
    parser.add_argument("files", nargs="+", help="Markdown files to lint")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    results = [analyze_file(f) for f in args.files]

    if args.json:
        print(json.dumps(results, indent=2))
        return

    all_passed = True
    for r in results:
        if "error" in r:
            print(f"[-] {r['error']}")
            all_passed = False
            continue

        print(f"\n==========================================")
        print(f" Prose Audit: {r['file']}")
        print(f"==========================================")
        print(f"Sentences: {r['total_sentences']} | Avg Word Count: {r['avg_sentence_length']} words/sentence")

        if r['cliche_violations']:
            all_passed = False
            print(f"\n[!] Detected {len(r['cliche_violations'])} banned AI clichés:")
            for v in r['cliche_violations']:
                print(f"    Line {v['line']}: Found '{v['matched']}'")
        else:
            print("[+] Zero banned AI clichés detected.")

        if r['monotony_count'] > 0:
            print(f"[?] Notice: {r['monotony_count']} monotonous sentence runs detected. Consider varying cadence.")

    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
