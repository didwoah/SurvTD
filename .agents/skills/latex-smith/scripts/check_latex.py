#!/usr/bin/env python3
"""
Automated LaTeX Quality & Typography Linting Tool.
Validates LaTeX manuscripts against ICLR/NeurIPS publication guidelines.
"""

import sys
import re
import os
import argparse
from typing import Dict, List, Set

def audit_latex(tex_file: str, bib_file: str = "research/references.bib") -> Dict[str, any]:
    if not os.path.exists(tex_file):
        return {"error": f"File {tex_file} not found."}

    # Resolve bib_file if default doesn't exist but alternatives do
    if not os.path.exists(bib_file):
        for candidate in ["paper/references.bib", "references.bib", "../references.bib"]:
            if os.path.exists(candidate):
                bib_file = candidate
                break

    if not os.path.exists(bib_file):
        return {"error": f"Bibliography file '{bib_file}' not found. Cannot verify citations."}

    with open(tex_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Check for amateur vertical rules in tabular/tabularx/tabular*
    vert_rule_violations = []
    for m in re.finditer(r'\\begin\{(?:tabular\*?|tabularx)\}', text):
        line_no = text[:m.start()].count('\n') + 1
        # Extract arguments on the same line or following lines up to first \n\n or \hline
        rest_of_line = text[m.end():].split('\n')[0]
        # Check all {..} groups in this declaration
        brace_contents = re.findall(r'\{([^}]+)\}', rest_of_line)
        for b in brace_contents:
            if '|' in b:
                vert_rule_violations.append({
                    "line": line_no,
                    "col_spec": b.strip()
                })
                break

    # 2. Check for deprecated math environments (eqnarray, $$)
    deprecated_math_violations = []
    for m in re.finditer(r'\\begin\{eqnarray\*?\}', text):
        line_no = text[:m.start()].count('\n') + 1
        deprecated_math_violations.append({
            "line": line_no,
            "type": "eqnarray (Use amsmath 'align' instead)"
        })
    for m in re.finditer(r'(?<!\\)\$\$.*?\$\$', text, flags=re.DOTALL):
        line_no = text[:m.start()].count('\n') + 1
        deprecated_math_violations.append({
            "line": line_no,
            "type": "$$...$$ (Use '\\[ ... \\]' or 'equation' environment instead)"
        })

    # 3. Check for missing / dangling citations against .bib
    cited_keys = set()
    for m in re.finditer(r'\\cite[tp]?\*?\{([^}]+)\}', text):
        keys = [k.strip() for k in m.group(1).split(',')]
        cited_keys.update(keys)

    missing_bib_keys = set()
    with open(bib_file, 'r', encoding='utf-8') as f:
        bib_text = f.read()
    defined_keys = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', bib_text))
    missing_bib_keys = cited_keys - defined_keys

    # 4. Check for undefined references (\ref, \Cref, \eqref vs \label)
    labels = set(re.findall(r'\\label\{([^}]+)\}', text))
    refs = set(re.findall(r'\\(?:Cref|ref|eqref)\{([^}]+)\}', text))
    missing_labels = refs - labels

    passed = (
        len(vert_rule_violations) == 0 and
        len(missing_bib_keys) == 0 and
        len(missing_labels) == 0 and
        len(deprecated_math_violations) == 0
    )

    return {
        "file": tex_file,
        "bib_file": bib_file,
        "vertical_rule_violations": vert_rule_violations,
        "deprecated_math_violations": deprecated_math_violations,
        "total_citations": len(cited_keys),
        "missing_bib_keys": list(missing_bib_keys),
        "missing_labels": list(missing_labels),
        "passed": passed
    }

def main():
    parser = argparse.ArgumentParser(description="Lint LaTeX source for conference submission standards.")
    parser.add_argument("tex_file", help="Path to main.tex or section.tex")
    parser.add_argument("--bib", default="research/references.bib", help="Path to references.bib")
    args = parser.parse_args()

    r = audit_latex(args.tex_file, args.bib)
    if "error" in r:
        print(f"[-] {r['error']}")
        sys.exit(1)

    print(f"\n==========================================")
    print(f" LaTeX Typography & Integrity Audit: {r['file']}")
    print(f"==========================================")

    if r['vertical_rule_violations']:
        print(f"\n[!] FAIL: Found {len(r['vertical_rule_violations'])} table(s) with amateur vertical lines ('|'):")
        for v in r['vertical_rule_violations']:
            print(f"    Line {v['line']}: spec '{v['col_spec']}' (Use booktabs \\toprule, \\midrule, \\bottomrule instead!)")
    else:
        print("[+] PASS: Zero vertical table lines detected (Booktabs standard satisfied).")

    if r['missing_bib_keys']:
        print(f"\n[!] FAIL: {len(r['missing_bib_keys'])} dangling citations not found in {args.bib}:")
        for k in r['missing_bib_keys']:
            print(f"    - '{k}' (will produce [?] in PDF)")
    else:
        print(f"[+] PASS: All {r['total_citations']} citations successfully resolved in {args.bib}.")

    if r['deprecated_math_violations']:
        print(f"\n[!] FAIL: Found {len(r['deprecated_math_violations'])} deprecated math environment(s):")
        for d in r['deprecated_math_violations']:
            print(f"    Line {d['line']}: {d['type']}")
    else:
        print("[+] PASS: No deprecated math environments (eqnarray, $$).")

    if r['missing_labels']:
        print(f"\n[!] FAIL: Found {len(r['missing_labels'])} unresolved reference labels:")
        for l in r['missing_labels']:
            print(f"    - '{l}' (will produce '??' in PDF)")
    else:
        print("[+] PASS: All cross-references (\\ref, \\Cref, \\eqref) resolved.")

    sys.exit(0 if r['passed'] else 1)

if __name__ == "__main__":
    main()
