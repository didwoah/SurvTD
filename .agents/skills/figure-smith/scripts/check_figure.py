#!/usr/bin/env python3
"""Lint a figure specification before it is drawn, or a matplotlib figure after.

Reviewers read figures before prose, and the defects that cost a paper are
mechanical: a comparative claim plotted without error bars, an axis with no unit,
a palette that collapses when the review is printed in grayscale. None of those
require taste to catch.

Two modes:

  spec  — lint a JSON figure spec (before drawing anything):
              python3 check_figure.py spec fig1.json

  live  — lint the current matplotlib figure from inside a plotting script:
              from check_figure import lint_current
              lint_current(claim_kind="comparative")

Spec format:

    {
      "id": "fig:1",
      "supports_claim": "C0",
      "claim_kind": "comparative",       // comparative | trend | qualitative | schematic
      "caption": "Phase-stratified credit widens the 128K accuracy gap; uniform broadcast does not.",
      "x": {"label": "Context length", "unit": "tokens"},
      "y": {"label": "Exact match", "unit": "%"},
      "series": ["vanilla GRPO", "PS-GRPO"],
      "error_bars": "std over 5 seeds",
      "palette": ["#4269d0", "#efb118"]
    }

Exit 0 = clean, 1 = at least one hard failure, 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# A caption that names the figure's subject instead of its finding wastes the one
# line every reviewer reads.
DEAD_CAPTION = re.compile(
    r"^\s*(overview|architecture|illustration|diagram|framework|pipeline|"
    r"results?|comparison|performance|our (method|model|approach))\b[\s.:]*$",
    re.I,
)
DEAD_OPENER = re.compile(r"^\s*(overview of|illustration of|architecture of|the pipeline of)\b", re.I)

# Relative luminance gap needed for two series to stay separable in grayscale.
GRAY_GAP = 0.18


def luminance(hex_color: str) -> float | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_color.strip())
    if not m:
        return None
    r, g, b = (int(m.group(1)[i : i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


class Report:
    def __init__(self) -> None:
        self.failed = False

    def fail(self, check: str, detail: str) -> None:
        self.failed = True
        print(f"FAIL  {check}: {detail}")

    def warn(self, check: str, detail: str) -> None:
        print(f"WARN  {check}: {detail}")

    def ok(self, check: str, detail: str = "") -> None:
        print(f"ok    {check}{': ' + detail if detail else ''}")


def lint_spec(spec: dict, r: Report) -> None:
    kind = spec.get("claim_kind", "")
    if not spec.get("supports_claim"):
        r.fail("claim_binding", "the figure supports no claim id — a figure that serves no claim is decoration")
    else:
        r.ok("claim_binding", str(spec["supports_claim"]))

    cap = (spec.get("caption") or "").strip()
    if not cap:
        r.fail("caption", "absent")
    elif DEAD_CAPTION.match(cap) or DEAD_OPENER.match(cap):
        r.fail(
            "caption_takeaway",
            f"{cap!r} names the subject, not the finding — state what the reader should conclude, "
            "because the caption is often the only text a reviewer reads on this page",
        )
    elif len(cap.split()) < 6:
        r.warn("caption_takeaway", "very short; a takeaway caption usually needs a full sentence")
    else:
        r.ok("caption_takeaway")

    if kind in ("comparative", "trend"):
        for axis in ("x", "y"):
            a = spec.get(axis) or {}
            if not (a.get("label") or "").strip():
                r.fail(f"{axis}_axis_label", "missing")
            elif not (a.get("unit") or "").strip():
                r.warn(f"{axis}_axis_unit", f"{a['label']!r} has no unit; state it or say 'unitless'")
            else:
                r.ok(f"{axis}_axis", f"{a['label']} ({a['unit']})")

    if kind == "comparative":
        if not (spec.get("error_bars") or "").strip():
            r.fail(
                "error_bars",
                "a comparative figure with no error bars cannot show that the difference exceeds noise — "
                "add them and state what they are (std over seeds / 95% CI)",
            )
        elif not re.search(r"\b(std|stdev|standard deviation|ci|confidence|sem|iqr|quartile|min.?max)\b",
                           spec["error_bars"], re.I):
            r.warn("error_bars", f"{spec['error_bars']!r} does not say what the bars represent")
        else:
            r.ok("error_bars", spec["error_bars"])
    elif kind:
        r.ok("error_bars", f"not required for claim_kind={kind}")

    palette = spec.get("palette") or []
    series = spec.get("series") or []
    if len(series) > 1 and palette:
        lums = [(c, luminance(c)) for c in palette[: len(series)]]
        bad = [c for c, l in lums if l is None]
        if bad:
            r.warn("palette", f"unparsed color(s) {bad}; use #rrggbb to enable the grayscale check")
        vals = sorted(l for _, l in lums if l is not None)
        gaps = [b - a for a, b in zip(vals, vals[1:])]
        if gaps and min(gaps) < GRAY_GAP:
            r.fail(
                "grayscale_legibility",
                f"two series differ by only {min(gaps):.3f} in luminance (need ≥ {GRAY_GAP}) — "
                "they merge when the review is printed or read by a colorblind reviewer; "
                "vary lightness, or add a marker/dash distinction that does not depend on hue",
            )
        else:
            r.ok("grayscale_legibility", f"min luminance gap {min(gaps):.3f}" if gaps else "single series")
    elif len(series) > 1:
        r.warn("palette", "not declared; the grayscale check cannot run")

    if len(series) > 6:
        r.warn("series_count", f"{len(series)} series in one panel; past ~6 the legend outruns the reader")


def lint_current(claim_kind: str = "comparative", supports_claim: str | None = None) -> bool:
    """Lint the live matplotlib figure. Returns True when clean."""
    import matplotlib.pyplot as plt  # imported lazily so the spec mode stays dependency-free

    fig = plt.gcf()
    r = Report()
    print("check_figure (live matplotlib figure)\n")
    for i, ax in enumerate(fig.get_axes()):
        tag = f"axes[{i}]"
        if not ax.get_xlabel().strip():
            r.fail(f"{tag}.x_axis_label", "missing")
        if not ax.get_ylabel().strip():
            r.fail(f"{tag}.y_axis_label", "missing")
        if claim_kind == "comparative":
            has_bars = any(getattr(c, "has_xerr", False) or getattr(c, "has_yerr", False)
                           for c in ax.containers)
            if not has_bars:
                r.fail(f"{tag}.error_bars", "no ErrorbarContainer on a comparative plot")
        if len(ax.get_lines()) + len(ax.containers) > 1 and ax.get_legend() is None:
            r.warn(f"{tag}.legend", "multiple series without a legend")
    spec = {"claim_kind": claim_kind, "supports_claim": supports_claim, "caption": "(set in LaTeX)",
            "x": {"label": "-", "unit": "-"}, "y": {"label": "-", "unit": "-"}}
    if supports_claim:
        r.ok("claim_binding", supports_claim)
    else:
        r.warn("claim_binding", "no claim id passed; pass supports_claim= to bind this figure to the spine")
    print("\n" + ("VERDICT: fail" if r.failed else "VERDICT: pass"))
    return not r.failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["spec"], help="only `spec` is available from the CLI")
    ap.add_argument("path", help="path to the figure spec JSON")
    args = ap.parse_args()
    try:
        with open(args.path, encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"error: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2
    r = Report()
    print(f"check_figure: {args.path}\n")
    lint_spec(spec, r)
    print()
    print("VERDICT: fail" if r.failed else "VERDICT: pass")
    return 1 if r.failed else 0


if __name__ == "__main__":
    sys.exit(main())
