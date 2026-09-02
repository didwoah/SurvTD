---
name: figure-smith
description: >-
  Design and lint the figures and tables in a paper — a Figure 1 that renders the claim rather
  than the architecture, takeaway captions, mandatory error bars on comparative claims,
  grayscale- and colorblind-legible palettes, and camera-ready plot settings. TRIGGER when the
  user asks what Figure 1 should show, how to design or improve a figure or results table, what
  plot settings to use for a paper, or to check a figure before submission. DO NOT TRIGGER for
  general-purpose charts and dashboards outside a paper context.
---

# Figure Smith

Reviewers read figures before prose. The defects that cost papers here are mechanical — a
comparative claim plotted without error bars, an axis with no unit, a palette that collapses
when the review is printed — and none of them require taste to catch.

## 1. Every figure serves exactly one claim

Before drawing anything, write the spec:

```json
{
  "id": "fig:1",
  "supports_claim": "C0",
  "claim_kind": "comparative",
  "caption": "Phase-stratified credit widens the accuracy gap over vanilla GRPO as context grows to 128K; uniform broadcast stays flat.",
  "x": {"label": "Context length", "unit": "tokens"},
  "y": {"label": "Exact match", "unit": "%"},
  "series": ["vanilla GRPO", "PS-GRPO"],
  "error_bars": "std over 5 seeds",
  "palette": ["#0072B2", "#E69F00"]
}
```

```bash
python3 "$SKILL_DIR/scripts/check_figure.py" spec fig1.json
```

`claim_kind` is `comparative` (A vs B), `trend` (behavior over a parameter), `qualitative`
(examples), or `schematic` (a mechanism diagram). It determines which checks apply.

A figure that supports no claim id is decoration. Cut it, or find the claim it makes.

## 2. Figure 1 renders the claim, not the architecture

The most common wasted page in an ML paper is a system diagram as Figure 1. It shows what you
built; the reader is deciding whether to care about what you found.

**Figure 1 should be the picture of `C0` being true.** For a comparative claim that is usually
the gap. For a mechanism claim it is often the negative control sitting beside the main result —
the strongest single image a mechanistic paper can lead with, because it shows the effect
appearing and disappearing on demand.

The architecture diagram belongs in the method section, where a reader who has decided to care
is ready for it.

## 3. The caption states the takeaway

| ✗ | ✓ |
|---|---|
| Overview of our method | Redistributing credit onto pivotal phases widens the 128K gap; uniform broadcast is flat |
| Results on RULER | The gain appears only past 32K, and vanishes under label permutation |
| Comparison with baselines | At matched tuning budget, the strongest baseline closes half the gap at 16K and none at 128K |

The caption is often the only text a reviewer reads on that page. A caption naming the subject
instead of the finding wastes it. Write the sentence you want quoted in the meta-review.

## 4. Error bars are mandatory on comparative claims

No bars means the figure cannot show the difference exceeds noise, which means it does not
support a comparative claim — whatever the means look like.

State **what the bars are** in the caption: std over N seeds, 95% CI, min–max, IQR. "Error
bars" alone is not a definition, and reviewers ask.

If the interval overlaps and you are still claiming a difference, `evidence-auditor` should have
caught it first; fix the claim, not the figure.

## 5. It must survive grayscale and colorblindness

Reviews get printed. Roughly 8% of male reviewers have a color vision deficiency. A figure
readable only in color is a figure some of your reviewers cannot read.

- Use the Okabe–Ito palette in `scripts/plot_style.py`. It is ordered so adjacent entries also
  differ in **lightness**, which is what survives grayscale.
- **Vary a second channel** — marker shape, dash pattern — so identity never depends on hue
  alone. `plot_style.series_style(i)` returns color + marker + dash together.
- `check_figure.py` fails a palette whose series differ by less than 0.18 in relative luminance.

## 6. Camera-ready settings

```python
from plot_style import use, series_style, savefig
use(column="single")        # 3.3in, sized to the venue text block; "double" for full width
...
savefig("figures/fig1.pdf") # refuses raster formats
```

- **Vector output.** PDF, not PNG. A raster camera-ready is blurry at the zoom reviewers use.
- **Type-42 fonts.** Several venues reject Type-3 outright.
- **Font size ≥ 8pt at final size.** Check by viewing the PDF at 100% next to the paper, not by
  zooming into the plot window.
- Draw at final size. Scaling a figure down after the fact shrinks its text below legibility,
  and this is the single most common camera-ready defect.

## 7. Tables

- **± always**, and define what follows it.
- **Bold only when the difference exceeds noise.** Bolding a within-noise best number is a
  quiet overclaim, and a reviewer who checks the seed count will find it.
- **A compute column** whenever methods are compared. A method that wins at 4× the compute has
  not won, and stating the cost yourself is far better than being asked.
- **Mark what is copied from another paper** versus what you ran. Numbers taken from a paper
  were produced under that paper's conditions.
- Group rows by lineage, not alphabetically.

---

## Output

Per figure: the spec JSON, a clean `check_figure.py` run, and the plotting code using
`plot_style`. For live linting inside a plotting script:

```python
from check_figure import lint_current
lint_current(claim_kind="comparative", supports_claim="C0")
```

## Rules

- **No figure without a claim id.**
- **No comparative figure without error bars.**
- **No caption that names its subject instead of its finding.**
- **No raster in the camera-ready.**
