#!/usr/bin/env python3
"""Camera-ready matplotlib defaults for ML conference papers.

    from plot_style import use, PALETTE, savefig
    use(column="single")            # or "double" for a full-width figure
    ...
    savefig("figures/fig1.pdf")

The settings are chosen for how a paper is actually consumed: fonts that stay
legible when a 3.3-inch column figure is read at 100%, a palette that survives
grayscale printing and the common colorblindness types, and vector output so the
camera-ready does not ship a blurry raster.

Run this file directly to render a self-test figure exercising every element.
"""

from __future__ import annotations

# Okabe–Ito, reordered so adjacent entries also differ in lightness. That second
# property is what keeps two series apart in a printed review, where hue is gone.
PALETTE = [
    "#0072B2",  # blue          L≈0.19
    "#E69F00",  # orange        L≈0.47
    "#009E73",  # bluish green  L≈0.28
    "#F0E442",  # yellow        L≈0.77
    "#D55E00",  # vermillion    L≈0.24
    "#56B4E9",  # sky blue      L≈0.45
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Hue is not the only channel. Anything comparative should also vary these, so the
# figure still reads in grayscale and for a colorblind reviewer.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = [(None, None), (4, 1.5), (1, 1.2), (6, 1.5, 1, 1.5), (3, 1, 3, 1)]

# NeurIPS/ICML/ICLR text width, in inches.
WIDTH = {"single": 3.3, "double": 6.9}


def use(column: str = "single", height_ratio: float = 0.66, base_font: float = 8.0) -> None:
    """Install the style. `column` sizes the figure to the venue text block."""
    import matplotlib as mpl

    w = WIDTH.get(column, WIDTH["single"])
    mpl.rcParams.update(
        {
            "figure.figsize": (w, w * height_ratio),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Type-42 keeps text selectable and searchable; several venues reject
            # Type-3 fonts outright at camera-ready.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": base_font,
            "axes.titlesize": base_font + 1,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
            "legend.fontsize": base_font - 1,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "errorbar.capsize": 2.0,
        }
    )


def series_style(i: int) -> dict:
    """Color + marker + dash for series `i`, so it is identifiable without color."""
    dash = DASHES[i % len(DASHES)]
    style = {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)]}
    if dash != (None, None):
        style["dashes"] = dash
    return style


def savefig(path: str, **kw) -> None:
    """Save as vector, and refuse the silent mistake of a raster camera-ready."""
    import matplotlib.pyplot as plt

    if not path.lower().endswith((".pdf", ".svg", ".eps")):
        raise ValueError(
            f"{path!r} is a raster format. Ship figures as PDF so they stay sharp at any zoom; "
            "a reviewer who cannot read your axis labels does not squint, they complain."
        )
    plt.savefig(path, **kw)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use(column="single")
    x = [16, 32, 64, 128]
    runs = {
        "vanilla GRPO": ([62.6, 55.1, 48.0, 41.2], [1.0, 1.1, 0.9, 0.5]),
        "PS-GRPO": ([62.9, 57.8, 52.4, 44.9], [0.4, 0.7, 0.6, 0.5]),
    }
    fig, ax = plt.subplots()
    for i, (name, (mean, err)) in enumerate(runs.items()):
        ax.errorbar(x, mean, yerr=err, label=name, **series_style(i))
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{v}K" for v in x])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Exact match (%)")
    ax.legend()
    savefig("plot_style_selftest.pdf")
    print("ok: wrote plot_style_selftest.pdf")
