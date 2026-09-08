---
name: latex-smith
description: >-
  Transform markdown drafts into publication-grade, compilation-verified LaTeX manuscripts
  matching official conference templates (ICLR, NeurIPS, ICML) — typeset theorems and proofs,
  format booktabs tables without vertical rules, link bibtex citations without dangling keys,
  and refine sentence rhythm, clause balance, and page budgets. TRIGGER when the user asks to
  convert a markdown draft to LaTeX, format tables for LaTeX, polish paper prose for flow and
  rhythm, prepare camera-ready tex, or fix LaTeX compilation errors. DO NOT TRIGGER for
  generating the original prose (that is paper-composer) or designing figure content (that is figure-smith).
---

# LaTeX Smith

The bridge between a drafted markdown manuscript and a camera-ready, top-tier conference submission.
A great paper can be rejected for sloppy typesetting: bad margins, amateur vertical lines in tables,
overflowing equations, dangling `[?]` references, or monotonous robotic sentence cadence.

LaTeX Smith guarantees typography, mathematical precision, and compilation integrity.

---

## 1. Conference Template & Package Ecosystem

Default targets:
- **ICLR**: `iclr2026_conference.sty`, `\usepackage{times}`
- **NeurIPS**: `neurips_2026.sty`
- **ICML**: `icml2026.sty`

### Mandatory Package Stack

```latex
\usepackage{booktabs}       % Professional publication tables (NO vertical lines)
\usepackage{microtype}      % Sub-pixel typographical margin kerning & protrusion
\usepackage{amsmath, amssymb, amsthm, mathtools} % Mathematical rigor
\usepackage{cleveref}       % Automatic \Cref{thm:...}, \Cref{fig:...} naming
\usepackage{subcaption}     % Subfigures and subtables
\usepackage{xspace}         % Proper macro spacing (e.g. \MethodName\xspace)
\usepackage{nicefrac}       % Compact fractions in text
\usepackage{bm}             % Bold math symbols (\bm{\theta})
```

---

## 2. Mathematical Typesetting Rigor

Never leave math as informal markdown text. Formalize into standard LaTeX environments:

### Theorem & Definition Environments
```latex
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{definition}{Definition}
\newtheorem{assumption}{Assumption}
\newtheorem{remark}{Remark}
```

### Formatting Rules
1. **Equations**: Display equations must use `\begin{align}...\end{align}` with labels `\label{eq:...}` on key results. Use `\notag` on intermediate algebraic steps.
2. **Operators**: Never use raw italics for operators. Define:
   ```latex
   \DeclareMathOperator*{\argmax}{arg\,max}
   \DeclareMathOperator*{\argmin}{arg\,min}
   \newcommand{\E}{\mathbb{E}}
   \newcommand{\Prob}{\mathbb{P}}
   \newcommand{\R}{\mathbb{R}}
   \newcommand{\Loss}{\mathcal{L}}
   ```
3. **Multiplication**: Never use `*` for multiplication. Use `\cdot` for scalar products or juxtaposition for function arguments.
4. **Intervals**: Use `[0, \infty)` rather than `[0, inf)`.

---

## 3. Table Linting & Booktabs Standard

A table with vertical lines (`|`) immediately signals amateur drafting to an ICLR/NeurIPS reviewer.

### The Booktabs Invariants
1. **Zero vertical lines**: Never use `|` in column specs (`{lcccc}`, not `{|l|c|c|c|}`).
2. **Three horizontal rules**:
   - `\toprule` above header.
   - `\midrule` below header.
   - `\bottomrule` at the bottom.
   - Optional `\cmidrule(lr){2-3}` for grouped multi-column headers.
3. **Number Formatting**:
   - Always format standard deviation as `$0.954 \pm 0.027$`.
   - Wrap the best performer in `\mathbf{}`.
   - Align decimals using `S` column type (`siunitx`) or right-align numbers.

### Canonical Masterpiece Table:
```latex
\begin{table}[t]
\centering
\small
\caption{\textbf{Main Empirical Benchmark Comparison.} Evaluation over 5 independent seeds (mean $\pm$ std) under strict tuning parity and zero-leakage protocols. Bold indicates top performance.}
\label{tab:main_benchmark}
\begin{tabular}{llcccc}
\toprule
\textbf{Method} & \textbf{Backbone / Family} & \multicolumn{2}{c}{\textbf{Primary Benchmark Metric}} & \multicolumn{2}{c}{\textbf{Secondary Benchmark Metric}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6}
& & \textbf{Accuracy} $\uparrow$ & \textbf{Error / Loss} $\downarrow$ & \textbf{F1 Score} $\uparrow$ & \textbf{Latency (ms)} $\downarrow$ \\
\midrule
Baseline A & Transformer & $0.865 \pm 0.025$ & $0.072 \pm 0.008$ & $0.858 \pm 0.023$ & $43.4 \pm 1.2$ \\
Baseline B & ResNet & $0.759 \pm 0.030$ & $0.105 \pm 0.011$ & $0.753 \pm 0.033$ & $\mathbf{15.2 \pm 0.5}$ \\
Baseline C & State-Space Model & $0.812 \pm 0.018$ & $0.089 \pm 0.009$ & $0.806 \pm 0.021$ & $24.8 \pm 0.9$ \\
\midrule
\textbf{Ours (Proposed)} & [Novel Mechanism] & $\mathbf{0.954 \pm 0.017}$ & $\mathbf{0.045 \pm 0.006}$ & $\mathbf{0.948 \pm 0.015}$ & $26.1 \pm 1.1$ \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 4. Latex Rhythm Refiner (Prose Polishing)

Absorbed from top academic writing practices: AI-generated text has monotonous sentence lengths (~18-22 words per sentence). Human masterpieces vary sentence cadence dramatically.

### The Rhythm Invariants
1. **The 3-Sentence Cadence**:
   - Sentence 1 (Impact): Short, declarative thesis (8–14 words).
   - Sentence 2 (Analytical Depth): Compound analytical elaboration (22–32 words) detailing mechanism.
   - Sentence 3 (Resolution / Evidence): Medium clincher (14–20 words) pointing to data or theorem.
2. **Transitional Variety**:
   - Ban consecutive paragraph openers starting with *"Moreover,"*, *"Furthermore,"*, *"In addition,"*.
   - Use structural transitions: *"By contrast,"*, *"Crucially,"*, *"This equivalence holds because..."*, *"Under continuous scaling..."*.
3. **Parenthetical vs Textual Citations**:
   - If the authors are the subject: `\citet{vaswani2017attention} introduce the self-attention mechanism...`
   - If the claim is being cited: `...as demonstrated in recent empirical evaluations \citep{lecun2015deep}.`
   - Never produce `\citep{}` where grammar requires `\citet{}` (e.g. avoid *"In (Vaswani et al., 2017), they show"*).

---

## 5. Compilation & Automated Integrity Checks

Always execute deterministic static analysis before running the heavy compiler:

```bash
# 1. Run deterministic static linter (checks booktabs, citations, deprecated math, labels)
python3 scripts/check_latex.py paper/main.tex --bib research/references.bib

# 2. Check for overflow / overfull hboxes after pdflatex
pdflatex -interaction=nonstopmode main.tex | grep -i "overfull"
```

---

## Output Structure

When converting a paper, LaTeX Smith creates:
```
paper/
├── main.tex               # Master document containing preamble & section inputs
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related.tex
│   ├── 03_formulation.tex
│   ├── 04_method.tex
│   ├── 05_experiments.tex
│   └── 06_conclusion.tex
├── figures/               # Vector PDF plots from figure-smith
├── tables/                # Booktabs tex tables
├── references.bib         # Verified bibliography
└── Makefile / latexmkrc   # Reproducible build script
```
