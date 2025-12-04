# WWW'26 Submission Package for CaseSentinel

This folder collects the \LaTeX{} sources for the CaseSentinel paper prepared for The Web Conference (WWW) 2026.

## Contents

- `www26-crimementor.tex` – main manuscript using the ACM `acmart` class (anonymous review format).
- `references.bib` – bibliography database in Bib\TeX{} format.

## Prerequisites

Ensure the ACM primary class files are available. The repository ships the official `acmart` sources under `../acmart-primary/`. Copy or symlink that directory next to the manuscript, or install `acmart` via your TeX distribution.

Recommended toolchain:

- TeX Live 2024 or later (includes `latexmk`, `pdflatex`, `biblatex` support)
- `biber` (if you switch to biblatex) – not required for the default Bib\TeX{} workflow.

## Build Instructions

```bash
cd paper
latexmk -pdf www26-crimementor.tex
```

The command produces `www26-crimementor.pdf`. To clean auxiliary files:

```bash
latexmk -c www26-crimementor.tex
```

## Notes

- The manuscript currently uses the `anonymous,review` options; remove these for the camera-ready version.
- Update the ACM metadata (`\acmISBN`, `\acmDOI`, `\acmSubmissionID`) when the official values are assigned.
- Figures are rendered with TikZ for portability; no external image assets are required.
- Extend `references.bib` as new citations are added.
