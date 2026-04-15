---
name: LaTeX conversion rules
description: Running list of things to fix when converting thesis markdown (Google Docs export) to LaTeX via pandoc
type: reference
---

# LaTeX Conversion Rules

Things that will not survive the Google Docs → markdown → pandoc → LaTeX pipeline cleanly and need manual intervention.

## 1. Table Alignment

Google Docs exports all columns as left-aligned (`:----`). Numeric columns must be manually changed to right-aligned (`----:`).

**Rule:** After any Docs export, scan all pipe tables. Any column containing numbers (metrics, pixel counts, percentages, run IDs) should be right-aligned. Text columns (model names, notes, borough names) stay left-aligned.

**Example:**
```
# Docs export (wrong for numbers):
| Val F2 | Test F2 |
| :---- | :---- |

# Corrected for LaTeX:
| Val F2 | Test F2 |
| ------: | ------: |
```

## 2. Math Equations

Google Docs equation objects do not export to markdown. They will appear as blank, garbled, or missing in the exported `.md` file.

**Rule:** Any equation written in the Docs equation editor must be manually rewritten in LaTeX math syntax after export. Inline math uses `$...$`; display math uses `$$...$$` or `\begin{equation}...\end{equation}`.

**Known equations in this thesis to watch for:**
- F2 score definition: `$F_2 = \frac{5 \cdot \text{precision} \cdot \text{recall}}{4 \cdot \text{precision} + \text{recall}}$`
- IoU definition: `$\text{IoU} = \frac{TP}{TP + FP + FN}$`
- Composite loss: `$\mathcal{L} = w_{\text{bce}} \cdot \text{BCE}(y, p) + w_{\text{lovász}} \cdot \text{Lovász}(y, p)$`

## 3. [PLACEHOLDER: additional rules as discovered]

Add here as new issues are found during conversion.
