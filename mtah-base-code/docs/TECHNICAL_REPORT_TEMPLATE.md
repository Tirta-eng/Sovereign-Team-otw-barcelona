# Technical report — More Than A Hack 2027

## Approach
Describe the baseline and the supervised model, including features and smoothing window.

## Assets used
List ATD metadata and raw tracking. State whether Smart Tagging, homography, tactical video, or broadcast video were used or intentionally excluded.

## Possession rules
Explain how the model distinguishes `H`, `A`, `D`, and `X`, especially during missing-ball frames, loose balls, restarts, and stoppages.

## Evaluation
Report the official validator result and the public `spellF1` and `macroF1`. Compare against the supplied baseline. Do not present the public five-minute score as a reliable full-match estimate.

## Limitations
Discuss the small, contiguous public labelled window; tracking noise; missing ball observations; and any known failure modes.

## Reproduction
Give the exact commands and dependency versions required to reproduce the CSV.
