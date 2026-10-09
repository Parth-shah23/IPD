## H1 — ECPE Transfer to Journal Text

**Null hypothesis (H₀):** An ECPE model trained on a public English news benchmark achieves the same pair F1-score on unseen journal entries as on the benchmark's own test set.

**Alternative hypothesis (H₁):** An ECPE model trained on a public English news benchmark achieves a lower pair F1-score on unseen journal entries than on the benchmark's own test set.

**Reported results:** News test pair F1-score **51.3%** · Journal pair F1-score **1.1%** · Difference **50.1 points** (95% CI 43.7–56.6) · p < 0.001 → **H₀ rejected**  
*Benchmark: GoodNewsEveryone headlines; journal side uses the 350 synthetic entries. Part of the gap comes from labelling conventions (news marks single trigger words), but it persists when any span overlap counts as a match (64.8% vs 10.0%, p < 0.001).*

## H2 — ECPE Adaptation with Journal Data

**Null hypothesis (H₀):** Fine-tuning on journal-style data does not increase pair F1-score over the off-the-shelf model on unseen journal entries.

**Alternative hypothesis (H₁):** Fine-tuning on journal-style data increases pair F1-score over the off-the-shelf model on unseen journal entries.

**Reported results:** Precision **59.9%** · Recall **67.2%** · Pair F1-score **63.4%** vs **1.1%** off-the-shelf · Difference **62.2 points** (95% CI 57.5–66.8) · Holm-adjusted p < 0.001 → **H₀ rejected**  
*5-fold cross-validation on synthetic entries; confirm on the human-written test set. Training on journal data alone reaches 60.5%; news pre-training adds 2.9 points, which is not significant (p = 0.11).*
