# Amazon ML Challenge 2026 — Top-Tier Business Entity Resolution Research Dossier

**Purpose:** Consolidate the strongest publicly visible Amazon ML Challenge 2026 / Business Entity Resolution approaches found during the current challenge window, focusing on repositories reporting **Macro F0.5 >= 0.985** and techniques that can be combined toward a **0.999x+ local Macro F0.5** system.

**Important:** The competition metric is **macro-averaged F0.5**, not ordinary accuracy. A score of `0.9994` below means `99.94% Macro F0.5` on the repository's stated validation/ablation population unless explicitly marked as a leaderboard score. Public leaderboard scores and private leaderboard scores are separate from local validation.

**Research snapshot:** 25 September 2026, during the live challenge.

---

# 1. Executive Objective

## Current objective

Build the strongest possible Business Entity Resolution system for the Amazon ML Challenge 2026.

## Desired target

```text
Primary target:       Macro F0.5 >= 0.9990
Stretch target:       Macro F0.5 >= 0.9995
Aspirational target:  Macro F0.5 >= 0.9999
```

The target is an engineering/research objective, not a guaranteed attainable leaderboard score.

### Core principle

At this level, simply adding another classifier is unlikely to be sufficient.

```text
DATA UNDERSTANDING
        ↓
MULTI-VIEW NORMALIZATION
        ↓
HIGH-RECALL CANDIDATE GENERATION
        ↓
PAIRWISE EVIDENCE
        ↓
HARD-NEGATIVE LEARNING
        ↓
PAIR SCORING / RANKING
        ↓
ENTITY-LEVEL / GRAPH-LEVEL REASONING
        ↓
PRECISION-FIRST DECISION
        ↓
VALIDATION + ERROR ANALYSIS
```

---

# 2. Competition Facts That Must Drive the Design

## 2.1 Metric

Official metric:

\[
F_{0.5} =
\frac{1.25 \cdot Precision \cdot Recall}
{0.25 \cdot Precision + Recall}
\]

The metric is calculated per Source-1 entity and then macro-averaged.

This makes false-positive merges especially expensive.

## 2.2 Cardinality

The task is not simple binary pair classification.

For each S1 entity:

```text
0 matches
1 match
multiple matches
```

are all possible.

At the same time, public dataset analysis reports that each S2/S3 entity links to at most one S1 in the analyzed ground truth.

Therefore there are two interacting structures:

```text
S1 → zero/many S2/S3
S2/S3 → at most one S1
```

That asymmetry is potentially exploitable.

## 2.3 Dataset scale

Public dataset analysis reports roughly:

```text
Total records: ~26.4M

Train S1: 2,206,821
Train S2: 5,034,616
Train S3: 5,285,603

Test S1: 1,732,544
Test S2: 4,887,273
Test S3: 5,082,316
```

This scale means:

- brute-force pair comparison is impossible;
- blocking/retrieval is a first-class model component;
- memory-safe streaming matters;
- candidate set size is itself an engineering variable.

## 2.4 Country structure

Public dataset analysis reports no cross-country ground-truth matches in the analyzed training ground truth. Country partitioning can therefore be used as a strong blocking restriction while keeping the model open-set.

Important design distinction:

```text
country = strong blocking key
BUT
country-specific hard-coded linguistic rules = risky
```

## 2.5 France domain shift

France is unseen during training.

A robust system should therefore explicitly test cross-domain behavior rather than relying on an in-domain random split only.

## 2.6 Singleton behavior

Singleton/no-match S1 entities matter disproportionately because a false match turns a potential perfect `1.0` entity score into `0.0`.

The decision layer therefore needs:

```text
match
vs.
abstain / empty set
```

logic rather than unconditional top-1 matching.

---

# 3. Current Public Leaderboard Snapshot

Observed from the live Unstop leaderboard screenshot on 25 Sep 2026:

| Rank | Team | Public Score |
|---:|---|---:|
| 1 | CDS_Team_iisc | 0.986955 |
| 2 | Asi Baat Hai Kya | 0.986854 |
| 3 | Git Wrecked | 0.986421 |
| 4 | Banana | 0.986 |
| 5 | NeuralNukes | 0.985 |
| 6 | Confusion Matrix | 0.985 |

These are **public leaderboard scores**, not local validation scores from the GitHub repositories below.

No public evidence found so far conclusively connects these top visible leaderboard teams to the open-source repositories analyzed in this document.

---

# 4. Primary Public Repositories With >= 0.985 Reported Score

| Repository | Reported Metric | Reported Score | Evidence Type | Important Caveat |
|---|---|---:|---|---|
| **ChandrimaNandi/Amazon-ML-Hackathon-2026** | Macro F0.5 | **0.9994** | `results/ablation_experiments.csv` | Local ablation/validation result; not verified public leaderboard |
| **prem970/Amazon-ML-challenge** | Macro F0.5 | **0.99813** peak | README + artifacts | README conflicts with older `training_metrics.json` value of 0.99684 |
| **AyanAhmedKhan/amazon-ml-challenge** | Macro F0.5 | **0.98703 ± 0.00016** | Explicit validation benchmark | Lower score, but unusually strong retrieval/error-analysis research |

### Critical interpretation

Do not blindly rank these by the raw number. They use different validation populations, splits, sampling densities, and experimental definitions.

The useful question is:

> Which techniques explain the strongest score under a validation protocol that best approximates the hidden leaderboard?

---

# 5. Repository A — ChandrimaNandi/Amazon-ML-Hackathon-2026

Repository: https://github.com/ChandrimaNandi/Amazon-ML-Hackathon-2026

## Reported performance

The repository's `results/ablation_experiments.csv` reports:

```text
Experiment              Features   Candidate Recall   Precision   Recall    F0.5

Exact matching             5          1.0000            0.5767     0.5300   0.5596
Name similarity only      11          1.0000            0.9037     0.9028   0.9004
Address similarity only   11          1.0000            0.9300     0.9278   0.9269
Name + address            22          1.0000            0.9939     0.9889   0.9916
+ country                 25          1.0000            0.9939     0.9889   0.9916
+ script                  27          1.0000            0.9906     0.9878   0.9892
+ retrieval agreement     30          1.0000            0.9906     0.9900   0.9889
+ BM25                    36          1.0000            0.9894     0.9900   0.9887
+ TF-IDF                  42          1.0000            0.9933     0.9922   0.9923
Full model                61          1.0000            1.0000     0.9983   0.9994
```

This is the strongest explicit **99.9%+ local Macro F0.5 claim** found in the repository search.

### Important warning

It is still a repository-reported validation/ablation result, not a verified 0.9994 Unstop leaderboard result.

---

## Architecture

### Stage 1 — Unicode-safe normalization

The project explicitly handles multilingual data:

- Unicode NFKC normalization
- mark-safe Unicode handling
- Devanagari transliteration
- Latin accent stripping
- normalized names and addresses

A key lesson is that naive regex cleanup can remove Unicode combining marks and damage Indic text.

### Stage 2 — Multi-channel candidate retrieval

The repository combines:

1. Exact normalized name
2. Exact normalized address
3. Exact combined name + address
4. BM25 name
5. BM25 combined
6. Character TF-IDF name
7. Character TF-IDF address

The retrieval channels preserve:

- channel membership
- retrieval score
- rank
- reciprocal rank
- agreement count

So the model receives second-order evidence:

```text
How similar are the strings?
+
How many independent retrieval mechanisms agree?
+
How highly do they rank this candidate?
```

---

## Feature system

The README documents 57 main deterministic features, while the ablation table says 61 features in the full model. This discrepancy should be checked by code inspection.

### Exact features

- exact raw name
- exact normalized name
- exact raw address
- exact normalized address
- exact combined representation

### Name similarity

- Levenshtein
- Jaro-Winkler
- RapidFuzz ratio
- partial ratio
- token sort
- token set
- token overlap
- token count differences
- length ratios
- transliteration-aware edit distance

### Address similarity

Equivalent families for addresses, plus structured address evidence.

### Retrieval metadata

- retrieval channel flags
- BM25 score/rank
- TF-IDF score/rank
- best rank
- reciprocal rank
- retrieval agreement count

### Country/script features

- country exact match
- country mismatch
- country missingness
- script match/mismatch

### Disambiguation

The code contains signals such as:

- acronym match
- distinct-name mismatch
- US-state mismatch
- missing-address penalty

---

## Negative sampling

The repository explicitly separates difficult negative classes:

```text
1. lexical hard negatives
2. address collisions
3. retrieval hard negatives
4. same-country negatives
5. random negatives
```

This is one of the most important ideas in the repository.

For a 0.999x target, random negatives are not enough. The model must repeatedly see:

```text
very similar but WRONG
```

examples.

---

## Model

Primary model:

```text
LightGBM
```

Training is performed after candidate generation and evaluated against Macro F0.5.

---

## Thresholding

The repository uses two dimensions:

```text
absolute score threshold
+
margin threshold
```

For a candidate group:

```text
margin = best_score - second_best_score
```

A candidate can therefore be rejected even when its absolute score is reasonably high if the top candidates are too close.

---

## Query exclusivity

The repository enforces:

```text
each S2/S3 query → at most one S1
```

while allowing:

```text
S1 → multiple S2/S3
```

This is a structurally faithful constraint and a strong precision mechanism.

---

## Streaming architecture

The repository uses memory-safe chunked inference and sparse CPU retrieval rather than attempting to materialize enormous dense matrices.

---

## What to extract

```text
A. multilingual normalization
B. 7-channel candidate retrieval
C. retrieval agreement/rank features
D. 5-category hard negative mining
E. 57–61 deterministic features
F. absolute + margin threshold search
G. query exclusivity
H. singleton/no-match handling
I. chunked inference
```

### Highest-value investigation

The ablation jumps from:

```text
0.9923
```

to the reported:

```text
0.9994
```

in the full model. Claude should inspect the exact code/config differences between stages 9 and 10 rather than accepting the entire pipeline as a black box.

---

# 6. Repository B — prem970/Amazon-ML-challenge

Repository: https://github.com/prem970/Amazon-ML-challenge

## Reported performance

README reports:

```text
Peak Validation Macro F0.5 = 0.99813
Optimal alpha = 0.35
Optimal threshold = 0.64
Official validator = PASS
```

However, the repository also contains:

```text
experiments/training_metrics.json

validation_macro_f05 = 0.99684
optimal_alpha = 0.7
optimal_threshold = 0.54
blocking_recall = 0.2492
```

### This discrepancy must be investigated

Possible explanations:

- an older experiment file was not updated;
- the README reflects a later run;
- different validation configurations were used;
- the blocking recall field is stale or computed differently.

For reproducibility, do not treat 0.99813 as definitively reproduced until this is resolved.

---

## Architecture

### Nine-pass blocking

```text
B1/B2 exact/legal/compact name
B3    high-information name token
B4    postal code
B5    locality/state + name
B6    house number + address token
B7    character n-gram retrieval
B8    TF-IDF name retrieval
B9    TF-IDF address retrieval
```

Candidate sets are unioned.

### Multi-view normalization

```text
raw
basic
legal
 token
compact
char_n-gram
```

Keeping multiple views avoids the common failure mode of a single aggressive normalizer destroying useful signal.

---

## 48-dimensional feature vector

### Name — 18

Includes:

- raw/basic/legal/compact equality
- Levenshtein
- Jaro-Winkler
- token sort/set
- token Jaccard
- character 3-gram Jaccard
- prefixes / first-last token
- length/token-count features

### Address — 17

Includes:

- edit/token similarities
- exact postal agreement/conflict
- house number agreement
- state agreement
- component agreement/conflict
- missingness

### Interaction/metadata — 13

Includes:

- name × address
- mean/min/max/difference
- agreement count
- country agreement/conflict
- missing country
- source indicators
- corner-case indicators

---

## Models

```text
XGBoost
+
LightGBM
```

Hybrid probability:

\[
P = \alpha P_{XGB} + (1-\alpha)P_{LGBM}
\]

The repository jointly searches the blend weight and final threshold against Macro F0.5.

---

## Hard-negative loop

The implementation mines candidates that:

```text
are NOT ground-truth matches
BUT
receive a high model probability
```

and feeds those false positives back into training.

This is directly aligned with the precision-heavy objective.

---

## Precision safety layer

Documented safeguards include:

- threshold rejection
- singleton abstention
- country contradiction guards
- preserving multiple legitimate matches
- no forced match for weak evidence

---

## Streaming inference

An SQLite-backed index is used for the large test set, reducing RAM pressure while keeping inference scalable.

---

# 7. Repository C — AyanAhmedKhan/amazon-ml-challenge

Repository: https://github.com/AyanAhmedKhan/amazon-ml-challenge

## Reported validation

The repository maintains an explicit experiment leaderboard.

Reported strongest completed result:

```text
Macro F0.5 = 0.98703 ± 0.00016
```

Architecture:

```text
multi-view blocking
+
87 features
+
3-fold cross-fitted LightGBM
+
collective/sibling features
+
stage-2 LightGBM
+
expected-F decision
+
query exclusivity
```

Although its score is below the two repositories above, its retrieval research and error analysis are exceptionally useful for constructing the next-generation system.

---

## Blocking results

Reported full-pool benchmark:

```text
Union pair recall = 0.9905
Candidates/S1 = 112
Oracle Macro F0.5 = 0.9972
```

Pruned configuration:

```text
~43–51 candidates/S1
pair recall ≈ 0.989
oracle ≈ 0.9965
```

The gap between actual model score and oracle indicates substantial classifier/decision headroom after blocking.

### Reverse retrieval

A key finding:

```text
target → S1 reverse retrieval
```

was much stronger than name-only retrieval.

Reported approximately:

```text
R@1 ≈ 0.972
R@8 ≈ 0.9875
```

The intuition is powerful: if each target belongs to at most one S1, the reverse query is naturally constrained.

---

# 8. Sibling / Look-Alike Entity Discovery

Ayan's error analysis is one of the highest-value pieces of the public research.

It identifies distractors with:

```text
same/near-same business name
same street
nearby house numbers
```

It also notes that TRUE copies can have small house-number offsets.

Therefore:

```text
house-number mismatch ≠ automatic rejection
```

but also:

```text
ignoring house-number evidence = false-merge risk
```

The proposed remedy is **collective/sibling reasoning**:

```text
Does this target agree with a cluster of confident copies
associated with one S1?
```

This is fundamentally different from ordinary pairwise classification.

---

# 9. Learned Cross-Script Dictionary

Ayan's pipeline learns native-script → Latin mappings from supplied training pairs instead of relying only on a static transliteration library.

This is potentially stronger because it learns the challenge's actual transformations.

Requirements for safe use:

- derive mappings only from training data;
- cross-fit where required;
- prevent validation leakage;
- evaluate under leave-one-country-out protocols.

---

# 10. Static Multilingual Embeddings

The repository uses an optional permissively licensed multilingual embedding model (`potion-multilingual-128M`) for cosine features.

The useful design pattern is not "use a giant neural model" but:

```text
lexical similarity
+
multilingual semantic similarity
+
GBDT
```

This can cover failure modes not handled by character/token methods.

---

# 11. Lower-Scoring Repositories Worth Mining for Ideas

These do not meet the >=0.985 primary threshold, but contain techniques worth importing selectively.

## Nikkilreddy01/fast-record-indexer

Reported:

```text
Validation Macro F0.5 = 0.9832
threshold = 0.660
```

Useful:

- high-recall inverted indexes
- phonetic blocking
- calibrated LightGBM
- streaming inference
- detailed research on oracle ceilings

https://github.com/Nikkilreddy01/fast-record-indexer

## SanthoshReddy352/Amazon-ML-Challenge-2026

Previously reported approximately:

```text
best local F0.5 ≈ 0.9821
```

Useful:

- exact/rare-token blocking
- TF-IDF/kNN
- multilingual embedding retrieval
- LightGBM reranking
- calibration
- set-level subset selection

https://github.com/SanthoshReddy352/Amazon-ML-Challenge-2026

## Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution

Reported:

```text
Macro F0.5 ≈ 0.98006
```

Useful:

- 55 engineered features
- XGBoost + LightGBM + CatBoost
- GroupKFold
- global consistency resolution
- hierarchical postal matching
- street-number distance
- harmonic/weakest-link similarities

https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution

---

# 12. Cross-Repository Feature Matrix

| Capability | Chandrima | Prem970 | Ayan | Best combined design |
|---|---:|---:|---:|---|
| Exact name blocking | ✓ | ✓ | ✓ | ✓ |
| Exact address blocking | ✓ | ✓ | ✓ | ✓ |
| Combined exact blocking | ✓ | ✓ | ✓ | ✓ |
| BM25 | ✓ | indirect/possible | ✓ | ✓ |
| Character TF-IDF | ✓ | ✓ | ✓ | ✓ |
| Reverse target→S1 retrieval | not central | not central | **✓** | **✓** |
| Retrieval rank features | **✓** | limited | **✓** | **✓** |
| Retrieval agreement count | **✓** | limited | **✓** | **✓** |
| Multiscript normalization | **✓** | ✓ | **✓** | **✓** |
| Learned transliteration map | translit | not central | **✓** | **✓ OOF** |
| 48–60+ pair features | **✓** | **✓** | **✓** | **✓** |
| Address components | ✓ | **✓** | ✓ | **✓** |
| House-number geometry | limited | ✓ | **✓ research** | **✓** |
| Hard-negative mining | **✓** | **✓** | ✓ | **✓** |
| XGBoost | — | **✓** | — | **✓** |
| LightGBM | **✓** | **✓** | **✓** | **✓** |
| CatBoost | optional | — | research | optional |
| Model blending | — | **✓** | stage stacking | **blend + stacking** |
| Cross-fitted validation | ✓ | entity split | **✓** | **✓** |
| Margin threshold | **✓** | safety layer | research | **✓** |
| Singleton abstention | **✓** | **✓** | ✓ | **✓** |
| Query exclusivity | **✓** | safety layer | **✓** | **✓** |
| Sibling/collective features | limited | limited | **✓** | **✓** |
| Global assignment | ✓ concept | safety layer | exclusivity | **✓** |
| Calibration | possible | optional | ✓ | **✓ only if validated** |
| Streaming inference | **✓** | **✓** | **✓** | **✓** |
| Official validator | ✓ | **✓** | ✓ | **✓** |

---

# 13. Strongest Combined Architecture

## Layer 0 — Data profiling

Measure:

```text
country distribution
missing-name distribution
missing-address distribution
script distribution
name length
address length
duplicate normalized names
duplicate normalized addresses
common tokens
rare tokens
postal coverage
house-number coverage
S2 vs S3 statistics
```

Every validation score should be sliceable by these variables.

---

## Layer 1 — Multi-View Normalization

For every record retain:

```text
raw
unicode-normalized
lowercase
accent-stripped
legal-normalized
tokenized
sorted-token
compact-alphanumeric
transliterated
character n-gram
component-parsed
```

Never throw away the original string.

---

## Layer 2 — Country Partition

Use the documented no-cross-country structure as a strong blocking key:

```text
S1(country=X)
      ↓
search only S2/S3(country=X)
```

but keep normalization and model behavior open-set so France is not hard-coded away.

---

## Layer 3 — Bidirectional Retrieval

Combine:

### Forward retrieval

```text
S1 → target
```

### Reverse retrieval

```text
target → S1
```

Candidate channels:

```text
exact
rare-token
name lexical
address lexical
BM25
char TF-IDF
embedding KNN
reverse target→S1
sibling/cluster expansion
```

---

## Layer 4 — Retrieval Agreement Features

For each candidate track:

```text
retrieved_by_exact_name?
retrieved_by_exact_address?
retrieved_by_BM25_name?
retrieved_by_BM25_combined?
retrieved_by_TFIDF_name?
retrieved_by_TFIDF_address?
retrieved_by_embedding?
retrieved_by_reverse?
```

Then derive:

```text
agreement_count
best_rank
second_best_rank
mean_rank
best_reciprocal_rank
score_sum
score_max
score_min
```

---

# 14. Pairwise Feature Engine

Target roughly:

```text
80–120 features
```

not because more is automatically better, but because different error modes need different evidence.

### Name

```text
exact raw
exact normalized
exact legal
exact compact
Levenshtein
Jaro-Winkler
RapidFuzz ratio
partial ratio
token sort
token set
token Jaccard
character n-gram similarity
prefix/suffix similarity
first token match
last token match
acronym match
token count difference
character length ratio
transliteration similarity
```

### Address

```text
Levenshtein
Jaro-Winkler
RapidFuzz ratios
token-set/sort
postal exact
postal prefix
house number exact
house-number absolute difference
house-number relative difference
house-number digit edit distance
city similarity
state similarity
locality similarity
component agreement count
component conflict count
missingness pattern
```

### Cross-field

```text
name × address
harmonic mean
minimum evidence
maximum evidence
difference
agreement count
```

### Entity-context

```text
number of competing S1 candidates
number of competing target candidates
rank gap
score gap
same-name frequency
name rarity
address rarity
token rarity
sibling cluster support
candidate density
```

---

# 15. Hard Negative Mining

This must be a first-class subsystem.

## Negative buckets

```text
random negative
same-name / different-address
same-address / different-name
same-locality / different-business
same-country lexical collision
top TF-IDF false positive
top BM25 false positive
top embedding false positive
same-name sibling
nearby-house-number sibling
cross-source duplicate-looking negative
```

Train iteratively:

```text
model v1
   ↓
score difficult candidate pool
   ↓
select highest-confidence false positives
   ↓
augment training set
   ↓
model v2
   ↓
repeat until saturation
```

---

# 16. Model Stack

A strong candidate configuration:

```text
Model A: XGBoost
Model B: LightGBM
Model C: optional CatBoost
```

Then optionally:

```text
Stage 1:
pair-level model probabilities

Stage 2:
S1-level / target-level collective features

Stage 2 model:
LightGBM
```

Useful Stage-2 features:

```text
p1
p2
p1-p2
sum(top-k probabilities)
same-S1 support
same-target support
sibling cluster support
source agreement
house-number consistency
candidate density
name collision density
```

---

# 17. Entity / Graph Reasoning

Represent the task as a bipartite graph:

```text
S1 nodes
   ↕
S2/S3 nodes
```

Edges store pair probabilities.

Constraint:

```text
Each S2/S3 target → degree <= 1 across S1
```

while S1 may accumulate multiple true targets.

Then reason at graph level:

```text
raw pair scores
      ↓
target competition
      ↓
S1 sibling support
      ↓
edge consistency
      ↓
global / component optimization
```

This is a stronger formulation than independent binary decisions.

---

# 18. Margin-Based Decision

For every target:

```text
p1 = top candidate probability
p2 = second candidate probability
margin = p1 - p2
```

Conceptual policy:

```text
p1 high AND margin high
    → accept

p1 high BUT margin low
    → ambiguous / abstain / defer

p1 low
    → no match
```

Thresholds must be learned on validation.

---

# 19. Singleton Gate

For each S1:

```text
top evidence < singleton threshold
        ↓
empty match list
```

Do not force top-1 for weak evidence.

Measure separately:

```text
singleton accuracy
singleton false-positive rate
singleton false-negative rate
```

---

# 20. Specialized Thresholds

A global threshold may be insufficient.

Potential validated grouping dimensions:

```text
country
S2 vs S3
missing-address pattern
missing-name pattern
candidate density
script pair
retrieval-agreement bucket
```

Use group-specific thresholds only where validation data supports them.

---

# 21. France / Domain-Shift Robustness

### Training-time map dropout

Randomly disable learned transliteration/abbreviation mappings on part of the training set so the model does not depend exclusively on learned maps.

### LOCO evaluation

Use:

```text
train US → validate India
train India → validate US
```

plus a controlled France-like robustness protocol using only challenge-provided data.

Never add external French business data.

---

# 22. Probability Calibration

Test:

```text
Platt scaling
isotonic regression
temperature scaling
```

Keep calibration only if Macro F0.5 or decision stability improves.

Do not assume calibration is automatically beneficial for tree ensembles.

---

# 23. Retrieval Safety Audit

Every candidate generator should report:

```text
candidate recall
candidate recall by source
candidate recall by country
candidate recall by missingness
candidate recall by script
average candidate count
p95 candidate count
p99 candidate count
reduction ratio
```

A classifier cannot recover a true pair removed by blocking.

---

# 24. Error Taxonomy

## False positives

```text
same-name collision
same-address collision
sibling entity
house-number collision
shared locality
generic company name
cross-script confusion
country error
missing-address hallucination
retrieval artifact
```

## False negatives

```text
transliteration
typo
legal suffix variation
address abbreviation
missing address
missing name
word-order change
cross-script
rare business
France domain shift
blocking miss
threshold rejection
```

Recurring error classes should become:

```text
new features
new hard negatives
new retrieval channels
new decision rules
```

---

# 25. What the 0.9994 Result Is Actually Telling Us

The Chandrima ablation gives a useful progression:

```text
Name only                         0.9004
Address only                      0.9269
Name + address                    0.9916
+ TF-IDF                          0.9923
Full model                        0.9994
```

The research lesson is not that every system will reproduce 0.9994.

The lesson is that the last percentage points can come from:

```text
retrieval diversity
+
multilingual normalization
+
hard negative structure
+
context/competition features
+
precision-aware decision logic
```

rather than from one exotic model.

---

# 26. What the 0.99813 Result Adds

Prem970 demonstrates a complementary route:

```text
9-pass blocking
+
48 features
+
XGBoost + LightGBM
+
hard-negative mining
+
alpha/threshold search
+
singleton/precision safety
+
streaming inference
```

The most useful action is to transplant the concepts, not blindly copy hyperparameters.

---

# 27. What the 0.98703 Result Adds

Ayan contributes the most valuable ceiling analysis:

```text
0.9905 candidate recall
0.9972 oracle Macro F0.5
0.98703 actual validation
```

This explicitly separates:

```text
blocking ceiling
vs.
classifier/decision performance
```

That distinction should be built into our own pipeline from day one.

---

# 28. Recommended Best-of-All Architecture

## Phase A — Retrieval

```text
Country partition
      ↓
Exact normalized name
      +
Exact normalized address
      +
Exact combined
      +
Rare-token index
      +
BM25 name
      +
BM25 combined
      +
Char TF-IDF name
      +
Char TF-IDF address
      +
Embedding KNN
      +
Reverse target→S1
      +
Sibling/cluster expansion
      ↓
Candidate union
```

Target:

```text
candidate recall >= 0.995 where computationally practical
```

## Phase B — Feature matrix

```text
~100 complementary features
```

## Phase C — Pair scorer

```text
XGBoost
+
LightGBM
```

Use out-of-fold probabilities.

## Phase D — Hard-negative loop

```text
train
→ score
→ mine hardest false positives
→ retrain
→ repeat
```

## Phase E — Collective model

Add:

```text
candidate rank
probability margin
sibling support
S1 competition
target competition
retrieval agreement
house-number cluster evidence
```

## Phase F — Global decision

```text
1. contradiction screening
2. high-confidence deterministic accepts
3. Stage-2 probability
4. margin gate
5. target exclusivity
6. singleton abstention
7. multi-match acceptance for independently strong edges
8. optional component/graph reconciliation
```

---

# 29. Validation Design Required for a Credible 0.999x

A single random split is insufficient.

Use at least:

```text
Protocol A: entity-level random split
Protocol B: GroupKFold by S1
Protocol C: density-matched validation
Protocol D: leave-one-country-out
Protocol E: hard-negative stress test
Protocol F: public leaderboard submissions
```

Report:

```text
A score
B score
C score
D score
E score
public leaderboard score
```

The validation-to-public gap must be tracked as its own metric.

---

# 30. Preventing False 0.999 Scores

Before trusting any very high result, verify:

### Split integrity

```text
S1_train ∩ S1_validation = ∅
after every preprocessing/cache step
```

### Target leakage

Validation targets must not influence:

```text
frequency features
learned abbreviation maps
transliteration dictionaries
collective probabilities
candidate pruning
threshold selection
```

unless the operation is explicitly cross-fitted.

### Candidate leakage

Validation candidates must be generated without using validation labels.

### Threshold leakage

Do not tune thresholds on the same evaluation population used to report the final headline score unless the score is explicitly labeled as an optimization score rather than an honest held-out estimate.

---

# 31. Highest-Value Experiments to Run First

If compute/time is limited:

```text
1. Reproduce Chandrima's 0.9994 experiment exactly.
2. Verify its validation population and leakage safety.
3. Resolve Prem970's 0.99813 vs 0.99684 discrepancy.
4. Add Prem970's XGB + LGBM blend.
5. Add Ayan reverse target→S1 retrieval.
6. Add retrieval-agreement/rank features.
7. Add sibling/look-alike collective features.
8. Add house-number geometry.
9. Add hard-negative mining by FP type.
10. Optimize threshold + margin jointly.
11. Add graph/entity-level reconciliation.
12. Add multilingual embedding features only if they improve validation.
13. Stress-test France and missing-address strata.
```

Do not start by building a large neural model.

---

# 32. Claude Prompt — Research / Architecture Task

Use this prompt after attaching this Markdown file and the current `pareto-frontier` codebase.

```text
You are the lead ML/entity-resolution researcher for the Amazon ML Challenge 2026.

Goal:
Design the strongest technically credible Business Entity Resolution pipeline possible, targeting >=0.9990 Macro F0.5 and ideally >=0.9995, while minimizing hidden/private leaderboard degradation.

Context:
- Official metric is macro-averaged F0.5 at the S1-entity level.
- Precision is weighted more heavily than recall.
- S1 can map to zero, one, or multiple S2/S3 records.
- Each S2/S3 target maps to at most one S1 in the analyzed ground truth.
- Test contains unseen France.
- External business databases/APIs/geocoding are prohibited.
- Final model must satisfy the competition model/license constraints.
- Current public leaderboard is around 0.987.
- Public repositories contain local validation claims up to approximately 0.9994.

Attached:
1. This research dossier.
2. Our current pareto-frontier codebase.

Task:
Do NOT give generic ML advice.

A. Audit every high-signal repository approach in this dossier.
B. Identify which ideas are complementary and which are redundant.
C. Inspect pareto-frontier and map:
   - what we already have,
   - what is missing,
   - what is incorrect,
   - what is too expensive,
   - what is most likely to increase Macro F0.5.

D. Design a concrete best-of-all architecture covering:
   1. normalization
   2. blocking/retrieval
   3. bidirectional retrieval
   4. feature engineering
   5. hard-negative mining
   6. pair models
   7. stacking/collective features
   8. sibling/look-alike reasoning
   9. graph/entity-level constraints
   10. margin/threshold decision logic
   11. singleton abstention
   12. France/domain-shift robustness
   13. probability calibration
   14. streaming inference

E. Explicitly investigate the strongest reported 0.9994 approach.
Do not accept its score blindly.
Determine:
- validation population,
- full vs sampled evaluation,
- candidate recall,
- leakage risk,
- exact mechanism behind the 0.9994 result.

F. Explicitly resolve the Prem970 discrepancy:
- README: 0.99813
- training_metrics.json: 0.99684
Explain which result is reproducible and why.

G. Build a mathematically sound validation framework:
- entity-level split,
- GroupKFold,
- density-matched split,
- leave-one-country-out,
- hard-negative stress test,
- error slices,
- public leaderboard gap tracking.

H. Produce an experiment plan ordered by expected information gain per unit time.
Every experiment must have:
- hypothesis,
- code change,
- expected effect,
- metric,
- runtime,
- pass/fail criterion,
- rollback condition.

I. Do not recommend complexity for its own sake.
Prefer a simple validated improvement over a complicated speculative model.

J. Focus on the final 0.1–0.3 percentage points:
- sibling entities,
- shared names,
- house-number collisions,
- missing-address cases,
- multilingual normalization,
- retrieval misses,
- singleton false positives,
- target competition,
- candidate density,
- score margins.

K. Return:
1. Final architecture diagram.
2. Data-flow diagram.
3. Exact feature catalog.
4. Exact candidate-generation rules.
5. Exact negative-sampling strategy.
6. Model/ensemble design.
7. Entity-level decision algorithm.
8. Validation protocol.
9. Experiment roadmap.
10. Expected bottleneck analysis.
11. Concrete code modification plan for pareto-frontier.
12. Final leaderboard-submission checklist.

Important:
- Separate measured evidence from hypotheses.
- Never treat a local validation score as a leaderboard score.
- Flag possible leakage.
- Do not claim 0.999x is guaranteed.
- Prefer reproducible experiments over theoretical speculation.
```

---

# 33. Final Research Hypothesis

The strongest combined hypothesis is:

```text
99.9x requires
NOT
"a better similarity model"

but

near-complete candidate recall
+
high-quality pair evidence
+
hard-negative discrimination
+
bidirectional retrieval
+
sibling/collective evidence
+
target-level competition
+
precision-first abstention
+
honest validation
```

The last part is critical.

A pipeline that reports 0.9994 because of an optimistic or leaky validation setup is less useful than a pipeline that reports 0.997 honestly and closes the gap on the hidden leaderboard.

---

# 34. Source Links

## Primary repositories

- ChandrimaNandi/Amazon-ML-Hackathon-2026
  https://github.com/ChandrimaNandi/Amazon-ML-Hackathon-2026

- prem970/Amazon-ML-challenge
  https://github.com/prem970/Amazon-ML-challenge

- AyanAhmedKhan/amazon-ml-challenge
  https://github.com/AyanAhmedKhan/amazon-ml-challenge

## Supporting repositories

- Nikkilreddy01/fast-record-indexer
  https://github.com/Nikkilreddy01/fast-record-indexer

- SanthoshReddy352/Amazon-ML-Challenge-2026
  https://github.com/SanthoshReddy352/Amazon-ML-Challenge-2026

- Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution
  https://github.com/Aamod007/Amazon-ML-Challenge-2026-Business-Entity-Resolution

- imMohammedSahil/Amazon-ML
  https://github.com/imMohammedSahil/Amazon-ML

- Ankittian/amazon_2026_toolkit
  https://github.com/Ankittian/amazon_2026_toolkit

## Dataset analysis

- akshatbakshi/amazon-ml-challenge-2026
  https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026

## Official challenge organization repository

- Amazon-ML-Challenge-2026/business-entity-resolution
  https://github.com/Amazon-ML-Challenge-2026/business-entity-resolution

---

## Final instruction to the next researcher

Do not stop at the repository-reported numbers.

Determine:

```text
Which mechanisms genuinely generalize?
Which numbers are validation artifacts?
What is the true bottleneck?
What can be reproduced quickly?
What combination has the highest evidence-backed ceiling?
```

The next output should be a concrete implementation plan for `pareto-frontier`, not a generic literature summary.
