# Emotion-Cause Journaling System (ECPE): Project Documentation

Working title: **Emotion-cause analysis for digital journaling**
Document status: working draft. Last updated for the 350-entry synthetic dataset (v3).

---

## 1. Project summary

Traditional journaling is private and unstructured. People write about their day, but they rarely see patterns such as which emotions recur and what causes them. Existing mood trackers mostly count emotions and do not say why they occurred.

This project converts a daily journal entry into structured data. For each entry, the system extracts:

- the **emotions** the writer expresses (for example joy, fear, guilt),
- the **cause** of each emotion (the clause that explains it),
- the **pairing** between each emotion and its cause (emotion-cause pair extraction, ECPE),
- the **life area** of each cause (studies, family, health, and so on),
- the **named entities** mentioned (people, places, organisations, dates),
- a **safety flag** for crisis-related language, which triggers support resources.

The extracted structure then powers an emotion dashboard, a trend and cause graph, a chatbot that answers only from the user's own entries, and scheduled summaries.

### Scope of this project

The project is scoped to **models, data and evaluation**, with the full application presented as a proposed system design:

- **Built or being built:** the dataset, the clause and aspect baselines, the NER evaluation, the safety model, and the hypothesis tests.
- **Proposed design (future work):** the web application, backend, storage, RAG chatbot, summariser and feedback loop.

### What this project is not

- It is not a diagnostic tool. It does not diagnose depression or stage its severity.
- It is not a therapist or a medical device. The chatbot does not give advice.
- It does not invent a new ECPE algorithm. It applies and evaluates existing approaches on a new domain (personal journal text) and contributes data and evaluation.

---

## 2. Primary hypothesis

Emotion-cause pair extraction from first-person journal text performs measurably worse than on news benchmarks, and adapting the model with journal-style data closes part of that gap.

---

## 3. Research questions and hypotheses

Each hypothesis has a null (H0) and an alternative (H1). All tests use α = 0.05, one-sided where the direction is stated, and Holm correction across comparisons within a task.

| # | Task | H0 | H1 | Metric | Test |
|---|---|---|---|---|---|
| H1 | ECPE transfer | Pair F1 on the journal test set equals pair F1 on the public English benchmark | Pair F1 on journals is lower | Pair F1 | Paired bootstrap over entries |
| H2 | ECPE adaptation | The fine-tuned model has the same pair F1 as the off-the-shelf model on journals | The fine-tuned model has higher pair F1 | Pair F1 | Paired bootstrap, Holm |
| H3 | Safety classifier | RoBERTa has the same precision at 95% recall as the TF-IDF baseline | RoBERTa has higher precision at that recall | Precision at fixed recall | McNemar or paired bootstrap |
| H4 | NER | Pretrained BERT NER (dslim/bert-base-NER) and spaCy have the same entity F1 on the human test set | BERT NER has higher entity F1 | Entity-level F1 (seqeval) | Paired bootstrap over entries, Holm |
| H5 | Aspect classification | The transformer has the same macro-F1 as TF-IDF + SVM on cause clauses | The transformer has higher macro-F1 | Macro-F1 | Paired bootstrap over entries |
| H6 | Retrieval (optional) | Filter-then-rank retrieves relevant entries no better than plain semantic search | Filter-then-rank retrieves more relevant entries | Recall@5, MRR | Wilcoxon signed-rank per query |

### Hypothesis testing rules

- Write each hypothesis before running any test.
- All models in a comparison use identical splits.
- Never tune on the test set. Choose thresholds and hyperparameters on validation data only.
- Resample entries, not rows, in bootstrap tests, because the pairs from one entry are not independent.
- Report confidence intervals and effect sizes, not only p-values.
- Report a non-significant result honestly. A cheaper model that is not significantly worse can be the chosen model on cost and latency grounds.

---

## 4. Dataset

### 4.1 Overview

| Property | Value |
|---|---|
| Entries | 350 (IDs S001 to S350) |
| Emotion-cause pair rows | 438 |
| Source | LLM-written (Claude), in three batches, with the team's direction |
| Status of labels | LLM draft labels. Automatic checks pass. **Not yet human-reviewed.** |
| Intended use | Training and development. **Not** the final test set. |
| File | `ECPE_Synthetic_Journal_Dataset_v3.xlsx` |

### 4.2 Why synthetic data is used

No public journal-style ECPE dataset is known to exist, and real journals need consent, anonymisation and ethics approval. The synthetic set is a stand-in for development, so that every model's pipeline can be built and checked before human-written data is ready.

### 4.3 The limitation that governs everything

LLM-written text is cleaner and more explicit than real journals. Emotion clauses often contain the emotion word itself ("so angry", "really scared"), and cause clauses are often spelled out. **Scores on the synthetic data will be higher than on real journals.** Synthetic results must never be presented as real-journal performance. The human-written test set (30 to 50 entries minimum, written and annotated by people other than the writer) is the only source for reported results.

### 4.4 Workbook structure

| Sheet | Contents | Notes |
|---|---|---|
| Read Me | Purpose, status, how to add human entries | Read first |
| Entries | ID, Author, Source, Text, planned_emotions, planned_structure, length_words, length_bucket, status, batch | Length and bucket are formulas |
| Pairs | entry_id, emotion_clause, emotion, cause_clause, cause, ner, aspect, annotator | One row per emotion-cause pair; `ner` filled on the first row of each entry, `same` on the rest |
| Balance | Live counts against the plan (structure, emotion, cause area, length, entities) | Blue cells are editable targets |
| Guidelines | Emotion definitions, tie-breakers, aspect definitions, entity rules, row conventions | The labelling rules used to produce the draft labels |

Dropdowns for emotion, aspect and structure are set on the Pairs and Entries sheets.

### 4.5 Entry fields

- **ID:** unique code (S001 to S350). Human entries will use H001 onward.
- **Author:** writer initial. Synthetic entries are marked "Claude (LLM)".
- **Source:** "Synthetic" for these entries. Human entries will be "Human".
- **Text:** the entry, one cell, no line breaks, typos preserved. Clause and entity text must be copied exactly from it.
- **planned_emotions / planned_structure:** the writer's intended coverage. Used only for balancing the dataset, never for scoring.
- **length_bucket:** short (under 40 words), medium (40 to 99), long (100 to 200), very long (over 200). Computed by formula.

### 4.6 Pair row fields

- **emotion_clause:** the words that express the emotion, copied exactly.
- **emotion:** one of 11 labels (section 5.1).
- **cause_clause:** the words that explain the emotion, copied exactly. Empty when no cause is stated. "implicit" when the cause is not stated but could be inferred.
- **cause:** a short free-text label. Kept for reference. Not used in evaluation, because free text does not give reliable agreement.
- **aspect:** life area of the cause (section 5.2).
- **ner:** entities in the form `text|TYPE; text|TYPE`, or `none`, or `same`.
- **annotator:** the labeller's initial, or "LLM-draft" for synthetic labels.

### 4.7 Current composition (v3, 350 entries)

**Structure of entries (entries per type):**

| Structure | Entries | Target |
|---|---|---|
| Cause before emotion | 87 | 88 |
| Cause after emotion | 35 | 35 |
| Different sentences | 53 | 52 |
| Implicit cause | 52 | 52 |
| Two emotions, two causes | 70 | 70 |
| One cause, two emotions | 18 | 18 |
| Emotion only | 17 | 17 |
| No emotion | 18 | 17 |

**Emotion labels (pair rows, n = 438):**

| Emotion | Rows | Share |
|---|---|---|
| joy | 55 | 12.6% |
| fear | 52 | 11.9% |
| sadness | 51 | 11.6% |
| contentment | 46 | 10.5% |
| anger | 42 | 9.6% |
| stress | 41 | 9.4% |
| pride | 39 | 8.9% |
| guilt | 34 | 7.8% |
| gratitude | 30 | 6.8% |
| surprise | 30 | 6.8% |
| neutral | 18 | 4.1% |

**Cause areas (pair rows):**

| Aspect | Rows | Share |
|---|---|---|
| studies_work | 77 | 17.6% |
| friends | 55 | 12.6% |
| health | 46 | 10.5% |
| other | 40 | 9.1% |
| money | 36 | 8.2% |
| family | 34 | 7.8% |
| self | 34 | 7.8% |
| relationships | 29 | 6.6% |
| none | 87 | 19.9% |

**Length:**

| Bucket | Entries | Target |
|---|---|---|
| Short (under 40 words) | 148 | 70 |
| Medium (40 to 99) | 165 | 175 |
| Long (100 to 200) | 34 | 88 |
| Very long (over 200) | 3 | 17 |

**Entities:** 227 entries contain at least one entity. 90 entries mention a person by role (for example Mom, my sister, the doctor).

### 4.8 Known gaps in the synthetic set

- Long entries are under-represented, which means the model will be tested on few long texts.
- Very long entries are almost absent (3 entries).
- Language is English with some kinship terms. There is no real code-mixing (Hinglish) even though it is common among the intended users.
- Role mentions are over-represented relative to real journals, which makes NER easier than it would be in practice.
- Labels have not been reviewed by a human.
- The "neutral" emotion has only 18 rows, which is too few to learn from.

### 4.9 Human-written test set (to do)

| Item | Requirement |
|---|---|
| Size | 30 to 50 entries minimum; more if time allows |
| Writers | Team members, consented volunteers, or both. Say which. |
| Content | Real-style journal writing, with the same structure and length mix as the synthetic set, and long entries included |
| Annotation | By someone other than the writer. Two annotators on a subset, with Cohen's kappa reported. |
| Storage | Separate from the training data. Frozen before any test is run. |
| Use | Final evaluation only. Never used for training or tuning. |

---

## 5. Labelling scheme

### 5.1 Emotion labels (11)

| Label | Meaning | Example clause |
|---|---|---|
| joy | Happiness, excitement, being pleased | "I am so excited I cannot sit still" |
| contentment | Mild, calm satisfaction | "feeling calm and content" |
| pride | Pleased with own or others' achievement | "Felt really proud of myself" |
| gratitude | Thankful toward someone or something | "so grateful for her" |
| sadness | Sadness, disappointment, loneliness, helplessness | "I am really sad" |
| anger | Anger, frustration, irritation, betrayal | "I was furious" |
| fear | Fear or anxiety about something specific or uncertain; panic | "Honestly scared" |
| stress | General overload or pressure, not a specific threat | "Overwhelmed and cannot even think straight" |
| guilt | Guilt, shame, embarrassment | "I feel so guilty" |
| surprise | Surprised or shocked by something unexpected | "Totally shocked" |
| neutral | No feeling stated; a plain log of the day | (no emotion clause) |

Mapping to the basic emotions for comparison with the literature:

| Our label | Basic emotion |
|---|---|
| joy, contentment, pride, gratitude | happiness |
| sadness, guilt | sadness |
| fear, stress | fear |
| anger | anger |
| surprise | surprise |

Disgust is excluded because it is rare in journals. Neutral is not a basic emotion and is kept for entries with no feeling.

### 5.2 Tie-breakers

- **fear vs stress:** fear points at a specific threat or uncertainty. Stress is general overload.
- **joy vs contentment:** joy is energetic. Contentment is calm.
- **sadness vs guilt:** guilt involves something the writer did wrong or felt embarrassed about.
- **surprise vs distress:** if the clause is mainly about something unexpected, use surprise. If it is mainly about distress, use the negative emotion.
- **shocked, astonished, taken aback:** all map to surprise. There is no separate "shocked" label.

### 5.3 Aspect labels (cause areas)

| Label | Covers |
|---|---|
| studies_work | College, exams, assignments, jobs, interviews, projects |
| health | Fitness, sleep, illness, medical visits |
| family | Parents, siblings, relatives, home |
| friends | Friends, roommates, neighbours, classmates |
| relationships | Romantic partner |
| self | Habits, identity, motivation, memories, personal milestones |
| money | Rent, fees, stipend, bills, purchases |
| other | Commute, lost items, everyday events that fit nothing above |
| none | No cause stated, or cause is implicit |

### 5.4 Entity types

| Type | Covers | Does not cover |
|---|---|---|
| PERSON | Named people (Raj, Neha, Prof. Mehta) | Role terms |
| ROLE | One person referred to by role or kinship (Mom, Dadi, my sister, the doctor, Sir) | Plural groups such as "my friends" or "relatives" |
| LOCATION | Named places (Pune, Nagpur, US) | Generic places (library, canteen, home, college) |
| ORG | Named organisations (TCS, Infosys, COEP) | Generic words such as "college" |
| DATE | Explicit time references (today, tonight, last night, this morning, Friday, March, next week) | Bare words such as "morning", and durations such as "two hours" |

Subjects such as "DBMS" or "workout plan" are not entities in this scheme. Kinship terms are tagged ROLE whether or not they are used like a name (for example "Dadu" is ROLE).

### 5.5 Row conventions

- Clause and entity text is copied exactly from the entry, including typos.
- Implicit cause: `cause_clause` = implicit, aspect = none.
- Emotion only: `cause_clause` empty, aspect = none.
- No emotion: one row, emotion = neutral, `emotion_clause` empty.
- An entry with two emotions sharing one cause has two rows with the same cause clause.
- The `ner` value appears on the first row of an entry only; later rows say `same`.

---

## 6. Models

### 6.1 Model summary

| Model | Built? | Training needed? | Data used | Current status |
|---|---|---|---|---|
| NER | Evaluation only | No (pretrained) | Pairs.ner column | Planned; next to run |
| Safety classifier | Not yet | Yes (Kaggle crisis data) | Kaggle set for training; journal set for test | Planned |
| ECPE extractor | Baselines | Yes after baselines | Public English ECPE benchmark; journal set | Baseline planned |
| Clause emotion classifier | Baseline run | Yes | Pairs.emotion_clause → emotion | Cross-validated baseline done |
| Aspect classifier | Baseline run | Yes | Pairs.cause_clause → aspect | Cross-validated baseline done |
| Embedding model (retrieval) | Not yet | No (pretrained) | Entries | Optional |
| Summariser (LLM) | Not yet | No (prompted) | Entries | Proposed |

### 6.2 NER (H4)

- **Candidates:** spaCy pretrained pipeline; `dslim/bert-base-NER` (trained on CoNLL-2003: PER, ORG, LOC, MISC). Optional third: a RoBERTa or Flair NER model.
- **Label mapping:** spaCy PERSON and BERT PER map to PERSON; GPE and LOC map to LOCATION; ORG maps to ORG; DATE maps to DATE. ROLE has no pretrained equivalent. Write the mapping down and state it in the paper.
- **Metric:** entity-level F1 with exact span and type match, using seqeval. Report per-type F1 as well.
- **Special reporting:** ROLE recall reported separately, since neither model is expected to find role mentions.
- **Test:** paired bootstrap over entries, 2,000 resamples, 95% confidence interval, one-sided p, Holm across comparisons.
- **Training:** none. Fine-tune only if the best pretrained model is clearly weak on journals, and then train on public data plus separate journal-style data, never on the test set.

### 6.3 Safety classifier (H3)

- **Purpose:** a safety net, not a diagnosis. It flags language that suggests self-harm or suicidal thoughts, and then shows support resources. It never blocks the user and does not run on the emotion dashboard.
- **Candidates:** TF-IDF + logistic regression (baseline); DistilRoBERTa; RoBERTa-base; optionally a mental-health-pretrained model.
- **Training data:** Kaggle suicide-risk dataset (Reddit-based). Clean out subreddit names and obvious label leaks before training.
- **Threshold:** chosen on the validation set to reach a target recall (for example 95%). Models are compared on precision at that fixed recall.
- **Test:** the human-written journal set. The synthetic set has no crisis content by design, so it can only measure false-alarm rate on non-crisis text.
- **Layering (optional):** a keyword and rule layer for explicit phrases, as a second check.
- **Limitations:** the Kaggle data is Reddit text, not clinician-labelled. It is not clinically validated. Report recall, precision and the confusion matrix, not only accuracy.
- **Ethics:** any crisis-language test entries must be written with mild, non-specific wording and approved by the supervisor first.

### 6.4 ECPE extractor (H1, H2)

- **Task:** given an entry, output emotion-cause pairs: (emotion clause, emotion, cause clause).
- **Baseline 1:** a published ECPE model used as is, trained on the English benchmark (novels-based set).
- **Baseline 2:** an LLM few-shot prompt, with a few examples from the benchmark or from the development data (not the test set).
- **Candidate:** the same published architecture fine-tuned on journal-style training data.
- **Fallback levels (if full ECPE is too slow to build):**
  - Level 1 (full): extract emotion clauses and cause clauses, then pair them with a trained model.
  - Level 2 (reduced): split entries into clauses, classify each clause as emotion, cause or neither, and pair with a nearest-clause rule or a small pair classifier.
  - Level 3 (minimum): LLM few-shot versus whichever trained model exists.
- **Metrics:** pair F1 (primary), emotion F1 and cause F1 (reported separately so partial progress still shows).
- **Test:** paired bootstrap over entries.

### 6.5 Clause classifiers (emotion and aspect)

These are the trainable baselines built on the synthetic set. They exist to check the pipeline and produce first numbers. They are not final models.

- **Emotion task:** input is the emotion clause; output is one of 11 emotions. Rows with an empty emotion clause are excluded.
- **Aspect task:** input is the cause clause; output is one of 8 aspects. Rows with no stated cause (`implicit` or empty) and rows labelled `none` are excluded.
- **Candidates:** majority-class baseline, TF-IDF + logistic regression, TF-IDF + linear SVM. A transformer (DistilRoBERTa) can be enabled with `--transformer`; it is untested in the authoring environment and needs a GPU.
- **Evaluation:** 5-fold stratified group cross-validation, grouped by entry so that pairs from one entry never appear in both training and test folds. Macro-F1 with 2,000-resample bootstrap confidence intervals over entries.
- **Script:** `clause_models.py`.

### 6.6 Aspect classifier (H5)

- **Candidates:** TF-IDF + SVM, DistilBERT, RoBERTa-base, zero-shot LLM.
- **Metric:** macro-F1, because classes are imbalanced.
- **Test:** paired bootstrap over entries, Holm.

### 6.7 Retrieval and chatbot (H6, optional)

- **Design:** a user chooses an emotion and a cause. A metadata filter in the relational DB selects that user's matching entries. Semantic search ranks those entries in the vector DB. An LLM answers using only the retrieved entries, and a response check screens the reply.
- **Ablation:** filter-then-rank versus plain semantic search over all of the user's entries.
- **Metrics:** Recall@5 and MRR against a 30 to 50 question set with relevant entries marked. Groundedness checked by a human rater.
- **Status:** the design is complete; the ablation is optional and only needed if the paper claims it.

---

## 7. System architecture

The complete design is shown in the consolidated diagram and in `ECPE_System_Architecture.drawio`. Solid borders are built or being built for this project. Dashed borders are the proposed system design.

### 7.1 Layers

1. **Client (proposed):** journal editor, trend dashboard, RAG chatbot, summaries.
2. **Backend API (proposed):** FastAPI. Authentication, per-user isolation, validation, consent.
3. **Ingestion pipeline (per entry, on submit):**
   - Input: text, or photo via OCR (optional).
   - Safety classifier: runs first, synchronously. If flagged, support resources are shown. The entry is still processed.
   - Store the raw entry in the relational DB.
   - In parallel: ECPE extractor, NER, aspect classifier.
   - Merge into a structured record (tags, clauses, entities, embedding).
4. **Features (proposed):** trend and cause graph (SQL on tags, no LLM); summariser (scheduled, LLM).
5. **RAG chatbot (proposed):** filter by tags (relational DB), rank by meaning (vector DB), answer (LLM, entries only), check reply (safety and grounding).
6. **Feedback loop (proposed):** user corrections stored in a feedback table (consented only), used by an offline retraining job, tested on the frozen test set, and registered as a new model version only if it does not get worse.
7. **Storage (proposed):** relational DB (entries, pairs, feedback); vector DB (per-user namespaces); summary DB; object store (photos, optional).
8. **Models and external services:** model registry (Hugging Face Hub, versioned); prompt configuration (templates, not models); LLM API.
9. **Offline:** datasets, training, evaluation, hypothesis tests, then push to the registry.
10. **Cross-cutting:** authentication, encryption, consent, delete-my-data, feedback loop, safety logging.

### 7.2 Design decisions and reasons

| Decision | Reason |
|---|---|
| Safety runs before storage and models | Support resources must appear immediately, and the system should never process a crisis message silently |
| Safety never blocks the user | A missed resource is a worse outcome than an unnecessary one; the user keeps access to their journal |
| Models run in parallel | They read the same stored entry and do not depend on each other |
| Dashboard uses SQL, not an LLM | Counts on tags are deterministic, fast and cheap |
| Chatbot filters before ranking | Ensures answers come only from entries with the chosen emotion and cause, and keeps other users' data out of the search |
| Prompted LLMs sit outside the model registry | A prompt is configuration, not a trained artefact |
| Feedback is used only with consent and only after testing | Prevents users' corrections from degrading the model unnoticed |

### 7.3 Suggested stack (flexible)

| Layer | Option |
|---|---|
| Frontend | React or Next.js |
| Backend | FastAPI (Python) |
| Relational DB | PostgreSQL |
| Vector DB | pgvector, Qdrant or Chroma (one namespace per user) |
| Models | Hugging Face Transformers, sentence-transformers |
| OCR | Tesseract or EasyOCR |
| Object store | S3 or any object store |
| Dashboard charts | Plotly |

---

## 8. Data flow for one journal entry

1. The writer submits text.
2. The safety classifier scores it. If flagged, the support message is shown.
3. The raw entry is saved with a timestamp and user ID.
4. The ECPE extractor returns emotion-cause pairs. The NER model returns entities. The aspect classifier labels each cause.
5. The results are merged into one record and saved to the relational DB.
6. The entry is embedded and saved to the vector DB with metadata (user ID, entry ID, emotions, causes, aspects, date).
7. The dashboard and graph update from the relational DB on the next load.

Example (from the team's own entry):

> "Felt proud for like five minutes, then got anxious because I haven't opened my notes for the DBMS exam."

| Field | Value |
|---|---|
| Pair 1 | emotion_clause "Felt proud for like five minutes" → pride; cause "started the workout plan" → health |
| Pair 2 | emotion_clause "got anxious" → fear; cause "I haven't opened my notes for the DBMS exam" → studies_work |
| NER | none (DBMS is a subject, not an entity in this scheme) |

---

## 9. Evaluation plan

### 9.1 Test sets

| Set | Source | Use |
|---|---|---|
| Development set | 350 synthetic entries (v3) | Pipeline checks, baselines, hyperparameters, error analysis |
| Test set | Human-written entries (30 to 50 minimum) | All reported results |
| Public benchmark | English ECPE benchmark (novels-based) | Domain-shift comparison (H1) |
| Safety test | Kaggle crisis data (held-out split) plus human journal entries | H3 |

### 9.2 Metrics

| Task | Primary | Also report |
|---|---|---|
| ECPE | Pair F1 | Emotion F1, cause F1, precision, recall |
| Emotion (clause) | Macro-F1 | Per-class F1, confusion matrix |
| Aspect | Macro-F1 | Per-class F1, confusion matrix |
| NER | Entity F1 (seqeval) | Per-type F1, ROLE recall |
| Safety | Precision at fixed recall (e.g. 95%) | Recall, precision, confusion matrix |
| Retrieval | Recall@5 | MRR |
| Summaries | Human rating (faithfulness, usefulness, 1 to 5) | |

### 9.3 Results so far (synthetic development set, cross-validated)

These are development numbers on LLM-written data. They are **not** evidence about real journals and must not be presented as such.

| Task | Rows | Entries | Best model | Macro-F1 | 95% CI | Accuracy |
|---|---|---|---|---|---|---|
| Emotion (clause) | 420 | 332 | TF-IDF + linear SVM | 0.815 | 0.777 to 0.849 | 0.805 |
| Emotion (clause) | 420 | 332 | TF-IDF + logistic regression | 0.796 | 0.756 to 0.833 | 0.788 |
| Emotion (clause) | 420 | 332 | Majority class | 0.023 | 0.019 to 0.028 | 0.131 |
| Aspect (cause clause) | 351 | 263 | TF-IDF + linear SVM | 0.379 | 0.322 to 0.430 | 0.416 |
| Aspect (cause clause) | 351 | 263 | TF-IDF + logistic regression | 0.372 | 0.316 to 0.423 | 0.393 |
| Aspect (cause clause) | 351 | 263 | Majority class | 0.045 | 0.036 to 0.053 | 0.219 |

Paired bootstrap:
- Emotion: SVM over logistic regression, mean difference +0.019, p = 0.038 (one-sided).
- Aspect: SVM over logistic regression, mean difference +0.007, p = 0.355. Not significant.

Interpretation:
- The emotion score is high partly because emotion clauses often contain the emotion word. Real journals will score lower.
- The aspect score is weak and did not improve with more data. The limit seems to be the task, since a short cause clause often does not say which life area it belongs to. A model that sees the whole entry may do better.

### 9.4 Results still to produce

- H1, H2 (ECPE): not started.
- H3 (safety): not started.
- H4 (NER): not started. Next to run.
- H5 (aspect, transformer): not started.
- H6 (retrieval): optional, not started.

### 9.5 Error analysis plan

Count model mistakes in these categories and report the counts:
- implicit cause missed;
- cause in a different sentence from the emotion;
- several emotions in one entry, some paired wrongly;
- role-based mentions missed by NER;
- emotion word absent from the clause (hard for models trained on synthetic data);
- borderline emotion pairs (fear vs stress, joy vs contentment, sadness vs guilt).

Error analysis is also the basis for any new method claim. A new method is only justified if one failure pattern clearly dominates and a targeted fix improves on the baseline.

---

## 10. Novelty and contributions

Use "to our knowledge" for any claim about being first. Do a proper literature search (Google Scholar, arXiv, ACL Anthology) before submitting.

### 10.1 Paper-worthy (if the evidence is produced)

1. **Application:** emotion-cause pair extraction applied to personal journal text. Existing ECPE work targets news, novels and conversations.
2. **Dataset:** a journal-style ECPE resource with annotation guidelines and agreement figures. The synthetic set is a development resource. The human-written set is the benchmark once annotated.
3. **Transfer study:** how existing ECPE and NER models behave on journal text, with hypothesis tests and error analysis.
4. **Domain shift for safety:** whether a Reddit-trained crisis detector holds on journal-style text (H3).
5. **Retrieval design (only with the ablation):** emotion-cause filter-then-rank versus plain semantic search (H6).
6. **Longitudinal insight (only with the aggregate-accuracy check):** whether the trend and cause graph built from model output agrees with the graph built from gold labels.

### 10.2 Supporting (strengthens the paper, not a claim on its own)

- LLM-bootstrapped annotation with reported correction rates.
- A recall-first safety gate that never blocks the user.
- Explainability through highlighted emotion and cause clauses.
- A user-grounded chatbot with guardrails.
- A privacy-aware design (per-user isolation, local-model option).
- The framing: journaling support with no diagnosis or depression staging.

### 10.3 Not novel (do not claim)

- The ECPE task itself.
- The pretrained models (BERT, RoBERTa, spaCy, NER models).
- Standard dashboards, the RAG pattern, and pipeline plumbing.
- Any new algorithm. The project does not propose one.

### 10.4 Honest phrasing

> "Existing emotion-cause pair extraction work covers news, novels and conversations. To our knowledge, it has not been applied to personal journal text. We build a journal-style evaluation resource, measure how existing models transfer to it, and analyse the errors."

---

## 11. Limitations

- **Synthetic development data:** LLM-written text, unreviewed labels, length and structure imbalance (section 4.8).
- **Small human test set:** with 30 to 50 entries, only large differences will be statistically detectable. Confidence intervals will be wide.
- **Single language:** English, with kinship terms. No real code-mixing.
- **Label reliability:** single-annotator labels are weaker than double-annotated ones. Agreement must be reported.
- **Domain shift:** Reddit-trained safety data and news-trained ECPE models may not transfer to journals.
- **No clinical validation:** the safety model is a safety net, not a screening tool, and has not been clinically tested.
- **Weak aspect performance:** cause-area classification is the weakest component so far.
- **No live users:** the application is designed, not deployed. User-facing claims about usefulness are not supported.
- **Ethics and consent:** any real journal data requires consent, anonymisation and possibly ethics approval.

---

## 12. Ethics, privacy and safety

- **Consent:** any human-written entries are provided with consent, or are written by the team as simulated entries. The paper must say which.
- **Anonymisation:** no real names, places or identifiable details in released data. Invented details are used.
- **Release:** data is released only if the ethics plan allows it. Once public, it cannot be withdrawn.
- **Crisis content:** no self-harm content is written or released without supervisor approval. Test entries use mild, non-specific language.
- **Safety responses:** support resources are shown calmly, without alarm. Region-appropriate helplines should be listed (for India, for example, Tele-MANAS at 14416). Confirm numbers before use.
- **No overclaiming:** the system does not diagnose, stage or treat anything.
- **Third-party APIs:** if an external LLM API is used, the user must be told that their text is sent to it, and must consent.
- **Data deletion:** users can delete their data (proposed feature).
- **Feedback use:** corrections are used for retraining only with consent.

---

## 13. Reproducibility

### 13.1 Files

| File | What it is |
|---|---|
| `ECPE_Synthetic_Journal_Dataset_v3.xlsx` | 350-entry synthetic dataset with Read Me, Entries, Pairs, Balance and Guidelines sheets |
| `clause_models.py` | Cross-validated clause classifier baselines (emotion and aspect), with bootstrap CIs and paired tests |
| `ECPE_System_Architecture.drawio` | Editable system architecture (open at app.diagrams.net) |
| `ECPE_Journal_Project_Plan.docx` | Project list: datasets, models, system components, features, experiments |
| `ECPE_Paper_Roadmap.docx` | Roadmap from dataset to paper, with checklists and risks |
| `ECPE_Project_Documentation.md` | This document |

### 13.2 Running the baselines

```bash
pip install pandas openpyxl scikit-learn
python clause_models.py --xlsx ECPE_Synthetic_Journal_Dataset_v3.xlsx --task emotion
python clause_models.py --xlsx ECPE_Synthetic_Journal_Dataset_v3.xlsx --task aspect
```

Transformer option (needs torch, transformers and a GPU, for example in Colab):

```bash
python clause_models.py --xlsx ECPE_Synthetic_Journal_Dataset_v3.xlsx --task emotion --transformer distilroberta-base
```

### 13.3 Rules for reproducible results

- Fix random seeds (the scripts use seed 42).
- Record library and model versions.
- Keep the human test set in a separate file, and record its checksum when it is frozen.
- Report every run that was made, including ones that did not help.
- Store the exact label mapping for NER with the results.

---

## 14. Timeline and status

| Phase | Work | Status |
|---|---|---|
| 0 | Plan, claim, venue, roles | Done (draft) |
| 1 | Literature review and gap statement | To do (search needed) |
| 2 | Synthetic development set (350 entries) | Done (draft labels, not reviewed) |
| 2 | Annotation guidelines and label scheme | Done (Guidelines sheet) |
| 2 | Human-written test set (30 to 50 entries) | **To do; critical path** |
| 2 | Label review by a second person | To do |
| 3 | Clause baselines (cross-validated) | Done |
| 3 | NER evaluation (pretrained models) | Next |
| 3 | Safety model (Kaggle training, journal test) | To do |
| 3 | ECPE baselines and fallback levels | To do |
| 3 | Aspect transformer | To do |
| 4 | Hypothesis tests on the human test set | To do |
| 4 | Error analysis | To do |
| 5 | Feature prototypes (dashboard, graph, chatbot) | Optional for the submission |
| 6 | Writing, figures, dataset release | To do |

### 14.1 Immediate next steps

1. Write the human-written test set (30 to 50 entries) and have it annotated by someone other than the writer.
2. Have a teammate review a sample of the synthetic labels.
3. Run the NER evaluation (H4) on the human test set; use the synthetic set only for a dry run.
4. Decide on the safety model's data plan and get supervisor approval for crisis-related test entries.
5. Run the clause baselines with a transformer on Colab.

---

## 15. Team roles (suggested for five people)

| Person | Ownership |
|---|---|
| A | Project lead, literature review, paper framing |
| B | Annotator 1, guidelines, agreement analysis |
| C | Annotator 2, dataset statistics, datasheet |
| D | ECPE baselines and experiments, hypothesis tests |
| E | Error analysis, figures, release package, LaTeX |

The two annotators should not see each other's labels during annotation. Everyone reads the draft.

---

## 16. Glossary

- **ECPE (emotion-cause pair extraction):** finding emotion clauses, cause clauses and which cause belongs to which emotion.
- **Clause:** a span of text expressing one emotion or one cause.
- **Aspect:** the life area of a cause (studies, family, health, and so on).
- **NER (named-entity recognition):** finding people, places, organisations and dates in text.
- **ROLE:** an entity referred to by role or kinship (Mom, my boss).
- **Macro-F1:** the average F1 across classes, so rare classes count equally.
- **Pair F1:** F1 for correctly extracted emotion-cause pairs, which requires both clauses and their link to be right.
- **Precision at fixed recall:** how many flagged entries are correct when the model is tuned to catch a set share of true cases.
- **Bootstrap confidence interval:** a range obtained by resampling the test data many times.
- **Paired bootstrap:** comparing two models by resampling the same test entries for both.
- **Holm correction:** adjusts p-values when several comparisons are made, to limit false positives.
- **Cohen's kappa:** agreement between two annotators, corrected for chance.
- **RAG (retrieval-augmented generation):** the LLM answers using retrieved text from the user's own data.
- **Domain shift:** a model trained on one kind of text performing worse on another kind.
- **Synthetic data:** text generated by a language model rather than written by people.
