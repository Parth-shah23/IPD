# Journal NER model

The trained model weights are not stored in this repository. Each user can train
the model locally from the included v3 workbook. Training downloads the
`dslim/bert-base-NER` starting checkpoint from Hugging Face and saves the
fine-tuned model into the ignored `best_ner_model/` directory.

## Setup (Windows PowerShell)

Use Python 3.13, then run:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the fine-tuned model

Run from the repository directory:

```powershell
python fine_tune_ner.py
```

The script reads `ECPE_Synthetic_Journal_Dataset_v3.xlsx` (`Entries` for
journal text and `Pairs` for NER annotations), evaluates on a fixed
entry-level validation split, and saves the selected model locally to
`best_ner_model/`.

## Predict NER for journal entries

After training:

```powershell
python ner_predict.py
```

Enter a journal entry when prompted. Predictions are printed as JSON with the
entity text, label, character offsets, and confidence. Submit a blank line to
exit.

## Run the Streamlit model demo

The interface runs ECPE and NER on a journal entry, runs crisis detection on a
separate clause, and then predicts an aspect from the editable ECPE/NER feature
fields:

```powershell
streamlit run app.py
```

Keep these model files in the repository directory:

- `ecpe_model.pkl`
- `crisis_model.pkl`
- `ner_model_v3.pkl`
- `aspect_model_rich_logreg.pkl`

The crisis checkpoint only contains `LABEL_0` and `LABEL_1`, so the interface
reports those raw class names rather than assuming which one represents a
crisis.

# Emotion-cause pair extraction (ECPE) model

A span-based joint model (`ecpe_model.py`). A RoBERTa encoder reads the whole
entry and:

1. tags emotion spans and cause spans (BIO tags),
2. classifies the emotion of each emotion span (10 labels; entries with no
   feeling simply have no emotion spans),
3. decides whether each emotion's cause is `stated`, `implicit` or `none`,
4. links each emotion span with a stated cause to its cause span(s).

Pairs can cross sentences, and one cause can be shared by several emotions.

## Cross-validate (development numbers)

```powershell
python train_ecpe.py --cv
```

Runs 5-fold cross-validation over entries, stratified by `planned_structure`.
Reports out-of-fold precision, recall and F1 with 95% bootstrap confidence
intervals over entries, for emotion spans, cause spans, pairs and
(emotion, label, cause) triplets. Each metric is reported with **exact** span
match and **relaxed** match (character overlap/union of at least 0.5), and
pair F1 is also broken down by structure. Results and out-of-fold predictions
(for error analysis) are written to `ecpe_results/`.

These are synthetic-data numbers. Do not report them as real-journal performance.

## Train the final model

```powershell
python train_ecpe.py
```

Trains on 90% of the entries, selects the epoch by validation pair F1, and
saves the model to the ignored `best_ecpe_model/` directory. Use
`--model distilroberta-base` for a smaller encoder, and `--batch-size 4
--grad-accum 2` on GPUs with little memory.

## Predict

```powershell
python ecpe_predict.py
```

## Hypothesis experiments (H1 transfer, H2 adaptation)

```powershell
python run_hypotheses.py
```

Downloads the GoodNewsEveryone news-headline corpus (Bostan et al., LREC 2020;
annotations CC BY 4.0) into the ignored `data_external/` folder, trains the
off-the-shelf model on news only, scores it on news and journal entries (H1),
then fine-tunes it on journal entries with the same 5 folds as
`train_ecpe.py --cv` and compares (H2). If `ecpe_results/oof_predictions.jsonl`
is present, journal-only training is a second H2 comparison. Results go to
`ecpe_results_hypotheses/`. Run `train_ecpe.py --cv` first, and use a GPU.

Until the human-written test set exists, the journal side of both tests uses
the synthetic entries, so these are development results.
