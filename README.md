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
