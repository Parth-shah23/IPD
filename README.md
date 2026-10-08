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
