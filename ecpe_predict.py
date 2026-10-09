import json
from pathlib import Path

import torch

from ecpe_model import MODEL_DIR, load_model, predict

MODEL_PATH = Path(__file__).resolve().parent / MODEL_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(MODEL_PATH, device)
    print(f"Loaded ECPE model from {MODEL_PATH} ({device}).")
    print("Enter a journal entry; submit a blank line to quit.")

    try:
        while True:
            text = input("\nJournal entry: ")
            if not text.strip():
                break
            result = predict(model, tokenizer, [text], device)[0]
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
