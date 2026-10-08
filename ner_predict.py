import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parent / "best_ner_model"
MAX_LENGTH = 512
STRIDE = 64
WORD_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def load_model():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Fine-tuned model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The saved model requires a fast tokenizer for word alignment.")

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_entities(text, tokenizer, model, device):
    words = list(WORD_PATTERN.finditer(text))
    if not words:
        return []

    encoded = tokenizer(
        [match.group() for match in words],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: encoded[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in encoded
    }

    logits_by_word = {}
    with torch.inference_mode():
        logits = model(**model_inputs).logits.cpu()

    for batch_index in range(logits.shape[0]):
        seen_word_ids = set()
        for token_index, word_id in enumerate(encoded.word_ids(batch_index=batch_index)):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            logits_by_word.setdefault(word_id, []).append(logits[batch_index, token_index])

    labels = ["O"] * len(words)
    confidences = [0.0] * len(words)
    for word_id, word_logits in logits_by_word.items():
        mean_logits = torch.stack(word_logits).mean(dim=0)
        probabilities = torch.softmax(mean_logits, dim=-1)
        label_id = int(torch.argmax(probabilities))
        labels[word_id] = model.config.id2label[label_id]
        confidences[word_id] = float(probabilities[label_id])

    entities = []
    entity_start = None
    entity_type = None

    def finish_entity(end_word):
        if entity_start is None:
            return
        start_offset = words[entity_start].start()
        end_offset = words[end_word - 1].end()
        confidence = sum(confidences[entity_start:end_word]) / (end_word - entity_start)
        entities.append({
            "text": text[start_offset:end_offset],
            "label": entity_type,
            "start": start_offset,
            "end": end_offset,
            "confidence": round(confidence, 4),
        })

    for word_index, label in enumerate(labels):
        if label.startswith("B-"):
            finish_entity(word_index)
            entity_start = word_index
            entity_type = label[2:]
        elif label.startswith("I-"):
            next_type = label[2:]
            if entity_type != next_type:
                finish_entity(word_index)
                entity_start = word_index
                entity_type = next_type
        else:
            finish_entity(word_index)
            entity_start = None
            entity_type = None

    finish_entity(len(words))
    return entities


def main():
    tokenizer, model, device = load_model()
    print(f"Loaded NER model from {MODEL_DIR} ({device}).")
    print("Enter a journal entry; submit a blank line to quit.")

    try:
        while True:
            text = input("\nJournal entry: ")
            if not text.strip():
                break
            print(json.dumps(predict_entities(text, tokenizer, model, device), ensure_ascii=False, indent=2))
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
