import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL_CHECKPOINT = "dslim/bert-base-NER"
DATA_FILE = "ECPE_Synthetic_Journal_Dataset_v3.xlsx"
OUTPUT_DIR = "ner-fine-tuned-v3"
BEST_MODEL_DIR = "best_ner_model"
ENTITY_TYPES = ("DATE", "LOCATION", "ORG", "PERSON", "ROLE")
LABEL_LIST = ["O"] + [
    f"{prefix}-{entity_type}"
    for entity_type in ENTITY_TYPES
    for prefix in ("B", "I")
]
LABEL2ID = {label: index for index, label in enumerate(LABEL_LIST)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
SEED = 42
MAX_LENGTH = 512


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)(
            outputs.logits.reshape(-1, outputs.logits.shape[-1]),
            labels.reshape(-1),
        )
        return (loss, outputs) if return_outputs else loss


def tokenize_words(text):
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def parse_entities(entity_data):
    if pd.isna(entity_data) or str(entity_data).strip().lower() == "none":
        return []

    entities = []
    for item in str(entity_data).split(";"):
        if item.strip().lower() == "same":
            continue
        parts = item.rsplit("|", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid NER annotation: {item!r}")
        entity_text, entity_type = (part.strip() for part in parts)
        if not entity_text or not entity_type:
            continue
        entity_type = entity_type.upper()
        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Unsupported NER label {entity_type!r}")
        entities.append((entity_text, entity_type))
    return entities


def load_and_prepare_data(filepath):
    """Build BIO-labeled token records from Entries and Pairs workbook sheets."""
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    entries = pd.read_excel(filepath, sheet_name="Entries")
    pairs = pd.read_excel(filepath, sheet_name="Pairs")
    required_entry_columns = {"ID", "Text"}
    required_pair_columns = {"entry_id", "ner"}
    if not required_entry_columns.issubset(entries.columns):
        raise ValueError(f"Entries must contain columns {sorted(required_entry_columns)}")
    if not required_pair_columns.issubset(pairs.columns):
        raise ValueError(f"Pairs must contain columns {sorted(required_pair_columns)}")

    entities_by_id = {}
    for entry_id, annotations in pairs.dropna(subset=["ner"]).groupby("entry_id")["ner"]:
        entities_by_id[entry_id] = [
            entity
            for annotation in annotations
            for entity in parse_entities(annotation)
        ]

    records = []
    for _, row in entries.iterrows():
        if pd.isna(row["Text"]) or not str(row["Text"]).strip():
            continue

        words = tokenize_words(str(row["Text"]))
        labels = ["O"] * len(words)
        entities = sorted(
            entities_by_id.get(row["ID"], []),
            key=lambda entity: -len(tokenize_words(entity[0])),
        )
        for entity_text, entity_type in entities:
            entity_words = tokenize_words(entity_text)
            entity_length = len(entity_words)
            matching_starts = [
                start
                for start in range(len(words) - entity_length + 1)
                if [word.casefold() for word in words[start:start + entity_length]]
                == [word.casefold() for word in entity_words]
            ]
            if not matching_starts:
                raise ValueError(
                    f"Annotated entity {entity_text!r} not found in entry {row['ID']!r}"
                )
            for start in matching_starts:
                if all(label == "O" for label in labels[start:start + entity_length]):
                    labels[start] = f"B-{entity_type}"
                    for offset in range(1, entity_length):
                        labels[start + offset] = f"I-{entity_type}"

        records.append({
            "id": str(row["ID"]),
            "tokens": words,
            "ner_tags": [LABEL2ID[label] for label in labels],
        })

    if not records:
        raise ValueError("No non-empty journal entries were found.")
    return records


def entity_spans(labels):
    spans = set()
    start = None
    entity_type = None
    for index, label in enumerate(labels + ["O"]):
        if label.startswith("B-") or label == "O":
            if entity_type is not None:
                spans.add((start, index, entity_type))
            if label.startswith("B-"):
                start, entity_type = index, label[2:]
            else:
                start, entity_type = None, None
        elif label.startswith("I-"):
            next_type = label[2:]
            if entity_type != next_type:
                if entity_type is not None:
                    spans.add((start, index, entity_type))
                start, entity_type = index, next_type
    return spans


def entity_metrics(predictions, references):
    true_positive = false_positive = false_negative = 0
    for predicted, expected in zip(predictions, references):
        predicted_spans = entity_spans(predicted)
        expected_spans = entity_spans(expected)
        true_positive += len(predicted_spans & expected_spans)
        false_positive += len(predicted_spans - expected_spans)
        false_negative += len(expected_spans - predicted_spans)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for journal-entry NER.")
    parser.add_argument("--data", default=DATA_FILE, help="Workbook containing Entries and Pairs sheets.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--best-model-dir", default=BEST_MODEL_DIR)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    records = load_and_prepare_data(args.data)
    train_records, validation_records = train_test_split(
        records, test_size=0.2, random_state=SEED, shuffle=True
    )
    print(
        f"Entries: {len(records)} | Train: {len(train_records)} | "
        f"Validation: {len(validation_records)}"
    )

    train_label_counts = np.bincount(
        [tag for record in train_records for tag in record["ner_tags"]],
        minlength=len(LABEL_LIST),
    )
    weights = np.sqrt(
        train_label_counts.sum() / (len(LABEL_LIST) * np.maximum(train_label_counts, 1))
    )
    weights = np.minimum(weights, 10.0)
    weights /= weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float)

    raw_datasets = DatasetDict({
        "train": Dataset.from_list(train_records),
        "validation": Dataset.from_list(validation_records),
    })
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, add_prefix_space=True)

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )
        aligned_labels = []
        for batch_index, word_labels in enumerate(examples["ner_tags"]):
            previous_word_index = None
            labels = []
            for word_index in tokenized.word_ids(batch_index=batch_index):
                if word_index is None or word_index == previous_word_index:
                    labels.append(-100)
                else:
                    labels.append(word_labels[word_index])
                previous_word_index = word_index
            aligned_labels.append(labels)
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True)
    model_config = AutoConfig.from_pretrained(MODEL_CHECKPOINT)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    def compute_metrics(evaluation):
        predictions, label_ids = evaluation
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.argmax(predictions, axis=2)
        predicted_sequences = []
        expected_sequences = []
        for prediction, labels in zip(predictions, label_ids):
            active = labels != -100
            predicted_sequences.append([LABEL_LIST[index] for index in prediction[active]])
            expected_sequences.append([LABEL_LIST[index] for index in labels[active]])
        precision, recall, f1 = entity_metrics(predicted_sequences, expected_sequences)
        return {"precision": precision, "recall": recall, "f1": f1}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=SEED,
        data_seed=SEED,
        logging_steps=10,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print(f"Training {MODEL_CHECKPOINT} on {training_args.device}.")
    trainer.train()
    best_metrics = trainer.evaluate(tokenized_datasets["validation"])
    if "eval_f1" not in best_metrics or not np.isfinite(best_metrics["eval_f1"]):
        raise RuntimeError("Validation did not produce a finite entity-level F1 score.")

    model_dir = Path(args.best_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(
        "Best validation scores: "
        f"precision={best_metrics['eval_precision']:.4f}, "
        f"recall={best_metrics['eval_recall']:.4f}, "
        f"F1={best_metrics['eval_f1']:.4f}"
    )
    print(f"Saved model to {model_dir}.")


if __name__ == "__main__":
    main()
