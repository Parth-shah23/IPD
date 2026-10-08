import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from fine_tune_ner import LABEL_LIST, SEED, entity_metrics, load_and_prepare_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved journal NER model.")
    parser.add_argument("--data", default="ECPE_Synthetic_Journal_Dataset_v3.xlsx")
    parser.add_argument("--model", default="best_ner_model")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    records = load_and_prepare_data(args.data)
    _, validation_records = train_test_split(
        records, test_size=0.2, random_state=SEED, shuffle=True
    )
    dataset = Dataset.from_list(validation_records)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model_labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
    if model_labels != LABEL_LIST:
        raise ValueError(
            f"Model labels do not match the v3 dataset: {model_labels!r}"
        )

    def tokenize_and_align_labels(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            max_length=512,
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

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="ner-evaluation",
            per_device_eval_batch_size=8,
            report_to="none",
            use_cpu=not torch.cuda.is_available(),
        ),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.predict(tokenized_dataset)
    predictions = np.argmax(result.predictions, axis=2)
    predicted_sequences = []
    expected_sequences = []
    for prediction, labels in zip(predictions, result.label_ids):
        active = labels != -100
        predicted_sequences.append([LABEL_LIST[index] for index in prediction[active]])
        expected_sequences.append([LABEL_LIST[index] for index in labels[active]])

    precision, recall, f1 = entity_metrics(predicted_sequences, expected_sequences)
    print(f"Validation entries: {len(validation_records)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    log_path = Path("evaluation_log.csv")
    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "FINETUNED BERT (v3 validation)",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            "N/A",
        ])
    print(f"Logged validation result to {log_path}.")


if __name__ == "__main__":
    main()
