"""Train and evaluate the span-based ECPE model.

Default mode trains a final model on a stratified 90/10 split and saves it.
--cv runs stratified k-fold cross-validation over entries and reports
out-of-fold pair F1 with bootstrap confidence intervals.
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer

from ecpe_model import (
    DATA_FILE,
    MODEL_CHECKPOINT,
    MODEL_DIR,
    ECPEModel,
    collate,
    compute_loss,
    encode_examples,
    load_examples,
    predict,
    save_model,
)

SEED = 42
RESULTS_DIR = "ecpe_results"
BOOTSTRAP_SAMPLES = 2000
RELAXED_OVERLAP = 0.5
METRICS = ("emotion_span", "emotion_labelled", "cause_span", "pair", "triplet")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Scoring ──────────────────────────────────────────────────────────────────

def gold_items(example):
    emotions = example["emotions"]
    return {
        "emotion_span": [(e["span"],) for e in emotions],
        "emotion_labelled": [(e["span"], e["emotion"]) for e in emotions],
        "cause_span": [(c,) for c in example["causes"]],
        "pair": [(e["span"], c) for e in emotions for c in e["causes"]],
        "triplet": [(e["span"], e["emotion"], c) for e in emotions for c in e["causes"]],
    }


def predicted_items(prediction):
    emotions = [
        {
            "span": (e["start"], e["end"]),
            "emotion": e["emotion"],
            "causes": [(c["start"], c["end"]) for c in e["causes"]],
        }
        for e in prediction["emotions"]
    ]
    causes = sorted({c for e in emotions for c in e["causes"]})
    return gold_items({"emotions": emotions, "causes": causes})


def span_overlap(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def items_match(predicted, gold, relaxed):
    for p, g in zip(predicted, gold):
        if isinstance(p, tuple):
            if relaxed and span_overlap(p, g) < RELAXED_OVERLAP:
                return False
            if not relaxed and p != g:
                return False
        elif p != g:
            return False
    return True


def match_counts(predicted, gold, relaxed):
    """Greedy one-to-one matching; returns (true positives, false positives, false negatives)."""
    unmatched = list(gold)
    true_positive = 0
    for item in predicted:
        for index, candidate in enumerate(unmatched):
            if items_match(item, candidate, relaxed):
                unmatched.pop(index)
                true_positive += 1
                break
    return true_positive, len(predicted) - true_positive, len(unmatched)


def entry_counts(examples, predictions):
    """Array of shape (entries, 2 match modes, metrics, 3 counts)."""
    counts = np.zeros((len(examples), 2, len(METRICS), 3), dtype=np.int64)
    for row, (example, prediction) in enumerate(zip(examples, predictions)):
        gold, predicted = gold_items(example), predicted_items(prediction)
        for mode, relaxed in enumerate((False, True)):
            for column, metric in enumerate(METRICS):
                counts[row, mode, column] = match_counts(predicted[metric], gold[metric], relaxed)
    return counts


def f1_from_counts(counts):
    true_positive, false_positive, false_negative = counts[..., 0], counts[..., 1], counts[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(true_positive + false_positive > 0, true_positive / (true_positive + false_positive), 0.0)
        recall = np.where(true_positive + false_negative > 0, true_positive / (true_positive + false_negative), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return precision, recall, f1


def status_accuracy(examples, predictions):
    """Cause-status accuracy on predicted emotion spans that overlap a gold emotion span."""
    correct = total = 0
    for example, prediction in zip(examples, predictions):
        for predicted in prediction["emotions"]:
            span = (predicted["start"], predicted["end"])
            for gold in example["emotions"]:
                if span_overlap(span, gold["span"]) >= RELAXED_OVERLAP:
                    total += 1
                    correct += predicted["cause_status"] == gold["status"]
                    break
    return correct / total if total else 0.0


def score(examples, predictions, bootstrap=False):
    counts = entry_counts(examples, predictions)
    precision, recall, f1 = f1_from_counts(counts.sum(axis=0))
    scores = {}
    for mode, name in enumerate(("exact", "relaxed")):
        for column, metric in enumerate(METRICS):
            scores[f"{metric}_{name}"] = {
                "precision": float(precision[mode, column]),
                "recall": float(recall[mode, column]),
                "f1": float(f1[mode, column]),
            }
    scores["cause_status_accuracy"] = status_accuracy(examples, predictions)

    if bootstrap:
        rng = np.random.default_rng(SEED)
        samples = rng.integers(0, len(examples), size=(BOOTSTRAP_SAMPLES, len(examples)))
        resampled = f1_from_counts(counts[samples].sum(axis=1))[2]
        for mode, name in enumerate(("exact", "relaxed")):
            for column, metric in enumerate(METRICS):
                low, high = np.percentile(resampled[:, mode, column], [2.5, 97.5])
                scores[f"{metric}_{name}"]["f1_ci95"] = [float(low), float(high)]
    return scores


def split_entries(examples, test_size=0.1):
    """Stratify by structure when every structure can appear on both sides of the split."""
    structures = [e["structure"] for e in examples]
    counts = {structure: structures.count(structure) for structure in structures}
    can_stratify = min(counts.values()) >= 2 and math.ceil(test_size * len(examples)) >= len(counts)
    return train_test_split(
        examples,
        test_size=test_size,
        random_state=SEED,
        stratify=structures if can_stratify else None,
    )


def selection_key(scores):
    return scores["pair_exact"]["f1"], scores["pair_relaxed"]["f1"]


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(train_examples, validation_examples, args, device, init_state=None):
    """Fine-tune from args.model, or from init_state (a full ECPEModel state dict) when given."""
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not tokenizer.is_fast:
        raise RuntimeError(f"{args.model} has no fast tokenizer; character offsets are required.")
    model = ECPEModel.from_checkpoint(args.model)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    train_data = encode_examples(train_examples, tokenizer)

    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    steps_per_epoch = math.ceil(len(train_data) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup_steps)
        * max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(SEED)

    best_key, best_state, best_scores, best_epoch, stale_epochs = None, None, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_data), generator=generator).tolist()
        batches = [
            collate([train_data[i] for i in order[start:start + args.batch_size]], tokenizer.pad_token_id)
            for start in range(0, len(order), args.batch_size)
        ]
        running_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(batches, start=1):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = compute_loss(model, batch, device)
            scaler.scale(loss / args.grad_accum).backward()
            running_loss += loss.item()
            if step % args.grad_accum == 0 or step == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        predictions = predict(model, tokenizer, [e["text"] for e in validation_examples], device)
        scores = score(validation_examples, predictions)
        key = selection_key(scores)
        print(
            f"  epoch {epoch:2d} | loss {running_loss / len(batches):.4f} | "
            f"val pair F1 exact {key[0]:.3f} relaxed {key[1]:.3f} | "
            f"emotion span F1 {scores['emotion_span_relaxed']['f1']:.3f}"
        )
        if best_key is None or key > best_key:
            best_key, best_scores, best_epoch, stale_epochs = key, scores, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if epoch >= args.min_epochs and stale_epochs >= args.patience:
                print(f"  early stop: no improvement for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    return model, tokenizer, best_scores, best_epoch


# ── Reporting ────────────────────────────────────────────────────────────────

def print_scores(scores, title):
    print(f"\n{title}")
    print(f"{'metric':<18}{'match':<9}{'P':>7}{'R':>7}{'F1':>7}   95% CI")
    for metric in METRICS:
        for name in ("exact", "relaxed"):
            values = scores[f"{metric}_{name}"]
            interval = values.get("f1_ci95")
            interval_text = f"{interval[0]:.3f}-{interval[1]:.3f}" if interval else ""
            print(
                f"{metric:<18}{name:<9}{values['precision']:7.3f}{values['recall']:7.3f}"
                f"{values['f1']:7.3f}   {interval_text}"
            )
    print(f"cause status accuracy (matched emotion spans): {scores['cause_status_accuracy']:.3f}")


def scores_by_structure(examples, predictions):
    """Pair F1 per structure; structures with no gold pairs get None and are judged by false positives."""
    groups = {}
    for example, prediction in zip(examples, predictions):
        groups.setdefault(example["structure"], ([], []))
        groups[example["structure"]][0].append(example)
        groups[example["structure"]][1].append(prediction)
    results = {}
    for structure, (group_examples, group_predictions) in sorted(groups.items()):
        group_scores = score(group_examples, group_predictions)
        gold_pairs = sum(len(gold_items(e)["pair"]) for e in group_examples)
        predicted_pairs = sum(len(predicted_items(p)["pair"]) for p in group_predictions)
        results[structure] = {
            "entries": len(group_examples),
            "gold_pairs": gold_pairs,
            "predicted_pairs": predicted_pairs,
            "pair_exact_f1": group_scores["pair_exact"]["f1"] if gold_pairs else None,
            "pair_relaxed_f1": group_scores["pair_relaxed"]["f1"] if gold_pairs else None,
            "emotion_span_relaxed_f1": group_scores["emotion_span_relaxed"]["f1"],
        }
    return results


def serialise_gold(example):
    return {
        "emotions": [
            {
                "emotion_clause": example["text"][e["span"][0]:e["span"][1]],
                "emotion": e["emotion"],
                "cause_status": e["status"],
                "causes": [example["text"][c[0]:c[1]] for c in e["causes"]],
            }
            for e in example["emotions"]
        ]
    }


def run_cross_validation(examples, args, device):
    structures = [example["structure"] for example in examples]
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    predictions = [None] * len(examples)
    fold_of = [None] * len(examples)
    for fold, (train_index, test_index) in enumerate(folds.split(examples, structures), start=1):
        train_part = [examples[i] for i in train_index]
        train_examples, validation_examples = split_entries(train_part)
        test_examples = [examples[i] for i in test_index]
        print(
            f"\nFold {fold}/{args.folds}: train {len(train_examples)} | "
            f"validation {len(validation_examples)} | test {len(test_examples)}"
        )
        model, tokenizer, _, best_epoch = train_model(train_examples, validation_examples, args, device)
        fold_predictions = predict(model, tokenizer, [e["text"] for e in test_examples], device)
        fold_scores = score(test_examples, fold_predictions)
        print(
            f"  fold {fold} test pair F1 exact {fold_scores['pair_exact']['f1']:.3f} "
            f"relaxed {fold_scores['pair_relaxed']['f1']:.3f} (best epoch {best_epoch})"
        )
        for index, prediction in zip(test_index, fold_predictions):
            predictions[index] = prediction
            fold_of[index] = fold
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = score(examples, predictions, bootstrap=True)
    by_structure = scores_by_structure(examples, predictions)
    print_scores(scores, f"Out-of-fold scores over {len(examples)} entries ({args.folds}-fold CV)")
    print(f"\n{'structure':<28}{'n':>4}{'gold pairs':>12}{'pred pairs':>12}{'pair exact':>12}{'pair relaxed':>14}")
    for structure, values in by_structure.items():
        if values["gold_pairs"]:
            f1_text = f"{values['pair_exact_f1']:>12.3f}{values['pair_relaxed_f1']:>14.3f}"
        else:
            f1_text = "   n/a: every predicted pair is a false positive"
        print(
            f"{structure:<28}{values['entries']:>4}{values['gold_pairs']:>12}"
            f"{values['predicted_pairs']:>12}{f1_text}"
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "cv_metrics.json").write_text(
        json.dumps({"config": vars(args), "scores": scores, "by_structure": by_structure}, indent=2),
        encoding="utf-8",
    )
    with open(results_dir / "oof_predictions.jsonl", "w", encoding="utf-8") as handle:
        for example, prediction, fold in zip(examples, predictions, fold_of):
            handle.write(json.dumps({
                "id": example["id"],
                "fold": fold,
                "structure": example["structure"],
                "text": example["text"],
                "gold": serialise_gold(example),
                "predicted": prediction,
            }, ensure_ascii=False) + "\n")
    print(f"\nSaved metrics and out-of-fold predictions to {results_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train the span-based ECPE model on journal entries.")
    parser.add_argument("--data", default=DATA_FILE, help="Workbook containing Entries and Pairs sheets.")
    parser.add_argument("--model", default=MODEL_CHECKPOINT, help="Hugging Face encoder checkpoint.")
    parser.add_argument("--output-dir", default=MODEL_DIR)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--cv", action="store_true", help="Run k-fold cross-validation instead of training a final model.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5, help="Encoder learning rate.")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Learning rate for the task heads.")
    parser.add_argument("--max-entries", type=int, help="Use only the first N entries (quick smoke tests).")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    examples = load_examples(args.data)
    if args.max_entries:
        examples = examples[:args.max_entries]
    pairs = sum(len(e["causes"]) for e in examples)
    print(f"Entries: {len(examples)} | stated cause spans: {pairs} | model: {args.model} | device: {device}")

    if args.cv:
        run_cross_validation(examples, args, device)
        return

    train_examples, validation_examples = split_entries(examples)
    print(f"Train: {len(train_examples)} | Validation: {len(validation_examples)}")
    model, tokenizer, scores, best_epoch = train_model(train_examples, validation_examples, args, device)
    print_scores(scores, f"Validation scores at best epoch {best_epoch} (used for model selection, so optimistic)")
    save_model(model, tokenizer, args.output_dir, metadata={
        "base_model": args.model,
        "best_epoch": best_epoch,
        "validation_pair_exact_f1": scores["pair_exact"]["f1"],
        "validation_pair_relaxed_f1": scores["pair_relaxed"]["f1"],
        "data": str(args.data),
        "seed": SEED,
    })
    print(f"Saved model to {args.output_dir}/")


if __name__ == "__main__":
    main()
