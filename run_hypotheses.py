"""Test H1 (transfer) and H2 (adaptation) from the project plan.

1. Train the span model on the benchmark only (--benchmark ecpe: Xia and Ding
   2019, translated to English; --benchmark gne: GoodNewsEveryone): the
   off-the-shelf model.
2. Score it on the benchmark test split and on every journal entry (it has never
   seen a journal entry).
3. Fine-tune it on journal entries with the same 5 folds as
   `train_ecpe.py --cv` and score the held-out folds.

H1: off-the-shelf pair F1 on journals is lower than on the benchmark
    (one-sided bootstrap, entries resampled independently in each set).
H2: fine-tuned pair F1 on journals is higher than off-the-shelf pair F1
    (one-sided paired bootstrap over journal entries, Holm across comparisons).
    When journal-only out-of-fold predictions from `train_ecpe.py --cv` are
    present, journal-only training is a second H2 comparison.

Primary metric: relaxed pair F1. Exact pair F1 is reported alongside.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from benchmark_ecpe import load_ecpe_splits
from benchmark_gne import load_gne_examples, split_gne
from ecpe_model import DATA_FILE, MODEL_CHECKPOINT, load_examples, predict, save_model
from train_ecpe import METRICS, SEED, entry_counts, f1_from_counts, set_seed, split_entries, train_model

BOOTSTRAP_SAMPLES = 10000
ALPHA = 0.05
PAIR = METRICS.index("pair")
MODES = ("exact", "relaxed")


def pair_scores(counts):
    precision, recall, f1 = f1_from_counts(counts.sum(axis=0))
    return {
        mode: {
            "precision": float(precision[index, PAIR]),
            "recall": float(recall[index, PAIR]),
            "f1": float(f1[index, PAIR]),
        }
        for index, mode in enumerate(MODES)
    }


def bootstrap_pair_f1(counts, samples):
    """Pair F1 (exact, relaxed) for each row of resampled entry indices."""
    out = []
    for chunk in np.array_split(samples, max(1, len(samples) // 500)):
        out.append(f1_from_counts(counts[chunk].sum(axis=1))[2][:, :, PAIR])
    return np.concatenate(out)


def one_sided_test(differences, observed):
    """p-value for H1: difference > 0, with the +1 correction for bootstrap p-values."""
    p_value = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": float(observed), "ci95": [float(low), float(high)], "p_value": float(p_value)}


def holm(p_values):
    """Holm-adjusted p-values, in input order."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def texts(examples):
    return [example["text"] for example in examples]


def cross_validate_from(examples, init_state, args, device):
    """Out-of-fold predictions after fine-tuning from init_state, using train_ecpe.py's folds."""
    structures = [example["structure"] for example in examples]
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    predictions = [None] * len(examples)
    fold_of = [None] * len(examples)
    for fold, (train_index, test_index) in enumerate(folds.split(examples, structures), start=1):
        train_examples, validation_examples = split_entries([examples[i] for i in train_index])
        print(f"\nFine-tune fold {fold}/{args.folds}: train {len(train_examples)} | validation {len(validation_examples)} | test {len(test_index)}")
        model, tokenizer, _, _ = train_model(train_examples, validation_examples, args, device, init_state=init_state)
        for index, prediction in zip(test_index, predict(model, tokenizer, [examples[i]["text"] for i in test_index], device)):
            predictions[index] = prediction
            fold_of[index] = fold
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return predictions, fold_of


def load_oof(path, examples, fold_of):
    """Journal-only predictions from train_ecpe.py --cv, checked to use the same folds."""
    rows = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        rows[row["id"]] = row
    if set(rows) != {example["id"] for example in examples}:
        print(f"Skipping {path}: it covers different entries.")
        return None
    if [rows[example["id"]]["fold"] for example in examples] != fold_of:
        print(f"Skipping {path}: its folds differ from this run.")
        return None
    return [rows[example["id"]]["predicted"] for example in examples]


def write_predictions(path, examples, predictions):
    with open(path, "w", encoding="utf-8") as handle:
        for example, prediction in zip(examples, predictions):
            handle.write(json.dumps({"id": example["id"], "text": example["text"], "predicted": prediction}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the H1 and H2 experiments for the ECPE model.")
    parser.add_argument("--data", default=DATA_FILE, help="Journal workbook (synthetic now; human test set later).")
    parser.add_argument("--model", default=MODEL_CHECKPOINT)
    parser.add_argument("--journal-only-oof", default="ecpe_results/oof_predictions.jsonl")
    parser.add_argument("--benchmark", choices=["ecpe", "gne"], default="ecpe",
                        help="ecpe: Xia and Ding (2019), translated to English; gne: GoodNewsEveryone headlines.")
    parser.add_argument("--results-dir", help="Default: ecpe_results_hypotheses_<benchmark>.")
    parser.add_argument("--offshelf-dir", help="Default: ecpe_offshelf_model_<benchmark>.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-entries", type=int, help="Quick smoke tests only.")
    parser.add_argument("--max-headlines", type=int, help="Quick smoke tests only (benchmark documents).")
    args = parser.parse_args()
    args.results_dir = args.results_dir or f"ecpe_results_hypotheses_{args.benchmark}"
    args.offshelf_dir = args.offshelf_dir or f"ecpe_offshelf_model_{args.benchmark}"

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    journal = load_examples(args.data)[:args.max_entries]
    if args.benchmark == "ecpe":
        gne_train, gne_validation, gne_test, skipped = load_ecpe_splits(max_docs=args.max_headlines)
    else:
        headlines, skipped = load_gne_examples()
        gne_train, gne_validation, gne_test = split_gne(headlines[:args.max_headlines])
    print(f"Journal entries: {len(journal)} | benchmark: {args.benchmark} (skipped {skipped})")
    print(f"Benchmark train {len(gne_train)} | validation {len(gne_validation)} | test {len(gne_test)} | device {device}")

    print(f"\n== Stage 1: train the off-the-shelf model on the {args.benchmark} benchmark only ==")
    model, tokenizer, _, best_epoch = train_model(gne_train, gne_validation, args, device)
    offshelf_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    save_model(model, tokenizer, args.offshelf_dir, metadata={"base_model": args.model, "trained_on": args.benchmark, "best_epoch": best_epoch, "seed": SEED})

    print("\n== Stage 2: score the off-the-shelf model ==")
    news_predictions = predict(model, tokenizer, texts(gne_test), device)
    offshelf_predictions = predict(model, tokenizer, texts(journal), device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n== Stage 3: fine-tune the off-the-shelf model on journal entries ==")
    adapted_predictions, fold_of = cross_validate_from(journal, offshelf_state, args, device)
    journal_only_predictions = None
    if Path(args.journal_only_oof).is_file() and not args.max_entries:
        journal_only_predictions = load_oof(args.journal_only_oof, journal, fold_of)

    counts = {
        "offshelf_news": entry_counts(gne_test, news_predictions),
        "offshelf_journal": entry_counts(journal, offshelf_predictions),
        "adapted_journal": entry_counts(journal, adapted_predictions),
    }
    if journal_only_predictions is not None:
        counts["journal_only_journal"] = entry_counts(journal, journal_only_predictions)
    scores = {name: pair_scores(c) for name, c in counts.items()}

    rng = np.random.default_rng(SEED)
    news_samples = rng.integers(0, len(gne_test), size=(BOOTSTRAP_SAMPLES, len(gne_test)))
    journal_samples = rng.integers(0, len(journal), size=(BOOTSTRAP_SAMPLES, len(journal)))
    boot = {
        name: bootstrap_pair_f1(c, news_samples if name == "offshelf_news" else journal_samples)
        for name, c in counts.items()
    }

    tests = {}
    for mode_index, mode in enumerate(MODES):
        h1 = one_sided_test(
            boot["offshelf_news"][:, mode_index] - boot["offshelf_journal"][:, mode_index],
            scores["offshelf_news"][mode]["f1"] - scores["offshelf_journal"][mode]["f1"],
        )
        h2 = {}
        for name in ("adapted_journal", "journal_only_journal"):
            if name in boot:
                h2[name] = one_sided_test(
                    boot[name][:, mode_index] - boot["offshelf_journal"][:, mode_index],
                    scores[name][mode]["f1"] - scores["offshelf_journal"][mode]["f1"],
                )
        for name, adjusted in zip(h2, holm([test["p_value"] for test in h2.values()])):
            h2[name]["p_holm"] = adjusted
        tests[mode] = {"H1_news_minus_journal": h1, "H2_vs_offshelf": h2}

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hypothesis_results.json").write_text(json.dumps({
        "config": vars(args),
        "sizes": {"benchmark": args.benchmark, "journal_entries": len(journal), "benchmark_train": len(gne_train), "benchmark_test": len(gne_test), "benchmark_skipped": skipped},
        "pair_scores": scores,
        "tests": tests,
        "alpha": ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }, indent=2), encoding="utf-8")
    write_predictions(results_dir / "offshelf_news_test_predictions.jsonl", gne_test, news_predictions)
    write_predictions(results_dir / "offshelf_journal_predictions.jsonl", journal, offshelf_predictions)
    write_predictions(results_dir / "adapted_journal_oof_predictions.jsonl", journal, adapted_predictions)

    print("\n== Pair scores (relaxed / exact F1) ==")
    for name, value in scores.items():
        print(f"{name:<24} P {value['relaxed']['precision']:.3f}  R {value['relaxed']['recall']:.3f}  "
              f"F1 {value['relaxed']['f1']:.3f}  | exact F1 {value['exact']['f1']:.3f}")
    for mode in MODES:
        print(f"\n== Tests on {mode} pair F1 (alpha {ALPHA}) ==")
        h1 = tests[mode]["H1_news_minus_journal"]
        print(f"H1 benchmark - journal: {h1['difference']:+.3f}  CI {h1['ci95'][0]:+.3f}..{h1['ci95'][1]:+.3f}  p {h1['p_value']:.4f}")
        for name, test in tests[mode]["H2_vs_offshelf"].items():
            print(f"H2 {name} - offshelf: {test['difference']:+.3f}  CI {test['ci95'][0]:+.3f}..{test['ci95'][1]:+.3f}  "
                  f"p {test['p_value']:.4f}  Holm p {test['p_holm']:.4f}")
    print(f"\nSaved results to {results_dir}/")


if __name__ == "__main__":
    main()
