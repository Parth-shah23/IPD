"""The standard ECPE benchmark (Xia and Ding, ACL 2019), machine-translated to English.

The corpus is 1,945 Chinese SINA news documents, split into clauses, with
emotion-cause pairs annotated at clause level. Each clause is translated on
its own with Helsinki-NLP/opus-mt-zh-en, so clause labels carry over exactly;
the English document is the translated clauses joined with commas, and every
emotion or cause span is one whole translated clause.

Documents where an emotion clause is its own cause (about a quarter of the
corpus) are excluded: the span tagger in ecpe_model.py cannot label one span
as both. The official fold 1 split from the authors' repository is used.
Data is downloaded into data_external/ and never committed.
"""
import ast
import json
import random
import urllib.request
from pathlib import Path

ECPE_URL = "https://raw.githubusercontent.com/NUSTM/ECPE/master/data_combine/{}"
ECPE_DIR = Path("data_external/ecpe_xia_ding")
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-zh-en"
TRANSLATION_CACHE = ECPE_DIR / "clauses_en.json"
EMOTION_MAP = {
    "happiness": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "disgust": "anger",
}


def download(name):
    path = ECPE_DIR / name
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} from the ECPE repository")
        with urllib.request.urlopen(ECPE_URL.format(name), timeout=120) as response:
            path.write_bytes(response.read())
    return path


def parse_documents(path):
    """Yield (doc_id, pairs, clauses) where clauses are (emotion_label, chinese_text)."""
    lines = open(path, encoding="utf-8").read().splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        doc_id, clause_count = lines[index].split()
        clause_count = int(clause_count)
        pairs = ast.literal_eval("[" + lines[index + 1].strip() + "]")
        clauses = []
        for line in lines[index + 2:index + 2 + clause_count]:
            _, emotion, _, text = line.split(",", 3)
            clauses.append((emotion, "".join(text.split())))
        yield doc_id, [tuple(pair) for pair in pairs], clauses
        index += 2 + clause_count


def translate(sentences, batch_size=64):
    """Translate Chinese clauses to English, caching results by source text."""
    cache = json.loads(TRANSLATION_CACHE.read_text(encoding="utf-8")) if TRANSLATION_CACHE.is_file() else {}
    missing = sorted({s for s in sentences if s not in cache})
    if missing:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Translating {len(missing)} clauses with {TRANSLATION_MODEL} on {device}")
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device).eval()
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_length=None, max_new_tokens=128, num_beams=4)
            for source, text in zip(batch, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
                cache[source] = text.strip()
            if (start // batch_size) % 50 == 0:
                print(f"  {min(start + batch_size, len(missing))}/{len(missing)}")
                TRANSLATION_CACHE.parent.mkdir(parents=True, exist_ok=True)
                TRANSLATION_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        TRANSLATION_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def to_example(doc_id, pairs, clauses, english):
    """Join translated clauses into one English text; each labelled clause becomes one span."""
    spans, parts, position = [], [], 0
    for _, source in clauses:
        text = english[source].rstrip(" ,.;:!?。，") or "..."
        spans.append((position, position + len(text)))
        parts.append(text)
        position += len(text) + 2
    text = ", ".join(parts) + "."

    emotions = []
    for emotion_clause in sorted({e for e, _ in pairs}):
        label = clauses[emotion_clause - 1][0].split("&")[0]
        emotions.append({
            "span": spans[emotion_clause - 1],
            "emotion": EMOTION_MAP.get(label, "joy"),
            "status": "stated",
            "causes": sorted({spans[c - 1] for e, c in pairs if e == emotion_clause}),
        })
    return {
        "id": f"ECPE-{doc_id}",
        "text": text,
        "structure": "news document",
        "emotions": emotions,
        "causes": sorted({spans[c - 1] for _, c in pairs}),
    }


def load_ecpe_splits(fold=1, max_docs=None):
    """Train, validation and test examples from the official fold, plus exclusion counts."""
    documents = list(parse_documents(download("all_data_pair.txt")))
    test_ids = {doc_id for doc_id, _, _ in parse_documents(download(f"fold{fold}_test.txt"))}
    if max_docs:
        # Smoke tests: keep a few documents from each side of the official split.
        documents = (
            [d for d in documents if d[0] in test_ids][:max_docs // 2]
            + [d for d in documents if d[0] not in test_ids][:max_docs - max_docs // 2]
        )
    usable, skipped = [], {"emotion clause is its own cause": 0, "clause is both emotion and cause": 0}
    for doc_id, pairs, clauses in documents:
        if any(e == c for e, c in pairs):
            skipped["emotion clause is its own cause"] += 1
        elif {e for e, _ in pairs} & {c for _, c in pairs}:
            skipped["clause is both emotion and cause"] += 1
        else:
            usable.append((doc_id, pairs, clauses))

    english = translate([source for _, _, clauses in usable for _, source in clauses])
    examples = [to_example(doc_id, pairs, clauses, english) for doc_id, pairs, clauses in usable]
    test = [e for e in examples if e["id"].removeprefix("ECPE-") in test_ids]
    train_pool = [e for e in examples if e["id"].removeprefix("ECPE-") not in test_ids]
    random.Random(42).shuffle(train_pool)
    validation_size = max(1, len(train_pool) // 10)
    return train_pool[validation_size:], train_pool[:validation_size], test, skipped


if __name__ == "__main__":
    train, validation, test, skipped = load_ecpe_splits()
    print(f"Skipped: {skipped}")
    print(f"Train {len(train)} | validation {len(validation)} | test {len(test)}")
    print(json.dumps(test[0], ensure_ascii=False, indent=1)[:1200])
