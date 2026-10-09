"""GoodNewsEveryone (Bostan, Kim and Klinger, LREC 2020) as an English ECPE benchmark.

Each headline has one dominant emotion, an emotion cue span and a cause span.
The annotations are CC BY 4.0, but the headlines belong to their publishers,
so the corpus is downloaded on demand into data_external/ and never committed.

Headlines whose cue lies inside the cause are excluded: the span tagger in
ecpe_model.py cannot represent nested spans.
"""
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

GNE_URL = (
    "https://www.ims.uni-stuttgart.de/documents/ressourcen/korpora/"
    "goodnewseveryone/goodnewseveryone-v1.0.zip"
)
GNE_FILE = Path("data_external/goodnewseveryone/gne-release-v1.0.jsonl")
SEED = 42
# GNE's 15 emotions mapped onto the journal label set. Pair F1 ignores the
# label, so this mapping only affects the emotion head during training.
EMOTION_MAP = {
    "anger": "anger",
    "annoyance": "anger",
    "disgust": "anger",
    "fear": "fear",
    "negative_anticipation_including_pessimism": "fear",
    "guilt": "guilt",
    "shame": "guilt",
    "joy": "joy",
    "love_including_like": "joy",
    "positive_anticipation_including_optimism": "joy",
    "trust": "contentment",
    "pride": "pride",
    "sadness": "sadness",
    "negative_surprise": "surprise",
    "positive_surprise": "surprise",
}


def download(path=GNE_FILE):
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading GoodNewsEveryone from {GNE_URL}")
    with urllib.request.urlopen(GNE_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(name for name in archive.namelist() if name.endswith("gne-release-v1.0.jsonl"))
    path.write_bytes(archive.read(member))
    return path


def gold_phrases(annotation):
    return [
        phrase.strip()
        for group in annotation.get("gold") or []
        for phrase in group
        if phrase and phrase.strip() and phrase.strip().lower() != "none"
    ]


def locate(headline, phrases):
    """Character span of the first phrase found in the headline, ignoring case."""
    lowered = headline.lower()
    for phrase in phrases:
        start = lowered.find(phrase.lower())
        end = start + len(phrase)
        if start != -1 and headline[start:end].lower() == phrase.lower():
            return start, end
    return None


def load_gne_examples(path=GNE_FILE):
    """Convert headlines to the example format produced by ecpe_model.load_examples."""
    examples = []
    skipped = {"no cue in headline": 0, "cue inside cause": 0, "unmapped emotion": 0}
    for line in open(download(path), encoding="utf-8"):
        record = json.loads(line)
        headline = record["headline"].strip()
        annotations = record["annotations"]
        emotion = EMOTION_MAP.get(annotations["dominant_emotion"]["gold"])
        if emotion is None:
            skipped["unmapped emotion"] += 1
            continue
        cue = locate(headline, gold_phrases(annotations["cue"]))
        if cue is None:
            skipped["no cue in headline"] += 1
            continue
        cause = locate(headline, gold_phrases(annotations["cause"]))
        if cause is not None and cue[0] < cause[1] and cause[0] < cue[1]:
            skipped["cue inside cause"] += 1
            continue
        examples.append({
            "id": f"GNE-{record['id']}",
            "text": headline,
            "structure": "news headline",
            "emotions": [{
                "span": cue,
                "emotion": emotion,
                "status": "stated" if cause else "none",
                "causes": [cause] if cause else [],
            }],
            "causes": [cause] if cause else [],
        })
    return examples, skipped


def split_gne(examples, seed=SEED):
    """Fixed 80/10/10 train/validation/test split (GNE has no official split)."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    validation_start = int(0.8 * len(shuffled))
    test_start = int(0.9 * len(shuffled))
    return shuffled[:validation_start], shuffled[validation_start:test_start], shuffled[test_start:]


if __name__ == "__main__":
    examples, skipped = load_gne_examples()
    train, validation, test = split_gne(examples)
    print(f"Usable headlines: {len(examples)} | skipped: {skipped}")
    print(f"Train {len(train)} | validation {len(validation)} | test {len(test)}")
    print(f"With a stated cause: {sum(bool(e['causes']) for e in examples)}")
