"""Span-based emotion-cause pair extraction (ECPE) for journal entries.

The model reads a whole entry, tags emotion and cause spans with BIO labels,
classifies the emotion of each emotion span, decides whether its cause is
stated, implicit or absent, and links emotion spans to cause spans.
"""
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

DATA_FILE = "ECPE_Synthetic_Journal_Dataset_v3.xlsx"
MODEL_CHECKPOINT = "roberta-base"
MODEL_DIR = "best_ecpe_model"
WEIGHTS_FILE = "ecpe_model.pt"
LABELS_FILE = "ecpe_labels.json"
MAX_LENGTH = 512
TAGS = ["O", "B-EMO", "I-EMO", "B-CAU", "I-CAU"]
TAG2ID = {tag: index for index, tag in enumerate(TAGS)}
# "neutral" is not a span label: entries without feeling simply have no emotion spans.
EMOTIONS = [
    "joy", "contentment", "pride", "gratitude", "sadness",
    "anger", "fear", "stress", "guilt", "surprise",
]
CAUSE_STATUSES = ["stated", "implicit", "none"]
# Lower bounds of token-distance buckets between an emotion span and a cause span.
DISTANCE_BUCKETS = (0, 1, 2, 4, 8, 16, 32, 64)
DISTANCE_DIM = 32
# Every gold emotion has exactly one stated cause, and no gold cause clause is
# shorter than four words, so decoding links one cause and drops tiny cause spans.
MAX_CAUSES_PER_EMOTION = 1
MIN_CAUSE_WORDS = 3


def clean_cell(value):
    if pd.isna(value) or not str(value).strip():
        return None
    return str(value).strip()


def find_span(text, clause, entry_id):
    start = text.find(clause)
    if start == -1:
        raise ValueError(f"Clause {clause!r} not found in entry {entry_id!r}")
    return start, start + len(clause)


def load_examples(filepath=DATA_FILE):
    """Read Entries and Pairs sheets into one example per entry with character spans."""
    if not Path(filepath).is_file():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    entries = pd.read_excel(filepath, sheet_name="Entries")
    pairs = pd.read_excel(filepath, sheet_name="Pairs")
    required_entry_columns = {"ID", "Text"}
    required_pair_columns = {"entry_id", "emotion_clause", "emotion", "cause_clause"}
    if not required_entry_columns.issubset(entries.columns):
        raise ValueError(f"Entries must contain columns {sorted(required_entry_columns)}")
    if not required_pair_columns.issubset(pairs.columns):
        raise ValueError(f"Pairs must contain columns {sorted(required_pair_columns)}")

    pairs_by_id = {entry_id: rows for entry_id, rows in pairs.groupby("entry_id")}
    examples = []
    for _, row in entries.iterrows():
        if pd.isna(row["Text"]) or not str(row["Text"]).strip():
            continue
        entry_id = str(row["ID"])
        text = str(row["Text"])
        emotions = {}
        causes = []
        rows = pairs_by_id.get(row["ID"])
        for _, pair in (rows.iterrows() if rows is not None else []):
            emotion_clause = clean_cell(pair["emotion_clause"])
            if emotion_clause is None:
                continue
            emotion = str(pair["emotion"]).strip().lower()
            if emotion not in EMOTIONS:
                raise ValueError(f"Unsupported emotion {emotion!r} in entry {entry_id!r}")
            emotion_span = find_span(text, emotion_clause, entry_id)

            cause_clause = clean_cell(pair["cause_clause"])
            if cause_clause is None:
                status, cause_span = "none", None
            elif cause_clause.lower() == "implicit":
                status, cause_span = "implicit", None
            else:
                status, cause_span = "stated", find_span(text, cause_clause, entry_id)
                if cause_span not in causes:
                    causes.append(cause_span)

            # One emotion clause listed on several rows means one emotion with several causes.
            record = emotions.setdefault(
                emotion_span,
                {"span": emotion_span, "emotion": emotion, "status": status, "causes": []},
            )
            if status == "stated":
                record["status"] = "stated"
                if cause_span not in record["causes"]:
                    record["causes"].append(cause_span)

        examples.append({
            "id": entry_id,
            "text": text,
            "structure": clean_cell(row.get("planned_structure")) or "unknown",
            "emotions": sorted(emotions.values(), key=lambda record: record["span"]),
            "causes": sorted(causes),
        })

    if not examples:
        raise ValueError("No non-empty journal entries were found.")
    return examples


def char_to_token_span(offsets, span):
    start, end = span
    tokens = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]
    return (tokens[0], tokens[-1]) if tokens else None


def encode_examples(examples, tokenizer):
    """Tokenize entries and convert gold character spans to token spans and BIO tags."""
    encoded = []
    for example in examples:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
        )
        offsets = tokenized["offset_mapping"]
        tags = [-100 if start == end else TAG2ID["O"] for start, end in offsets]

        def mark(token_span, kind):
            start, end = token_span
            tags[start] = TAG2ID[f"B-{kind}"]
            for index in range(start + 1, end + 1):
                tags[index] = TAG2ID[f"I-{kind}"]

        cause_index = {}
        cause_spans = []
        for span in example["causes"]:
            token_span = char_to_token_span(offsets, span)
            if token_span is None:
                continue
            cause_index[span] = len(cause_spans)
            cause_spans.append(token_span)
            mark(token_span, "CAU")

        emotion_spans, emotion_labels, status_labels, links = [], [], [], []
        for emotion in example["emotions"]:
            token_span = char_to_token_span(offsets, emotion["span"])
            if token_span is None:
                continue
            emotion_position = len(emotion_spans)
            emotion_spans.append(token_span)
            emotion_labels.append(EMOTIONS.index(emotion["emotion"]))
            status_labels.append(CAUSE_STATUSES.index(emotion["status"]))
            links.extend(
                (emotion_position, cause_index[span])
                for span in emotion["causes"]
                if span in cause_index
            )
            mark(token_span, "EMO")

        encoded.append({
            "input_ids": tokenized["input_ids"],
            "tags": tags,
            "emotion_spans": emotion_spans,
            "emotion_labels": emotion_labels,
            "status_labels": status_labels,
            "cause_spans": cause_spans,
            "links": links,
        })
    return encoded


def collate(batch, pad_token_id):
    length = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    tags = torch.full((len(batch), length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        size = len(item["input_ids"])
        input_ids[row, :size] = torch.tensor(item["input_ids"])
        attention_mask[row, :size] = 1
        tags[row, :size] = torch.tensor(item["tags"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "tags": tags, "items": batch}


def distance_bucket(distance):
    magnitude = sum(1 for bound in DISTANCE_BUCKETS[1:] if abs(distance) >= bound)
    return magnitude + (len(DISTANCE_BUCKETS) if distance < 0 else 0)


class ECPEModel(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tagger = nn.Linear(hidden, len(TAGS))
        self.span_encoder = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.emotion_head = nn.Linear(hidden, len(EMOTIONS))
        self.status_head = nn.Linear(hidden, len(CAUSE_STATUSES))
        self.distance_embedding = nn.Embedding(2 * len(DISTANCE_BUCKETS), DISTANCE_DIM)
        self.pair_head = nn.Sequential(
            nn.Linear(3 * hidden + DISTANCE_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint=MODEL_CHECKPOINT):
        return cls(AutoModel.from_pretrained(checkpoint))

    def encode(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        return hidden, self.tagger(hidden)

    def span_vectors(self, hidden, spans):
        """hidden: (sequence, hidden) for one entry; spans: inclusive token spans."""
        vectors = [
            torch.cat([hidden[start], hidden[end], hidden[start:end + 1].mean(dim=0)])
            for start, end in spans
        ]
        return self.span_encoder(torch.stack(vectors))

    def pair_logits(self, emotion_vectors, cause_vectors, emotion_spans, cause_spans):
        emotions = emotion_vectors.unsqueeze(1).expand(-1, len(cause_spans), -1)
        causes = cause_vectors.unsqueeze(0).expand(len(emotion_spans), -1, -1)
        buckets = torch.tensor(
            [
                [distance_bucket(cause[0] - emotion[0]) for cause in cause_spans]
                for emotion in emotion_spans
            ],
            device=emotion_vectors.device,
        )
        features = torch.cat(
            [emotions, causes, emotions * causes, self.distance_embedding(buckets)], dim=-1
        )
        return self.pair_head(features).squeeze(-1)


def compute_loss(model, batch, device):
    """Tagging loss plus emotion, cause-status and pairing losses on gold spans."""
    hidden, tag_logits = model.encode(
        batch["input_ids"].to(device), batch["attention_mask"].to(device)
    )
    tags = batch["tags"].to(device)
    tag_loss = F.cross_entropy(
        tag_logits.reshape(-1, len(TAGS)).float(), tags.reshape(-1), ignore_index=-100
    )

    emotion_losses, status_losses, pair_losses = [], [], []
    for row, item in enumerate(batch["items"]):
        if not item["emotion_spans"]:
            continue
        emotion_vectors = model.span_vectors(hidden[row], item["emotion_spans"]).float()
        emotion_losses.append(F.cross_entropy(
            model.emotion_head(emotion_vectors),
            torch.tensor(item["emotion_labels"], device=device),
        ))
        status_losses.append(F.cross_entropy(
            model.status_head(emotion_vectors),
            torch.tensor(item["status_labels"], device=device),
        ))
        if item["cause_spans"]:
            cause_vectors = model.span_vectors(hidden[row], item["cause_spans"]).float()
            logits = model.pair_logits(
                emotion_vectors, cause_vectors, item["emotion_spans"], item["cause_spans"]
            ).float()
            targets = torch.zeros_like(logits)
            for emotion_position, cause_position in item["links"]:
                targets[emotion_position, cause_position] = 1.0
            pair_losses.append(F.binary_cross_entropy_with_logits(logits, targets))

    loss = tag_loss
    for losses in (emotion_losses, status_losses, pair_losses):
        if losses:
            loss = loss + torch.stack(losses).mean()
    return loss


def decode_tags(tag_ids):
    """Return (kind, start, end) inclusive spans from a BIO sequence, tolerating stray I- tags."""
    spans = []
    start = kind = None
    for index, tag in enumerate([TAGS[tag_id] for tag_id in tag_ids] + ["O"]):
        if tag == "O" or tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != kind):
            if kind is not None:
                spans.append((kind, start, index - 1))
            start, kind = (index, tag[2:]) if tag != "O" else (None, None)
    return spans


@torch.inference_mode()
def predict(model, tokenizer, texts, device, batch_size=8):
    """Predict emotion-cause structure for each text, with character offsets."""
    model.eval()
    results = []
    for batch_start in range(0, len(texts), batch_size):
        chunk = texts[batch_start:batch_start + batch_size]
        tokenized = tokenizer(
            chunk,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = tokenized.pop("offset_mapping").tolist()
        hidden, tag_logits = model.encode(
            tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)
        )
        tag_confidence, tag_ids = torch.softmax(tag_logits.float(), dim=-1).max(dim=-1)
        tag_batch = tag_ids.cpu().tolist()
        tag_confidence = tag_confidence.cpu()

        for row, text in enumerate(chunk):
            offsets = offsets_batch[row]
            valid = [index for index, (start, end) in enumerate(offsets) if end > start]
            spans = decode_tags([tag_batch[row][index] for index in valid])

            def char_span(token_span):
                return offsets[token_span[0]][0], offsets[token_span[1]][1]

            def span_confidence(token_span):
                return round(float(tag_confidence[row, token_span[0]:token_span[1] + 1].mean()), 4)

            emotion_spans = [(valid[s], valid[e]) for kind, s, e in spans if kind == "EMO"]
            cause_spans = [
                (valid[s], valid[e])
                for kind, s, e in spans
                if kind == "CAU"
                and len(text[slice(*char_span((valid[s], valid[e])))].split()) >= MIN_CAUSE_WORDS
            ]

            emotions = []
            if emotion_spans:
                emotion_vectors = model.span_vectors(hidden[row], emotion_spans).float()
                emotion_probs = torch.softmax(model.emotion_head(emotion_vectors), dim=-1)
                status_probs = torch.softmax(model.status_head(emotion_vectors), dim=-1)
                pair_probs = None
                if cause_spans:
                    cause_vectors = model.span_vectors(hidden[row], cause_spans).float()
                    pair_probs = torch.sigmoid(model.pair_logits(
                        emotion_vectors, cause_vectors, emotion_spans, cause_spans
                    ).float())

                for position, token_span in enumerate(emotion_spans):
                    start, end = char_span(token_span)
                    emotion_id = int(emotion_probs[position].argmax())
                    status_id = int(status_probs[position].argmax())
                    linked = []
                    if CAUSE_STATUSES[status_id] == "stated":
                        if pair_probs is None:
                            # No cause span was found, so fall back to implicit or none.
                            status_id = 1 + int(status_probs[position, 1:].argmax())
                        else:
                            ranked = pair_probs[position].argsort(descending=True).tolist()
                            linked = ranked[:MAX_CAUSES_PER_EMOTION]
                    causes = []
                    for index in linked:
                        cause_start, cause_end = char_span(cause_spans[index])
                        causes.append({
                            "cause_clause": text[cause_start:cause_end],
                            "start": cause_start,
                            "end": cause_end,
                            "span_confidence": span_confidence(cause_spans[index]),
                            "confidence": round(float(pair_probs[position, index]), 4),
                        })
                    emotions.append({
                        "emotion_clause": text[start:end],
                        "start": start,
                        "end": end,
                        "span_confidence": span_confidence(token_span),
                        "emotion": EMOTIONS[emotion_id],
                        "emotion_confidence": round(float(emotion_probs[position, emotion_id]), 4),
                        "cause_status": CAUSE_STATUSES[status_id],
                        "causes": causes,
                    })
            results.append({"emotions": emotions})
    return results


def save_model(model, tokenizer, directory, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    torch.save(model.state_dict(), directory / WEIGHTS_FILE)
    labels = {"tags": TAGS, "emotions": EMOTIONS, "cause_statuses": CAUSE_STATUSES}
    (directory / LABELS_FILE).write_text(
        json.dumps({**labels, "metadata": metadata or {}}, indent=2), encoding="utf-8"
    )


def load_model(directory=MODEL_DIR, device="cpu"):
    directory = Path(directory)
    if not (directory / WEIGHTS_FILE).is_file():
        raise FileNotFoundError(f"Trained ECPE model not found in {directory}")
    labels = json.loads((directory / LABELS_FILE).read_text(encoding="utf-8"))
    if (labels["tags"], labels["emotions"], labels["cause_statuses"]) != (
        TAGS, EMOTIONS, CAUSE_STATUSES
    ):
        raise ValueError("Saved model labels do not match the labels in ecpe_model.py.")

    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    model = ECPEModel(AutoModel.from_config(config))
    model.load_state_dict(
        torch.load(directory / WEIGHTS_FILE, map_location=device, weights_only=True)
    )
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("The saved model requires a fast tokenizer for character offsets.")
    model.to(device)
    model.eval()
    return model, tokenizer
