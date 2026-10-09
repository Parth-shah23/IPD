import re
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizerFast, RobertaConfig, RobertaModel
from transformers.convert_slow_tokenizer import convert_slow_tokenizer


st.set_page_config(page_title="Journal Model Demo", page_icon="📓", layout="wide")

ROOT = Path(__file__).resolve().parent
ECPE_MODEL_PATH = ROOT / "ecpe_model.pkl"
CRISIS_MODEL_PATH = ROOT / "crisis_model.pkl"
NER_MODEL_PATH = ROOT / "ner_model_v3.pkl"
ASPECT_MODEL_PATH = ROOT / "aspect_model_rich_logreg.pkl"
WORD_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
MAX_LENGTH = 512


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_ecpe_model():
    payload = joblib.load(ECPE_MODEL_PATH)
    if payload.get("format") != "ecpe-span-v1":
        raise ValueError(f"Unsupported ECPE model format in {ECPE_MODEL_PATH.name}.")

    config = RobertaConfig.from_dict(payload["encoder_config"])
    encoder = RobertaModel(config)
    state_dict = payload["state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)

    hidden_size = config.hidden_size
    heads = {
        "tagger": nn.Linear(hidden_size, len(payload["labels"]["tags"])),
        "emotion_head": nn.Linear(hidden_size, len(payload["labels"]["emotions"])),
        "status_head": nn.Linear(hidden_size, len(payload["labels"]["cause_statuses"])),
    }
    for name, head in heads.items():
        head.load_state_dict({
            "weight": state_dict[f"{name}.weight"],
            "bias": state_dict[f"{name}.bias"],
        })

    device = get_device()
    encoder.to(device).eval()
    for head in heads.values():
        head.to(device).eval()

    return encoder, heads, payload["tokenizer"], payload["labels"], device


@st.cache_resource
def load_ner_model():
    payload = joblib.load(NER_MODEL_PATH)
    tokenizer = payload["tokenizer"]
    if not tokenizer.is_fast:
        tokenizer = BertTokenizerFast(
            tokenizer_object=convert_slow_tokenizer(tokenizer)
        )

    model = payload["model"]
    device = get_device()
    model.to(device).eval()
    return tokenizer, model, device


@st.cache_resource
def load_crisis_model():
    model = joblib.load(CRISIS_MODEL_PATH)
    device = get_device()
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_aspect_model():
    return joblib.load(ASPECT_MODEL_PATH)


def _encode_words(text, tokenizer):
    words = list(WORD_PATTERN.finditer(text))
    if not words:
        return [], None

    encoded = tokenizer(
        [match.group() for match in words],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return words, encoded


def predict_ecpe(text, encoder, heads, tokenizer, labels, device):
    words, encoded = _encode_words(text, tokenizer)
    if not words:
        return {"emotions": [], "causes": [], "truncated": False}
    original_word_count = len(words)

    model_inputs = {
        key: encoded[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in encoded
    }
    with torch.inference_mode():
        hidden = encoder(**model_inputs).last_hidden_state[0]
        tag_logits = heads["tagger"](hidden)

    word_hidden = {}
    word_logits = {}
    for token_index, word_id in enumerate(encoded.word_ids(batch_index=0)):
        if word_id is None:
            continue
        word_hidden.setdefault(word_id, []).append(hidden[token_index])
        word_logits.setdefault(word_id, []).append(tag_logits[token_index])

    word_vectors = [
        torch.stack(word_hidden[index]).mean(dim=0)
        for index in range(max(word_hidden) + 1)
    ]
    word_labels = [
        labels["tags"][int(torch.argmax(torch.stack(word_logits[index]).mean(dim=0)))]
        for index in range(max(word_logits) + 1)
    ]
    words = words[:len(word_labels)]

    spans = {"EMO": [], "CAU": []}
    active_start = None
    active_type = None

    def finish_span(end_word):
        if active_start is None or active_type is None:
            return
        start_offset = words[active_start].start()
        end_offset = words[end_word - 1].end()
        span_vector = torch.stack(word_vectors[active_start:end_word]).mean(dim=0)
        with torch.inference_mode():
            if active_type == "EMO":
                probabilities = torch.softmax(
                    heads["emotion_head"](span_vector), dim=-1
                )
                class_names = labels["emotions"]
                field_name = "emotion"
            else:
                probabilities = torch.softmax(
                    heads["status_head"](span_vector), dim=-1
                )
                class_names = labels["cause_statuses"]
                field_name = "status"
        class_id = int(torch.argmax(probabilities))
        spans[active_type].append({
            "clause": text[start_offset:end_offset],
            field_name: class_names[class_id],
            "confidence": float(probabilities[class_id]),
        })

    for word_index, label in enumerate(word_labels + ["O"]):
        if label.startswith("B-") or label == "O":
            finish_span(word_index)
            if label.startswith("B-"):
                active_start = word_index
                active_type = label[2:]
            else:
                active_start = None
                active_type = None
        elif label.startswith("I-"):
            next_type = label[2:]
            if active_type != next_type:
                finish_span(word_index)
                active_start = word_index
                active_type = next_type

    return {
        "emotions": spans["EMO"],
        "causes": spans["CAU"],
        "truncated": len(word_labels) < original_word_count,
    }


def predict_ner(text, tokenizer, model, device):
    words = list(WORD_PATTERN.finditer(text))
    if not words:
        return []

    encoded = tokenizer(
        [match.group() for match in words],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=64,
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: encoded[key].to(device)
        for key in ("input_ids", "attention_mask", "token_type_ids")
        if key in encoded
    }
    with torch.inference_mode():
        logits = model(**model_inputs).logits.cpu()

    logits_by_word = {}
    for batch_index in range(logits.shape[0]):
        seen_word_ids = set()
        for token_index, word_id in enumerate(encoded.word_ids(batch_index=batch_index)):
            if word_id is None or word_id in seen_word_ids:
                continue
            seen_word_ids.add(word_id)
            logits_by_word.setdefault(word_id, []).append(logits[batch_index, token_index])

    id2label = model.config.id2label
    entities = []
    start_word = None
    entity_type = None
    labels_by_word = ["O"] * len(words)
    confidence_by_word = [0.0] * len(words)
    for word_id, word_logits in logits_by_word.items():
        probabilities = torch.softmax(torch.stack(word_logits).mean(dim=0), dim=-1)
        label_id = int(torch.argmax(probabilities))
        labels_by_word[word_id] = id2label[label_id]
        confidence_by_word[word_id] = float(probabilities[label_id])

    def finish_entity(end_word):
        if start_word is None or entity_type is None:
            return
        start_offset = words[start_word].start()
        end_offset = words[end_word - 1].end()
        confidence = np.mean(confidence_by_word[start_word:end_word])
        entities.append({
            "text": text[start_offset:end_offset],
            "label": entity_type,
            "start": start_offset,
            "end": end_offset,
            "confidence": round(float(confidence), 4),
        })

    for word_index, label in enumerate(labels_by_word + ["O"]):
        if label.startswith("B-") or label == "O":
            finish_entity(word_index)
            if label.startswith("B-"):
                start_word = word_index
                entity_type = label[2:]
            else:
                start_word = None
                entity_type = None
        elif label.startswith("I-"):
            next_type = label[2:]
            if entity_type != next_type:
                finish_entity(word_index)
                start_word = word_index
                entity_type = next_type

    return entities


def predict_crisis(clause, model, tokenizer, device):
    encoded = tokenizer(
        clause,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in ("input_ids", "attention_mask", "token_type_ids")
    }
    with torch.inference_mode():
        probabilities = torch.softmax(model(**model_inputs).logits[0], dim=-1)
    class_id = int(torch.argmax(probabilities))
    return {
        "label": model.config.id2label[class_id],
        "confidence": float(probabilities[class_id]),
        "probabilities": {
            model.config.id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


def predict_aspect(model, emotion_clause, emotion, cause_clause, cause, ner):
    feature_text = re.sub(
        r"\s+",
        " ",
        " ".join((emotion_clause, emotion, cause_clause, cause, ner)),
    ).strip()
    prediction = str(model.predict([feature_text])[0])
    probabilities = model.predict_proba([feature_text])[0]
    ranked = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )
    return prediction, [(str(label), float(score)) for label, score in ranked[:5]]


def _render_records(title, records):
    st.markdown(f"**{title}**")
    if records:
        st.dataframe(records, width="stretch", hide_index=True)
    else:
        st.caption("No spans detected.")


st.title("Journal Analysis Models")
st.write(
    "Run the journal through ECPE and NER, analyze a separate crisis clause, "
    "then use the extracted information as inputs to aspect classification."
)

with st.form("journal_analysis"):
    journal_entry = st.text_area(
        "Journal entry",
        height=180,
        placeholder="Write or paste a journal entry for emotion/cause and entity detection.",
    )
    crisis_clause = st.text_area(
        "Crisis-detection clause",
        height=90,
        placeholder="Enter the separate clause to classify with the crisis model.",
    )
    analyze = st.form_submit_button("Run journal analysis", type="primary")

if analyze:
    if not journal_entry.strip():
        st.warning("Enter a journal entry before running ECPE and NER.")
    else:
        with st.spinner("Loading models and analyzing the text..."):
            ecpe_encoder, ecpe_heads, ecpe_tokenizer, ecpe_labels, device = (
                load_ecpe_model()
            )
            ecpe_result = predict_ecpe(
                journal_entry,
                ecpe_encoder,
                ecpe_heads,
                ecpe_tokenizer,
                ecpe_labels,
                device,
            )

            ner_tokenizer, ner_model, ner_device = load_ner_model()
            ner_entities = predict_ner(
                journal_entry, ner_tokenizer, ner_model, ner_device
            )

            crisis_result = None
            if crisis_clause.strip():
                crisis_model, crisis_device = load_crisis_model()
                crisis_result = predict_crisis(
                    crisis_clause.strip(),
                    crisis_model,
                    ecpe_tokenizer,
                    crisis_device,
                )

        st.session_state["journal_analysis_result"] = {
            "journal_entry": journal_entry,
            "crisis_clause": crisis_clause,
            "ecpe": ecpe_result,
            "entities": ner_entities,
            "crisis": crisis_result,
        }
        st.session_state.pop("aspect_result", None)

analysis = st.session_state.get("journal_analysis_result")
if analysis:
    st.divider()
    st.subheader("1. Emotion-Cause Extraction (ECPE)")
    if analysis["ecpe"]["truncated"]:
        st.info("ECPE input exceeded the model's 512-token limit; the remaining text was not analyzed.")
    left, right = st.columns(2)
    with left:
        _render_records("Emotion clauses", analysis["ecpe"]["emotions"])
    with right:
        _render_records("Cause clauses", analysis["ecpe"]["causes"])

    with st.expander("2. Named entities (NER)", expanded=True):
        _render_records("Entities in the journal entry", analysis["entities"])

    with st.expander("3. Crisis detection", expanded=True):
        if analysis["crisis"]:
            crisis = analysis["crisis"]
            st.write(f"Clause: {analysis['crisis_clause']}")
            st.metric(
                "Predicted class",
                crisis["label"],
                f"{crisis['confidence']:.1%} confidence",
            )
            st.json(crisis["probabilities"])
            st.caption(
                "This checkpoint stores generic LABEL_0/LABEL_1 class names, "
                "so the app shows the model's class labels without guessing "
                "which one means 'crisis'."
            )
        else:
            st.caption("No crisis clause was entered.")

    st.divider()
    st.subheader("4. Aspect classification")
    first_emotion = (
        analysis["ecpe"]["emotions"][0]
        if analysis["ecpe"]["emotions"]
        else {}
    )
    first_cause = (
        analysis["ecpe"]["causes"][0]
        if analysis["ecpe"]["causes"]
        else {}
    )
    entity_feature = "; ".join(
        f"{entity['text']}|{entity['label']}"
        for entity in analysis["entities"]
    ) or "none"
    if first_cause.get("status"):
        st.caption(
            f"ECPE cause status for the first detected cause: "
            f"{first_cause['status']} ({first_cause['confidence']:.1%})"
        )

    with st.form("aspect_prediction"):
        aspect_emotion_clause = st.text_input(
            "Emotion clause",
            value=first_emotion.get("clause", ""),
        )
        aspect_emotion = st.text_input(
            "Emotion",
            value=first_emotion.get("emotion", ""),
        )
        aspect_cause_clause = st.text_input(
            "Cause clause",
            value=first_cause.get("clause", ""),
        )
        aspect_cause = st.text_input(
            "Cause label (aspect feature)",
            placeholder="For example: missed train or good internal marks",
            help="This is the cause-category text expected by the aspect model; "
            "it is separate from ECPE's stated/implicit/none cause status.",
        )
        aspect_ner = st.text_input(
            "NER feature",
            value=entity_feature,
            help="Entities are formatted as text|label, separated by semicolons.",
        )
        predict_aspect_button = st.form_submit_button("Predict aspect")

    if predict_aspect_button:
        if not aspect_emotion_clause.strip() and not aspect_cause_clause.strip():
            st.warning("Enter an emotion clause or a cause clause for aspect prediction.")
        else:
            aspect_model = load_aspect_model()
            st.session_state["aspect_result"] = predict_aspect(
                aspect_model,
                aspect_emotion_clause,
                aspect_emotion,
                aspect_cause_clause,
                aspect_cause,
                aspect_ner,
            )

    if st.session_state.get("aspect_result"):
        prediction, alternatives = st.session_state["aspect_result"]
        st.metric("Predicted aspect", prediction)
        st.dataframe(
            [
                {"Aspect": label, "Model score": score}
                for label, score in alternatives
            ],
            width="stretch",
            hide_index=True,
        )
