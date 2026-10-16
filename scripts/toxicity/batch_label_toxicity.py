#!/usr/bin/env python3
"""Optional OpenAI Batch API runner for the toxicity re-annotation step.

Emits batch-input JSONL files matching the exact production format used for
`textdetox/multilingual_toxicity_dataset` (en, ar) re-annotation with
`gpt-4.1-mini` (see datasets/toxicity/DATASET.md for the verbatim prompt and
schema). The production snapshot of these batch inputs/outputs is
`openai_data.zip` in the study workspace; the runner here lets you re-run the
annotation yourself (e.g. to relabel after downloading the texts) and produces
the `{language}_labeled.csv` inputs consumed by
`datasets/toxicity/prepare_toxicity_splits.py`.

The actual API submission is left to the user (stub at the end), because batch
costs and account handling are site-specific: the script writes the JSONL batch
input, and (given a batch, i.e. a file of per-custom_id model outputs) assembles
the labeled CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEVELOPER_PROMPT = """You are labeling short social media texts.

Task:
For the given text, output binary labels for whether the text explicitly mentions:
1. gender
2. ethnicity
3. religion
4. nationality
5. location

Rules:
- Use 1 only for explicit mention, not implication.
- Base the decision only on the text provided.
- Do not infer author identity or target identity unless explicitly stated.
- Nationality and ethnicity are separate.
- Location means a geographic place name or region explicitly mentioned, mentioning a people or community does not count as the location, it must be an explicit mention of a place.
- Justifications must be extremely brief and cite the trigger word or say no explicit mention.
- Return only schema-compliant JSON.
"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "mentions_gender": {"type": "integer", "enum": [0, 1]},
        "mentions_ethnicity": {"type": "integer", "enum": [0, 1]},
        "mentions_religion": {"type": "integer", "enum": [0, 1]},
        "mentions_nationality": {"type": "integer", "enum": [0, 1]},
        "mentions_location": {"type": "integer", "enum": [0, 1]},
        "justification_gender": {"type": "string"},
        "justification_ethnicity": {"type": "string"},
        "justification_religion": {"type": "string"},
        "justification_nationality": {"type": "string"},
        "justification_location": {"type": "string"},
    },
    "required": [
        "mentions_gender",
        "mentions_ethnicity",
        "mentions_religion",
        "mentions_nationality",
        "mentions_location",
        "justification_gender",
        "justification_ethnicity",
        "justification_religion",
        "justification_nationality",
        "justification_location",
    ],
    "additionalProperties": False,
}

FIELDS = ["toxic", "mentions_gender", "mentions_ethnicity", "mentions_religion", "mentions_nationality", "mentions_location"]


def make_request(custom_id: str, text: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": "gpt-4.1-mini",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": DEVELOPER_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"Text: {text}"}],
                },
            ],
            "text": {"format": {"type": "json_schema", "name": "identity_location_labels", "schema": LABEL_SCHEMA, "strict": True}},
        },
    }


def cmd_build_batch(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, newline="", encoding="utf-8") as source, out.open(
        "w", encoding="utf-8"
    ) as sink:
        for index, row in enumerate(csv.DictReader(source)):
            text = row["text"].strip()
            request = make_request(f"{args.language}_{index:06d}", text)
            sink.write(json.dumps(request, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({index + 1 if index is not None else 0} requests)")


def cmd_assemble(args: argparse.Namespace) -> None:
    """Combine a batch result JSONL (one custom_id + model output per line) with
    the source texts into `{language}_labeled.csv`.

    The exact batch-output container depends on how the batch was retrieved
    (the OpenAI Batch API returns a JSONL file of per-request response bodies).
    This command expects one JSON object per line with at least
    `{"custom_id": ..., "output": {...schema fields...}}` or the raw OpenAI
    response envelope with the parsed content under
    `response.body.output_text` (stringified JSON) — adjust as needed for your
    retrieval format; see openai_data.zip in the study workspace for the exact
    production layout.
    """
    rows = []
    with open(args.results, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            output = record.get("output")
            if output is None:
                body = record.get("response", {}).get("body", {})
                output = json.loads(body.get("output_text", "{}"))
            rows.append({"custom_id": record["custom_id"], **output})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, newline="", encoding="utf-8") as source, out.open(
        "w", newline="", encoding="utf-8"
    ) as sink:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(sink, fieldnames=reader.fieldnames + FIELDS)
        writer.writeheader()
        for row, labeled in zip(reader, rows):
            row.update({field: labeled[field] for field in FIELDS})
            writer.writerow(row)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-batch", help="write batch-input JSONL from a texts CSV")
    build.add_argument("--input", required=True, help="CSV with a `text` column")
    build.add_argument("--language", required=True, help="language code (en/ar), used in custom_id")
    build.add_argument("--output", required=True, help="output JSONL path")
    build.set_defaults(func=cmd_build_batch)

    assemble = sub.add_parser("assemble-labeled", help="combine batch results with source texts into a labeled CSV")
    assemble.add_argument("--input", required=True, help="the same texts CSV used for build-batch")
    assemble.add_argument("--results", required=True, help="batch results JSONL")
    assemble.add_argument("--output", required=True, help="output {language}_labeled.csv path")
    assemble.set_defaults(func=cmd_assemble)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
