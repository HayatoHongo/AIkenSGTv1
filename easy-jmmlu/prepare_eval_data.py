"""Prepare MMLU-style CSV folders from the JCommonsenseQA JSONL files."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).parent
OUTPUT = ROOT / "eval_data"
LABELS = "ABCD"


def row_from_source(row):
    return [row["question"], *row["choices"], LABELS[row["label"]]]


def rows_from_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def write_csv(split, rows):
    folder = OUTPUT / split
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"jcommonsenseqa_{split}.csv"
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
    print(f"{path}: {len(rows)} questions")


train = rows_from_jsonl(ROOT / "train.jsonl")
write_csv("train", [row_from_source(row) for row in train])
write_csv("dev", [row_from_source(row) for row in train[:5]])

# The validation prompt-response JSONL is the held-out evaluation set.
validation_rows = []
for row in rows_from_jsonl(ROOT / "validation_prompt_response.jsonl"):
    lines = row["prompt"].splitlines()
    question = next(line.removeprefix("問題: ") for line in lines if line.startswith("問題: "))
    choices = [line[3:] for line in lines if len(line) > 3 and line[1:3] == ". "]
    answer = row["response"].strip()
    if len(choices) != 4 or answer not in LABELS:
        raise ValueError(f"Invalid four-choice row: {row}")
    validation_rows.append([question, *choices, answer])

write_csv("test", validation_rows)
