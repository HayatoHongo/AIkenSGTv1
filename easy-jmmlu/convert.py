"""Convert the small JCommonsenseQA source file to the MMLU CSV format."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent
LABELS = "ABCD"

# Read each original question. A label of 4 is excluded. For every other
# question, keep choices 0–3 and write the answer as A–D.
questions = {"dev": [], "test": []}
with (ROOT / "source.jsonl").open(encoding="utf-8") as source_file:
    for line in source_file:
        row = json.loads(line)
        if row["label"] == 4:
            continue
        choices = row["choices"][:4]
        answer = LABELS[row["label"]]
        questions[row["split"]].append([row["question"], *choices, answer])

for split in ("dev", "test"):
    output = ROOT / split / f"jcommonsenseqa_{split}.csv"
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerows(questions[split])
    print(f"{output}: {len(questions[split])} questions")
