#!/usr/bin/env python3
"""Drop questions whose explanations mention an option number."""

import json
from pathlib import Path


INPUT = Path("gkentei_questions_passed.jsonl")
OUTPUT = Path("gkentei_questions_without_numbered_explanations.jsonl")
NUMBERED_PHRASES = (
    "選択肢1", "選択肢2", "選択肢3", "選択肢4",
    "選択肢 1", "選択肢 2", "選択肢 3", "選択肢 4",
    "1番", "2番", "3番", "4番",
    "1番目", "2番目", "3番目", "4番目",
    "①", "②", "③", "④",
)

with INPUT.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as output:
    for line in source:
        record = json.loads(line)
        explanation = record["overall_explanation"]
        if not any(phrase in explanation for phrase in NUMBERED_PHRASES):
            output.write(line)
