#!/usr/bin/env python3
"""Create a shuffled copy of the G検定 questions."""

import json
import random
from pathlib import Path


INPUT = Path("gkentei_questions_passed.jsonl")
OUTPUT = Path("gkentei_questions_passed_shuffled.jsonl")
randomizer = random.Random(20260925)

with INPUT.open(encoding="utf-8") as source, OUTPUT.open("w", encoding="utf-8") as output:
    for line in source:
        record = json.loads(line)
        correct = int(record["correct_answers"])
        choices = [
            {"text": record[f"option_{i}"], "correct": i == correct}
            for i in range(1, 5)
        ]
        randomizer.shuffle(choices)
        for i, choice in enumerate(choices, start=1):
            record[f"option_{i}"] = choice["text"]
            if choice["correct"]:
                record["correct_answers"] = str(i)
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
