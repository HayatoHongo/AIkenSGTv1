#!/usr/bin/env python3
"""Drop questions whose explanations mention an option number."""

import argparse
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

def main():
    parser = argparse.ArgumentParser(description="選択肢番号に言及する解説の問題を除外します")
    parser.add_argument("input", nargs="?", type=Path, default=INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=OUTPUT)
    args = parser.parse_args()

    kept = 0
    removed = 0
    with args.input.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            record = json.loads(line)
            explanation = record["overall_explanation"]
            if any(phrase in explanation for phrase in NUMBERED_PHRASES):
                removed += 1
            else:
                kept += 1
                record["question_id"] = f"gkgen_{kept:05d}"
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"完了: 残した行={kept}, 除外した行={removed}, 出力={args.output}")


if __name__ == "__main__":
    main()
