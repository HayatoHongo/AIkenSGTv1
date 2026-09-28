#!/usr/bin/env python3
"""Convert G検定 JSONL records to prompt/response JSONL."""

import argparse
import json
from pathlib import Path


LABELS = "ABCD"


def convert(input_path: Path, output_path: Path) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_no, line in enumerate(source, 1):
            record = json.loads(line)
            try:
                correct = int(record["correct_answers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: invalid correct_answers") from exc
            if correct not in range(1, 5):
                raise ValueError(
                    f"line {line_no}: correct_answers must be 1-4, got {correct}"
                )

            stem = record["question_text"].strip()
            options = [record[f"option_{i}"].strip() for i in range(1, 5)]
            prompt = stem + "\n" + "\n".join(
                f"{LABELS[i]}. {option}" for i, option in enumerate(options)
            )
            response = "\\boxed{" + options[correct - 1] + "}"
            output.write(
                json.dumps(
                    {"prompt": prompt, "response": response}, ensure_ascii=False
                )
                + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G検定 JSONLをprompt/response形式に変換します"
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("gkentei_500k_shuffled_records.jsonl"),
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("gkentei_500k_prompt_response.jsonl"),
    )
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(f"変換完了: {count}件, 出力={args.output}")


if __name__ == "__main__":
    main()
