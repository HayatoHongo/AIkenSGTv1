#!/usr/bin/env python3
"""Remove generation error rows from a G検定 JSONL file."""

import argparse
import json
from pathlib import Path


def remove_errors(input_path, output_path):
    if input_path.resolve() == output_path.resolve():
        raise ValueError("入力ファイルと出力ファイルには別のパスを指定してください")

    kept = 0
    removed = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            record = json.loads(line)
            if record.get("generation_status") == "error":
                removed += 1
                continue

            kept += 1
            record["question_id"] = f"gkgen_{kept:05d}"
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="JSONLから生成エラー行を除外します")
    parser.add_argument("input", nargs="?", type=Path, default=Path("gkentei_concat.jsonl"))
    parser.add_argument(
        "output", nargs="?", type=Path, default=Path("gkentei_concat_no_errors.jsonl")
    )
    args = parser.parse_args()

    kept, removed = remove_errors(args.input, args.output)
    print(f"完了: 残した行={kept}, 除外したエラー行={removed}, 出力={args.output}")


if __name__ == "__main__":
    main()
