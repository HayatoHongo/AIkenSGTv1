#!/usr/bin/env python3
"""Review G検定 questions with gpt-oss-120b via OpenRouter."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


MODEL = "openai/gpt-oss-120b"
PROVIDER = "deepinfra"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_WORKERS = 20

REVIEW_INSTRUCTIONS = """G検定の四択問題を品質レビューしてください。
次のすべてを満たす場合だけPASSにしてください。それ以外はREJECTです。
- 問題文に対して正解が一つに定まり、correct_answersがその選択肢を指している。
- 問題文、選択肢、正解、解説に明白な事実誤認や矛盾がない。
- 問題として成立し、選択肢が問題文に答えている。
- 解説が正解の根拠を正しく説明している。
判断に迷う場合はREJECTにしてください。
次のJSONだけを返してください: {"verdict":"PASSまたはREJECT","reason":"短い日本語の理由"}
"""


def review_question(api_key, question):
    content = {key: value for key, value in question.items() if key in {
        "question_id", "keyword", "genre", "difficulty", "question_text",
        "option_1", "option_2", "option_3", "option_4", "correct_answers",
        "overall_explanation",
    }}
    body = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": REVIEW_INSTRUCTIONS + "\n" + json.dumps(content, ensure_ascii=False),
        }],
        "temperature": 0,
        "max_completion_tokens": 1200,
        "reasoning": {"effort": "low", "exclude": True},
        "response_format": {"type": "json_object"},
        "provider": {"order": [PROVIDER], "allow_fallbacks": False},
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))
    return json.loads(result["choices"][0]["message"]["content"])


def main():
    parser = argparse.ArgumentParser(description="G検定問題をPASS/REJECTでレビューします")
    parser.add_argument("input", type=Path, help="レビューするJSONLファイル")
    parser.add_argument("--output", type=Path, help="結果のJSONLファイル")
    args = parser.parse_args()
    output_path = args.output or args.input.with_name(f"{args.input.stem}_reviewed.jsonl")

    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]
    with args.input.open(encoding="utf-8") as source:
        questions = [json.loads(line) for line in source]

    with output_path.open("x", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            reviews = {
                executor.submit(review_question, api_key, question): question
                for question in questions
            }
            for future in as_completed(reviews):
                question = reviews[future]
                review = future.result()
                question["review_verdict"] = review["verdict"]
                question["review_reason"] = review["reason"]
                output.write(json.dumps(question, ensure_ascii=False) + "\n")
                print(f"{question['question_id']}: {review['verdict']} — {review['reason']}")

    print(f"レビュー完了: {output_path}")


if __name__ == "__main__":
    main()
