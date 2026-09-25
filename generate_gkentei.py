#!/usr/bin/env python3
"""Generate G検定 questions from Gkentei_keyword.csv via OpenRouter."""

import argparse
import csv
import json
import os
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv


MODEL = "openai/gpt-oss-120b"
PROVIDER = "deepinfra"
REASONING_EFFORT = "medium"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

GENRE_WEIGHTS = {
    "Appropriate Choice": 0.7,
    "Combination": 0.1,
    "Fill Blank": 0.1,
    "Inappropriate Choice": 0.1,
}

GENRE_INSTRUCTIONS = {
    "Appropriate Choice": (
        "技術の利用場面や解決したい課題を示し、それに合う手法・概念・説明を選ばせる。"
        "単純な用語定義だけでなく、状況判断や簡単な計算も扱い、正解を一つにする。"
    ),
    "Combination": (
        "問題文に（あ）（い）（う）など複数の空欄または対応項目を設ける。空欄は2~4個程度にする。"
        "各選択肢には空欄に入る語句の組み合わせを示し、全体として正しい組み合わせを一つ選ばせる。"
    ),
    "Fill Blank": (
        "説明や利用場面を含む一文の中に空欄（_____）を一つ設ける。"
        "文脈に最もよく当てはまる用語や語句を四択から一つ選ばせる。"
    ),
    "Inappropriate Choice": (
        "テーマに関する説明を四つ提示し、そのうち誤りを含む選択肢を一つだけ選ばせる。"
        "問題文では『最も不適切なものを1つ選べ』と明記し、曖昧さや複数の誤答を避ける。"
    ),
}


DIFFICULTY_WEIGHTS = {"easy": 0.25, "medium": 0.5, "hard": 0.25}

DIFFICULTY_INSTRUCTIONS = {
    "easy": "基本用語や基本原理の理解を問う。追加の推論や複数段階の判断を必要としない。",
    "medium": "学んだ概念を場面に当てはめるか、近い概念を区別させる。判断は一段階を基本とする。",
    "hard": "複数の条件や概念を組み合わせ、段階を追った判断を必要とする。曖昧さや細かな言い回しで難しくしない。",
}


def read_keywords(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [
            row for row in csv.DictReader(file)
            if row["用語"] and row["技術的な独自説明"]
        ]


def make_prompt(keyword, genre, difficulty):
    return f"""G検定対策の新作四択問題を1問作ってください。

キーワード: {keyword['用語']}
技術的説明: {keyword['技術的な独自説明']}
関連用語: {keyword['関連用語']}
問題形式: {GENRE_INSTRUCTIONS[genre]}
難易度: {DIFFICULTY_INSTRUCTIONS[difficulty]}

説明にない事実は追加せず、自然で簡潔な日本語で作ってください。
既存設問を写さず、正解が一つだけの問題にしてください。
関連用語から関係するものを比較対象や選択肢に使い、問い方に変化を付けてください。
overall_explanationは正解の概念・内容を根拠とともに簡潔に説明してください。選択肢の番号や位置を一切書かないでください。
「選択肢1」「選択肢4」「①」「正解は1」「正しい組み合わせは①」のように、番号や記号で正解を示す表現は禁止です。番号の代わりに、正解となる概念名や仕組みを主語にして説明してください。
correct_answerには、正解の選択肢番号を1〜4の整数で入れてください。
次のJSONだけを返してください。
{{"question":"問題文","options":["選択肢1","選択肢2","選択肢3","選択肢4"],"correct_answer":正解の選択肢の番号(1~4),"overall_explanation":"正解の根拠"}}"""


def generate_question(api_key, prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_completion_tokens": 4096,
        "reasoning": {"effort": REASONING_EFFORT, "exclude": True},
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

    message = result["choices"][0]["message"]
    return json.loads(message["content"]), result.get("usage", {})


def make_record(number, keyword, genre, difficulty, question, usage):
    correct_answer = question["correct_answer"]
    if correct_answer not in (1, 2, 3, 4):
        raise ValueError("correct_answer must be an integer from 1 to 4")

    record = {
        "question_id": f"gkgen_{number:04d}",
        "term_id": keyword["用語ID"],
        "keyword": keyword["用語"],
        "related_terms": keyword["関連用語"],
        "category": keyword["大カテゴリー"],
        "section": keyword["中カテゴリー"],
        "keyword_definition": keyword["技術的な独自説明"],
        "genre": genre,
        "difficulty": difficulty,
        "question_text": question["question"],
        "correct_answers": str(correct_answer),
        "overall_explanation": question["overall_explanation"],
        "model": MODEL,
        "provider": PROVIDER,
        "reasoning_tokens": (
            usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens"),
    }
    record.update({f"option_{i}": value for i, value in enumerate(question["options"], 1)})
    return record


def generate_question_record(api_key, task):
    number, keyword, genre, difficulty = task
    prompt = make_prompt(keyword, genre, difficulty)
    question, usage = generate_question(api_key, prompt)
    return make_record(number, keyword, genre, difficulty, question, usage)


def write_questions(output, api_key, assignments, workers):
    tasks = [
        (number, keyword, genre, difficulty)
        for number, (keyword, genre, difficulty) in enumerate(assignments, start=1)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(generate_question_record, api_key, task)
            for task in tasks
        ]

        for task, future in zip(tasks, futures):
            number, keyword, genre, difficulty = task
            record = future.result()
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{number}: {keyword['用語']} / {genre} / {difficulty}")


def main():
    parser = argparse.ArgumentParser(description="CSVからG検定問題を生成します")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("gkentei_generated.jsonl"))
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count は1以上を指定してください")
    if args.workers < 1:
        parser.error("--workers は1以上を指定してください")

    keywords = read_keywords(Path("Gkentei_keyword.csv"))
    if not keywords:
        parser.error("Gkentei_keyword.csv に有効なキーワードがありません")

    # Use each keyword once before randomly reusing any of them.
    selected = random.sample(keywords, min(args.count, len(keywords)))
    selected += random.choices(keywords, k=args.count - len(selected))
    # 重みは抽選確率。小さい件数では比率どおりにならない。
    genres = random.choices(
        list(GENRE_WEIGHTS), weights=list(GENRE_WEIGHTS.values()), k=args.count
    )
    difficulties = random.choices(
        list(DIFFICULTY_WEIGHTS), weights=list(DIFFICULTY_WEIGHTS.values()), k=args.count
    )
    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignments = zip(selected, genres, difficulties)
    with args.output.open("x", encoding="utf-8") as output:
        write_questions(output, api_key, assignments, args.workers)

    print(f"生成完了: {args.output}")


if __name__ == "__main__":
    main()
