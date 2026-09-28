#!/usr/bin/env python3
"""Generate G検定 questions from Gkentei_keyword.csv via OpenRouter."""

import argparse
import csv
import json
import os
import random
import ssl
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import certifi
from dotenv import load_dotenv


MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "medium"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

GENRE_WEIGHTS = {
    "Appropriate Choice": 0.7,
    "Combination A": 0.05,
    "Combination B/C": 0.05,
    "Fill Blank": 0.1,
    "Inappropriate Choice": 0.1,
}

GENRE_INSTRUCTIONS = {
    "Appropriate Choice": (
        "四択から、設問に最も適切な答えを一つ選ばせる。"
    ),
    "Combination A": (
        "各生成で、題材に合わせて、項目数を2個にするか3個にするか4個にするかを選ぶ。"
        "題材に合う型を一つ選ぶ。対応項目は2〜4個とし、記号（あ）（い）...を順に使う。半角括弧ではなく、全角括弧（）を必ず使うこと。"
        "手順・状況: 文章の中に2〜4個の記号付き空欄を埋め込む。各空欄に当てはまる役割・手法・段階を問う。"
        "...（あ)....（い）....のように、問題文の途中のどこかを記号（あ）（い）...でマスクした穴埋めのような形式にする。"
        "禁止事項: アンダーバーの使用を固く禁止する。_____や、（あ）_____ のように、絶対にアンダーバーを使ったり混ぜたりしてはいけない。必ず記号（あ）（い）...「だけ」を使うこと。厳重に注意する。"
        "選択肢に関しては、（あ）選択肢、（い）選択肢、....のような形式にする。"
        "末尾で記号順の組合せを選ばせる。選択肢の順序を揃え、正解は一つにする。四択問題"
         "指示は「〇〇の組み合わせとして、最も適切なものを1つ選べ。」の形式を基本とする。〇〇はそのまま書かずに、適切な表現に置き換える。"
    ),
    "Combination B/C": (
        "各生成で、題材に合わせて、配置形式B・Cから一つ選び、項目数を2個にするか3個にするか4個にするかを選ぶ。"
        "題材に合う型を一つ選ぶ。対応項目は2〜4個とし、記号（あ）（い）...を順に使う。半角括弧ではなく、全角括弧（）を必ず使うこと。"
        "B 説明・用語: 冒頭でテーマを示し、説明を記号ごとに改行して並べ、名称との対応を問う。"
        "選択肢に関しては、（あ）選択肢、（い）選択肢、....のような形式にする。"
        "C 事例・分類: 判定課題を示し、具体例を記号ごとに挙げ、分類名や結果との対応を問う。"
        "選択肢に関しては、（あ）選択肢、（い）選択肢、....のような形式にする。"
        "末尾で記号順の組合せを選ばせる。選択肢の順序を揃え、正解は一つにする。四択問題"
        "指示は「〇〇の組み合わせとして、最も適切なものを1つ選べ。」の形式を基本とする。〇〇はそのまま書かずに、適切な表現に置き換える。"
    ),
    "Fill Blank": (
        "問題文の途中のどこかに空欄（_____）を一つ設ける。必ず（_____）を用いる。全角括弧（）を必ず使うこと。"
        "文脈に最もよく当てはまる用語や語句を四択から一つ選ばせる。"
        "指示は「空欄に最もよく当てはまる選択肢を1つ選べ。」の形式を基本とする。"
    ),
    "Inappropriate Choice": (
        "テーマに関する説明を四つ提示し、そのうち誤りを含む選択肢を一つだけ選ばせる。"
        "指示は「最も不適切なものを1つ選べ。」の形式を基本とする。"
    ),
}

# ジャンルとは独立した問い方。ジャンルの形式に合わせて適用する。
QUESTION_PATTERN_WEIGHTS = {
    "scenario_to_solution": 1,
    "definition_to_term": 1,
    "term_to_explanation": 1,
    "mechanism_or_process": 1,
    "compare_concepts": 1,
    "numerical_calculation": 1,
    "result_interpretation": 1,
    "case_classification": 1,
    "component_or_structure": 1,
    "principle_or_criterion": 1,
}

QUESTION_PATTERN_INSTRUCTIONS = {
    "scenario_to_solution": "状況や目的を示し、合う手法・対応策を問う。",
    "definition_to_term": "定義や特徴を示し、該当する用語・手法を問う。",
    "term_to_explanation": "概念を指定し、その正しい説明や性質を問う。",
    "mechanism_or_process": "処理の流れや仕組みを示し、該当する原理・手順を問う。",
    "compare_concepts": "近い概念を対比し、正しい違いや使い分けを問う。",
    "numerical_calculation": "必要な数値と条件を示し、指標・統計量などを計算させる。フォールバック: 計算問題が難しい場合は、近い概念を対比し、正しい違いや使い分けを問う形式にする。",
    "result_interpretation": "実験結果や指標値を示し、その意味や妥当な解釈を問う。",
    "case_classification": "具体的な事例を示し、該当する分類・判定を問う。",
    "component_or_structure": "システムやモデルの要件を示し、適切な構成要素・構造を問う。",
    "principle_or_criterion": "条件や主張を示し、適用される原則・基準を問う。",
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


def make_prompt(keyword, genre, difficulty, question_pattern):
    pattern_instruction = QUESTION_PATTERN_INSTRUCTIONS[question_pattern]
    return f"""G検定対策の新作四択問題を1問作ってください。

キーワード: {keyword['用語']}
技術的説明: {keyword['技術的な独自説明']}
関連用語: {keyword['関連用語']}
問題形式: {GENRE_INSTRUCTIONS[genre]}
問い方: {pattern_instruction} ジャンル固有の問題形式を保ったまま、この切り口を適用してください。
難易度: {DIFFICULTY_INSTRUCTIONS[difficulty]}

説明にない事実は追加せず、自然で簡潔な日本語で作ってください。
既存設問を写さず、正解が一つだけの問題にしてください。
関連用語から関係するものを比較対象や選択肢に使い、問い方に変化を付けてください。
overall_explanationは正解の概念・内容を根拠とともに簡潔に説明してください。選択肢の番号や位置を一切書かないでください。
「選択肢1」「選択肢4」「①」「正解は1」「正しい組み合わせは①」のように、番号や記号で正解を示す表現は絶対に禁止です。番号の代わりに、正解となる概念名や仕組みを主語にして説明してください。
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
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    tls_context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, timeout=180, context=tls_context) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OpenRouter HTTP {error.code} {error.reason}: {detail}"
        ) from error

    message = result["choices"][0]["message"]
    return (
        json.loads(message["content"]),
        result.get("usage", {}),
        result.get("provider"),
    )


def make_record(
    number, keyword, genre, difficulty, question, usage, provider, question_pattern
):
    correct_answer = question.get("correct_answer")
    options = question.get("options")
    if type(correct_answer) is not int or correct_answer not in (1, 2, 3, 4):
        raise ValueError("correct_answer must be an integer from 1 to 4")
    if not isinstance(options, list) or len(options) != 4 or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("options must contain four non-empty strings")
    if any(
        not isinstance(question.get(field), str) or not question[field].strip()
        for field in ("question", "overall_explanation")
    ):
        raise ValueError("question and overall_explanation must be non-empty strings")

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
        "provider": provider,
        "question_pattern": question_pattern,
        "generation_status": "success",
        "generation_error": None,
        "reasoning_tokens": (
            usage.get("completion_tokens_details") or {}
        ).get("reasoning_tokens"),
    }
    record.update({f"option_{i}": value for i, value in enumerate(options, 1)})
    return record


def generate_question_record(api_key, task):
    number, keyword, genre, difficulty, question_pattern = task
    prompt = make_prompt(keyword, genre, difficulty, question_pattern)
    question, usage, provider = generate_question(api_key, prompt)
    return make_record(
        number, keyword, genre, difficulty, question, usage, provider, question_pattern
    )


def make_error_record(task, error):
    number, keyword, genre, difficulty, question_pattern = task
    return {
        "question_id": f"gkgen_{number:04d}",
        "term_id": keyword["用語ID"],
        "keyword": keyword["用語"],
        "related_terms": keyword["関連用語"],
        "category": keyword["大カテゴリー"],
        "section": keyword["中カテゴリー"],
        "keyword_definition": keyword["技術的な独自説明"],
        "genre": genre,
        "difficulty": difficulty,
        "question_text": "",
        "correct_answers": "",
        "overall_explanation": "",
        "model": MODEL,
        "provider": None,
        "question_pattern": question_pattern,
        "generation_status": "error",
        "generation_error": f"{type(error).__name__}: {error}"[:2000],
        "reasoning_tokens": None,
        "option_1": "",
        "option_2": "",
        "option_3": "",
        "option_4": "",
    }


def write_questions(output, api_key, assignments, workers):
    tasks = [
        (number, keyword, genre, difficulty, question_pattern)
        for number, (keyword, genre, difficulty, question_pattern)
        in enumerate(assignments, start=1)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_question_record, api_key, task): task
            for task in tasks
        }

        errors = 0
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:
                errors += 1
                record = make_error_record(task, error)

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"完了: {completed}/{len(tasks)} (エラー: {errors})", flush=True)


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
    question_patterns = random.choices(
        list(QUESTION_PATTERN_WEIGHTS),
        weights=list(QUESTION_PATTERN_WEIGHTS.values()),
        k=args.count,
    )
    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    assignments = zip(selected, genres, difficulties, question_patterns)
    with args.output.open("x", encoding="utf-8") as output:
        write_questions(output, api_key, assignments, args.workers)

    print(f"生成完了: {args.output}")


if __name__ == "__main__":
    main()
