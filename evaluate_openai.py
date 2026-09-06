"""Legacy evaluator kept only for reproducing historical runs.

New evaluations must use evaluate_mmlu_openai_permutation.py, which routes
through the shared mmlu_eval pipeline used by AIkenGPT.
"""
import warnings

warnings.warn(
    "evaluate_openai.py is outside the shared mmlu_eval pipeline. "
    "Use evaluate_mmlu_openai_permutation.py for new evaluations.",
    FutureWarning,
    stacklevel=2,
)

import argparse
import os
import time

import numpy as np
import pandas as pd
from openai import OpenAI


choices = ["A", "B", "C", "D"]

client = OpenAI()


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]

    k = df.shape[1] - 2

    for j in range(k):
        prompt += "\n{}. {}".format(
            choices[j],
            df.iloc[idx, j + 1],
        )

    prompt += "\nAnswer:"

    if include_answer:
        prompt += " {}\n\n".format(
            df.iloc[idx, k + 1]
        )

    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = (
        "The following are multiple choice questions "
        "(with answers) about{}.\n\n"
    ).format(format_subject(subject))

    if k == -1:
        k = train_df.shape[0]

    for i in range(k):
        prompt += format_example(train_df, i)

    return prompt


def eval_subject(args, subject, dev_df, test_df):
    cors = []
    all_probs = []

    if args.limit == 0:
        n_questions = test_df.shape[0]
    else:
        n_questions = min(
        args.limit,
        test_df.shape[0],
    )

    for i in range(n_questions):
        # 本番問題：答えなし
        prompt_end = format_example(
            test_df,
            i,
            include_answer=False,
        )

        # k-shot用のお手本
        train_prompt = gen_prompt(
            dev_df,
            subject,
            args.ntrain,
        )

        prompt = train_prompt + prompt_end

        # OpenAI Completion API
        # OpenAI Completion API
        # 一時的なAPIエラーに備えて最大5回まで再試行
        for attempt in range(5):
            try:
                response = client.completions.create(
                    model=args.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,
                    temperature=0,
                    echo=True,
        )

        # 成功したらretryループを抜ける
                break

            except Exception as e:
        # 5回目も失敗した場合は処理を停止
                if attempt == 4:
                    raise

                wait = 2 ** attempt

                print(
                    f"API error: {e}\n"
                    f"Retrying in {wait} seconds..."
                )

                time.sleep(wait)

        # 最後の出力位置のtop logprobs
        top_logprobs = (
            response
            .choices[0]
            .logprobs
            .top_logprobs[-1]
        )

        lprobs = []

        for ans in choices:
            token = " " + ans

            if token in top_logprobs:
                lprobs.append(
                    top_logprobs[token]
                )
            else:
                print(
                    f"Warning: {ans} not found. "
                    "Adding log prob -100."
                )
                lprobs.append(-100)

        # A/B/C/Dだけで正規化
        probs = softmax(
            np.array(lprobs)
        )

        # 最大logprobの選択肢
        pred = choices[
            np.argmax(lprobs)
        ]

        # 正解
        label = test_df.iloc[
            i,
            test_df.shape[1] - 1
        ]

        cor = pred == label

        cors.append(cor)
        all_probs.append(probs)

        print(
            f"{i + 1}/{n_questions} "
            f"pred={pred} "
            f"label={label} "
            f"correct={cor}"
        )

        print(
            "  logprobs:",
            dict(zip(choices, lprobs)),
        )

    acc = np.mean(cors)

    print()
    print(
        f"Average accuracy {acc:.3f} "
        f"- {subject}"
    )

    return np.array(cors), np.array(all_probs)


def main(args):
    test_dir = os.path.join(
        args.data_dir,
        "test",
    )

    # testフォルダから全subject名を取得
    subjects = sorted([
        filename.replace("_test.csv", "")
        for filename in os.listdir(test_dir)
        if filename.endswith("_test.csv")
    ])

    # --subject が指定された場合だけ1分野に限定
    if args.subject is not None:
        if args.subject not in subjects:
            raise ValueError(
                f"Unknown subject: {args.subject}"
            )

        subjects = [args.subject]

    print(f"Number of subjects: {len(subjects)}")

    all_cors = []

    for subject_index, subject in enumerate(
        subjects,
        start=1,
    ):
        print()
        print("=" * 60)
        print(
            f"[{subject_index}/{len(subjects)}] "
            f"{subject}"
        )
        print("=" * 60)

        dev_path = os.path.join(
            args.data_dir,
            "dev",
            subject + "_dev.csv",
        )

        test_path = os.path.join(
            args.data_dir,
            "test",
            subject + "_test.csv",
        )

        dev_df = pd.read_csv(
            dev_path,
            header=None,
        )

        test_df = pd.read_csv(
            test_path,
            header=None,
        )

        output_file = (
            f"results_"
            f"{args.model}_"
            f"{subject}_"
            f"{args.ntrain}shot.csv"
        )

        # 今回この分野で評価すべき問題数
        if args.limit == 0:
            expected_questions = test_df.shape[0]
        else:
            expected_questions = min(
                args.limit,
                test_df.shape[0],
            )

        # すでに完全な結果CSVが存在するなら再利用
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)

            if len(existing_df) == expected_questions:
                print(
                    f"Already completed: {subject} "
                    f"({len(existing_df)} questions). Skipping."
                )

                # 最後のoverall accuracy計算にも既存結果を含める
                existing_cors = (
                    existing_df["correct"]
                    .astype(str)
                    .str.lower()
                    .eq("true")
                    .to_numpy()
                )

                all_cors.extend(existing_cors)

                continue

        cors, probs = eval_subject(
            args,
            subject,
            dev_df,
            test_df,
        )

        all_cors.extend(cors)

        # 今までと同じ形式で結果保存
        output_df = test_df.iloc[
            : len(cors)
        ].copy()

        output_df["correct"] = cors

        for j, choice in enumerate(choices):
            output_df[
                f"{choice}_prob"
            ] = probs[:, j]

        

        output_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Saved results to {output_file}"
        )

    # 全subjectを合わせたaccuracy
    overall_accuracy = np.mean(all_cors)

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"n-shot: {args.ntrain}")
    print(f"Subjects: {len(subjects)}")
    print(f"Questions: {len(all_cors)}")
    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        "-e",
        type=str,
        default="davinci-002",
    )

    parser.add_argument(
        "--ntrain",
        "-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    main(args)
