import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from openai import OpenAI, BadRequestError

choices = ["A", "B", "C", "D"]

client = OpenAI()

def softmax(x):
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

class MissingChoiceLogprobError(Exception):
    pass


def format_subject(subject):
    return " " + " ".join(subject.split("_"))


def format_example(
    df,
    idx,
    include_answer=True,
    order=None,
):
    """
    order:
        表示位置 A/B/C/D に、
        元のどの選択肢を置くかを表す。

        identity:
            [0, 1, 2, 3]

        1つcyclic shift:
            [1, 2, 3, 0]
    """

    prompt = str(df.iloc[idx, 0])

    k = df.shape[1] - 2

    if order is None:
        order = list(range(k))

    for position, original_index in enumerate(order):
        prompt += "\n{}. {}".format(
            choices[position],
            df.iloc[idx, original_index + 1],
        )

    prompt += "\nAnswer:"

    if include_answer:
        original_label = str(
            df.iloc[idx, k + 1]
        ).strip()

        original_index = choices.index(
            original_label
        )

        # permutation後、その正解がどの位置に移ったか
        new_position = order.index(
            original_index
        )

        prompt += " {}\n\n".format(
            choices[new_position]
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
        prompt += format_example(
            train_df,
            i,
        )

    return prompt


def cyclic_order(shift):
    """
    shift=0: [0,1,2,3]
    shift=1: [1,2,3,0]
    shift=2: [2,3,0,1]
    shift=3: [3,0,1,2]
    """
    return [
        (i + shift) % 4
        for i in range(4)
    ]

def call_openai(args, prompt):
    for attempt in range(5):
        try:
            time.sleep(1.0)
            response = client.completions.create(
                model=args.model,
                prompt=prompt,
                max_tokens=1,
                temperature=0,
                logprobs=5,
            )

            top_logprobs = (
                response
                .choices[0]
                .logprobs
                .top_logprobs[-1]
            )

            # top5の最低logprob
            fifth_logprob = min(
                top_logprobs.values()
            )

            lprobs = []
            missing_count = 0

            for choice in choices:
                token = " " + choice

                if token in top_logprobs:
                    lprobs.append(
                        top_logprobs[token]
                    )
                else:
                    lprobs.append(-100.0)
                    missing_count += 1

            # -------------------------
            # -100近似の最大誤差
            # -------------------------
            if missing_count == 4:
                raise RuntimeError(
                "All A/B/C/D choices are missing from top-5."
            )
            
            if missing_count == 0:
                error_bound = 0.0

            else:
                visible_mass = sum(
                    np.exp(lp)
                    for lp in lprobs
                    if lp > -100
                )

                max_missing_mass = (
                    missing_count
                    * np.exp(fifth_logprob)
                )

                error_bound = (
                    max_missing_mass
                    / (
                        visible_mass
                        + max_missing_mass
                    )
                )

            probs = softmax(
                np.array(lprobs)
            )

            input_tokens = (
                response.usage.prompt_tokens
            )

            output_tokens = (
                response.usage.completion_tokens
            )

            return (
                probs,
                input_tokens,
                output_tokens,
                missing_count,
                error_bound,
                fifth_logprob,
            )

        except BadRequestError:
            raise

        except Exception as e:
            if attempt == 4:
                raise

            wait = 2 ** attempt

            print(
                f"API error: {e}\n"
                f"Retrying in {wait} seconds..."
            )

            time.sleep(wait)

def evaluate_question(
    args,
    subject,
    test_index,
    train_prompt,
    test_df,
):
    """
    1問について4 cyclic permutationsを評価。
    この関数1個がworker 1個に相当する。
    """

    permutation_probs = []
    mapped_probs = []

    permutation_missing_counts = []
    permutation_error_bounds = []
    permutation_fifth_logprobs = []

    total_input_tokens = 0
    total_output_tokens = 0

    for shift in range(4):
        order = cyclic_order(shift)

        prompt_end = format_example(
            test_df,
            test_index,
            include_answer=False,
            order=order,
        )

        prompt = (
            train_prompt
            + prompt_end
        )

        (
            position_probs,
            input_tokens,
            output_tokens,
            missing_count,
            error_bound,
            fifth_logprob,
        ) = call_openai(
            args,
            prompt,
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        permutation_probs.append(
            position_probs
        )

        permutation_missing_counts.append(
            missing_count
        )

        permutation_error_bounds.append(
            error_bound
        )

        permutation_fifth_logprobs.append(
            fifth_logprob
        )

        # 表示位置A/B/C/Dから
        # 元のsemantic optionへ戻す
        mapped = np.zeros(4)

        for position, original_index in enumerate(
            order
        ):
            mapped[original_index] = (
                position_probs[position]
            )

        mapped_probs.append(mapped)

    # permutation 0 = 元の順番
    baseline_probs = permutation_probs[0]

    baseline_pred = choices[
        np.argmax(baseline_probs)
    ]

    # 4 permutationを元の選択肢に戻して平均
    debiased_probs = np.mean(
        np.stack(mapped_probs),
        axis=0,
    )

    debiased_pred = choices[
        np.argmax(debiased_probs)
    ]

    row = test_df.iloc[test_index]

    label = str(
        row.iloc[-1]
    ).strip()

    result = {
        "subject": subject,
        "test_index": test_index,
        "question": row.iloc[0],
        "A": row.iloc[1],
        "B": row.iloc[2],
        "C": row.iloc[3],
        "D": row.iloc[4],
        "label": label,

        "baseline_pred": baseline_pred,
        "baseline_correct": (
            baseline_pred == label
        ),

        "debiased_pred": debiased_pred,
        "debiased_correct": (
            debiased_pred == label
        ),

        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }

    for shift in range(4):
        result[
            f"perm{shift}_missing_count"
        ] = permutation_missing_counts[shift]

        result[
            f"perm{shift}_error_bound"
        ] = permutation_error_bounds[shift]

        result[
            f"perm{shift}_fifth_logprob"
        ] = permutation_fifth_logprobs[shift]

    # 1問全体の診断値
    result["mean_error_bound"] = float(
        np.mean(permutation_error_bounds)
    )

    result["max_error_bound"] = float(
        np.max(permutation_error_bounds)
    )

    result["has_all_missing"] = any(
        x == 4
        for x in permutation_missing_counts
    )

    # baseline probabilities
    for j, choice in enumerate(choices):
        result[
            f"baseline_{choice}_prob"
        ] = baseline_probs[j]

    # debiased probabilities
    for j, choice in enumerate(choices):
        result[
            f"debiased_{choice}_prob"
        ] = debiased_probs[j]

    # 各permutationの生のA/B/C/D probabilitiesも保存
    for shift in range(4):
        for j, choice in enumerate(choices):
            result[
                f"perm{shift}_{choice}_prob"
            ] = permutation_probs[
                shift
            ][j]

    return result


def make_sample(
    data,
    sample_frac,
    seed,
    limit,
):
    """
    MMLU全subjectをまとめてから
    全体の10%をランダム抽出する。
    """

    candidates = []

    for subject, (_, test_df) in data.items():
        for test_index in range(
            len(test_df)
        ):
            candidates.append(
                (
                    subject,
                    test_index,
                )
            )

    n_total = len(candidates)

    n_sample = round(
        n_total * sample_frac
    )

    rng = np.random.default_rng(seed)

    selected_indices = rng.choice(
        n_total,
        size=n_sample,
        replace=False,
    )

    sampled = [
        candidates[i]
        for i in selected_indices
    ]

    if limit > 0:
        sampled = sampled[:limit]

    print(
        f"Total MMLU questions: {n_total}"
    )

    print(
        f"Sampled questions: {len(sampled)}"
    )

    print(
        f"Sample fraction: {sample_frac}"
    )

    print(
        f"Random seed: {seed}"
    )

    return sampled


def print_metrics(df):
    baseline_acc = (
        df["baseline_correct"]
        .astype(bool)
        .mean()
    )

    debiased_acc = (
        df["debiased_correct"]
        .astype(bool)
        .mean()
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Questions: {len(df)}"
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_acc:.4f}"
    )

    print(
        f"Debiased accuracy: "
        f"{debiased_acc:.4f}"
    )

    print(
        f"Delta: "
        f"{debiased_acc - baseline_acc:+.4f}"
    )

    # labelごとのrecall
    print()
    print("Recall by correct-answer label:")

    for prediction_type in [
        "baseline",
        "debiased",
    ]:
        recalls = []

        print(
            f"\n{prediction_type}:"
        )

        for choice in choices:
            subset = df[
                df["label"] == choice
            ]

            recall = (
                subset[
                    f"{prediction_type}_pred"
                ]
                == choice
            ).mean()

            recalls.append(recall)

            print(
                f"  {choice}: "
                f"{recall:.4f}"
            )

        rstd = np.std(recalls)

        print(
            f"  RStd: {rstd:.4f}"
        )


def main(args):
    test_dir = os.path.join(
        args.data_dir,
        "test",
    )

    subjects = sorted([
        filename.replace(
            "_test.csv",
            "",
        )
        for filename in os.listdir(
            test_dir
        )
        if filename.endswith(
            "_test.csv"
        )
    ])

    if args.subject is not None:
        if args.subject not in subjects:
            raise ValueError(
                f"Unknown subject: "
                f"{args.subject}"
            )

        subjects = [
            args.subject
        ]

    print(
        f"Number of subjects: "
        f"{len(subjects)}"
    )

    # 全データを読み込む
    data = {}

    train_prompts = {}

    for subject in subjects:
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

        data[subject] = (
            dev_df,
            test_df,
        )

        # subjectごとに一度だけ作ればよい
        train_prompts[subject] = (
            gen_prompt(
                dev_df,
                subject,
                args.ntrain,
            )
        )

    sampled = make_sample(
        data,
        args.sample_frac,
        args.seed,
        args.limit,
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    frac_tag = str(
        args.sample_frac
    ).replace(".", "p")

    subject_tag = (
        args.subject
        if args.subject is not None
        else "all"
    )

    output_file = os.path.join(
        args.output_dir,
        (
            f"results_"
            f"{args.model}_"
            f"{args.ntrain}shot_"
            f"seed{args.seed}_"
            f"frac{frac_tag}_"
            f"{subject_tag}.csv"
        ),
    )

    manifest_file = os.path.join(
        args.output_dir,
        (
            f"sample_manifest_"
            f"seed{args.seed}_"
            f"frac{frac_tag}_"
            f"{subject_tag}.csv"
        ),
    )

    # 抽出問題一覧を保存
    pd.DataFrame(
        sampled,
        columns=[
            "subject",
            "test_index",
        ],
    ).to_csv(
        manifest_file,
        index=False,
    )

    print(
        f"Saved sample manifest to "
        f"{manifest_file}"
    )

    # 中断から再開できるようにする
    results = []

    completed = set()

    if os.path.exists(output_file):
        existing_df = pd.read_csv(
            output_file
        )

        results = existing_df.to_dict(
            orient="records"
        )

        completed = set(
            zip(
                existing_df["subject"],
                existing_df[
                    "test_index"
                ].astype(int),
            )
        )

        print(
            f"Existing results: "
            f"{len(completed)}"
        )

    pending = [
        item
        for item in sampled
        if item not in completed
    ]

    print(
        f"Pending questions: "
        f"{len(pending)}"
    )

    print(
        f"Workers: {args.workers}"
    )

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_to_item = {}

        for subject, test_index in pending:
            _, test_df = data[subject]

            future = executor.submit(
                evaluate_question,
                args,
                subject,
                test_index,
                train_prompts[subject],
                test_df,
            )

            future_to_item[future] = (
                subject,
                test_index,
            )

        completed_now = 0

        for future in as_completed(
            future_to_item
        ):
            subject, test_index = (
                future_to_item[future]
            )

            result = future.result()

            results.append(result)

            completed_now += 1

            print(
                f"{completed_now}/"
                f"{len(pending)} "
                f"{subject}[{test_index}] "
                f"baseline="
                f"{result['baseline_pred']} "
                f"debiased="
                f"{result['debiased_pred']} "
                f"label="
                f"{result['label']}"
            )

            # 10問ごとにcheckpoint
            if completed_now % 10 == 0:
                checkpoint_df = (
                    pd.DataFrame(results)
                    .sort_values(
                        [
                            "subject",
                            "test_index",
                        ]
                    )
                )

                checkpoint_df.to_csv(
                    output_file,
                    index=False,
                )

    result_df = (
        pd.DataFrame(results)
        .sort_values(
            [
                "subject",
                "test_index",
            ]
        )
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved results to "
        f"{output_file}"
    )

    print(
        f"Total input tokens: "
        f"{result_df['input_tokens'].sum()}"
    )

    print(
        f"Total output tokens: "
        f"{result_df['output_tokens'].sum()}"
    )

    print_metrics(
        result_df
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        "-e",
        type=str,
        default="gpt-4o-mini",
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
        "--output_dir",
        type=str,
        default="results_permutation",
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
    )

    # 0なら抽出した10%すべて。
    # 動作確認時だけ10などにする。
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    main(args)