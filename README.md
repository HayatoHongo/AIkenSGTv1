# OpenAI / AIkenGPT 共通MMLU-style評価

## Overview

このリポジトリは、OpenAIモデルとAIkenGPTを同じ評価pipelineで比較します。

- OpenAIはモデル名と問題セットをCLIで指定します。
- AIkenGPTはColab Notebookの問題セット設定を変更してRun allします。
- 両方とも `MMLUEvaluator` を使い、モデル固有処理だけを `mmlu_eval/backends/` に分離しています。

正式なOpenAI entry pointは **`evaluate_mmlu_openai_permutation.py`** です。AIkenGPTでは、letter scoringに `evaluate_aikengpt_mmlu.ipynb`、選択肢本文のlikelihood scoringに `evaluate_aikengpt_mmlu_text.ipynb` を使います。

## Quick Start: OpenAI

Python 3.10以上で依存packageをインストールします。

```bash
python -m pip install -r requirements-openai.txt
```

API keyを設定します。

```bash
export OPENAI_API_KEY="YOUR_API_KEY"       # macOS / Linux
```

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"       # Windows PowerShell
```

モデル名と問題セットを指定して実行します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir ./datasets/mmlu
```

別モデル・別問題セットでは、主にこの2引数だけを変更します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model <another-model> \
  --data_dir ./datasets/another_dataset
```

APIを呼ばず、manifestと最終promptだけを確認するには `--dry_run` を付けます。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir ./datasets/mmlu \
  --limit 10 \
  --dry_run
```

`gpt-4o-mini`をChat Completionsで使う場合の例です。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir ./datasets/mmlu \
  --api_mode chat \
  --answer_prefix bare
```

標準の `--api_mode completions --answer_prefix space` は既存実験との互換設定です。利用モデルがendpointと回答token形式をサポートするか確認してください。API側の制約は [MMLU_REFACTOR.md](MMLU_REFACTOR.md) に記載しています。

## Quick Start: AIkenGPT

1. Colabで `evaluate_aikengpt_mmlu.ipynb`（letter）または `evaluate_aikengpt_mmlu_text.ipynb`（本文likelihood）を開きます。
2. 先頭の **User settings** で、少なくとも `DATA_DIR` と `OUTPUT_DIR` を変更します。
3. 必要なら `SUBJECT`、`LIMIT`、`MANIFEST_PATH` を変更します。
4. text版では `TEXT_REDUCTION` を `"sum"` または `"mean"` に設定します。
5. GPU runtimeを選び、Run allします。

checkpoint、tokenizer、device、dtypeは **Project settings** に分離しています。通常の問題セット変更では編集不要です。Drive上のリポジトリ位置が異なる場合だけ `PROJECT_DIR` も変更してください。

## Dataset Format

任意形式のCSVは受け付けません。利用者側で次の正式な入力形式へ変換してください。

```text
<data_dir>/
├── dev/
│   ├── <subject1>_dev.csv
│   └── <subject2>_dev.csv
└── test/
    ├── <subject1>_test.csv
    └── <subject2>_test.csv
```

各CSVはヘッダーなし、各行が厳密に6列です。

```text
question,choice_A,choice_B,choice_C,choice_D,correct_label
```

```csv
What is the capital of Japan?,Tokyo,Osaka,Kyoto,Nagoya,A
Which number is prime?,4,6,7,8,C
```

入力条件:

- 1列目は問題文、2〜5列目は4つの選択肢です。
- 6列目は正解ラベル `A`〜`D`。前後の空白は除去して検証します。
- 各 `test/<subject>_test.csv` に同名の `dev/<subject>_dev.csv` が必要です。`NTRAIN=0` でもdevファイルは必要です。
- `NTRAIN` / `--ntrain` は各subjectのdev行数以下にします。few-shotにはdev先頭から指定数を使います。
- test questionだけをpermutationし、few-shot例のchoice orderは固定です。
- 引用符や改行を含むfieldは通常のCSV規則でquoteしてください。
- Pandasが欠損値と解釈する `NA`、`N/A`、`null` などはprompt内で `nan` になることがあります。問題文・選択肢では避けてください。
- ヘッダーや追加列、4択以外の選択肢数には対応しません。

`download_mmlu_hf.py` で既存MMLUデータを取得・変換できます。データ自体はリポジトリに含めません。

## Configuration

| 目的 | OpenAI CLI | AIkenGPT Notebook |
| --- | --- | --- |
| モデル変更 | `--model` | 原則固定（Project settings） |
| 問題セット変更 | `--data_dir` | `DATA_DIR` |
| 出力先 | `--output_dir` | `OUTPUT_DIR` |
| subject指定 | `--subject` | `SUBJECT` |
| few-shot数 | `--ntrain` | `NTRAIN` |
| sampling比率 | `--sample_frac` | `SAMPLE_FRAC` |
| 問題数制限 | `--limit` | `LIMIT` |
| sampling seed | `--seed` | `SEED` |
| 同じ問題集合を再利用 | `--manifest` | `MANIFEST_PATH` |

主要なOpenAIオプション:

| オプション | 標準値 | 意味 |
| --- | --- | --- |
| `--model` | `gpt-4o-mini` | OpenAI model identifier |
| `--data_dir` | 必須 | 上記形式の問題セット |
| `--subject` | 全subject | 1 subjectだけを評価 |
| `--ntrain` | `5` | dev先頭から使うfew-shot数。`0`はzero-shot |
| `--sample_frac` | `0.1` | 全subjectのtest問題を結合した後の抽出割合 |
| `--limit` | `0` | 抽出後の先頭件数。`0`は抽出分をすべて使用 |
| `--seed` | `42` | sampling seed |
| `--manifest` | なし | 既存manifestの全行を記載順に使用 |
| `--output_dir` | `results_permutation_shared` | 出力先 |
| `--dry_run` | off | APIを呼ばずmanifestとpromptを生成 |

`--context_policy reduce --context_tokenizer gpt2 --max_context_length 2048` が標準設定です。`--context_policy fixed` を選んだ場合も2048-token上限は検査され、1つのpermutationでも超過すればエラーになります。

`python evaluate_mmlu_openai_permutation.py --help` ですべてのoptionを確認できます。

## Reusing a Manifest

複数モデルで同じ問題集合を使う場合は、最初のrunが生成した `.manifest.csv` を再利用します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model <another-model> \
  --data_dir ./datasets/mmlu \
  --manifest ./manifests/mmlu_10percent_seed42.csv
```

Notebookでは `MANIFEST_PATH = "/content/drive/MyDrive/manifests/mmlu_10percent_seed42.csv"` とします。

```text
同じdataset + 同じEvalConfig + 同じmanifest
→ backend直前まで同じquestion / choices / label / few-shot / permutation / prompt
```

`cases_hash` は各問題の4 cases、`.prompts.jsonl` は完全な入力を記録します。2 runの入力は厳密比較できます。

```bash
python -m mmlu_eval.compare path/to/openai.prompts.jsonl path/to/aikengpt.prompts.jsonl
```

## Evaluation Method

1. `dev/` と `test/` を読み込みます。
2. 全test問題を結合し、seed付きでsamplingします。manifest指定時はその全行を記載順に使います。
3. dev先頭のfew-shot例を追加します。demonstrationのchoice orderは固定です。
4. test questionだけを4つのcyclic orderで提示します。
5. backendが表示位置順の4 choice scoreを返します。
6. 共通側でsoftmaxし、各scoreを元のsemantic choiceへ戻します。
7. original orderのargmaxをbaseline、4配置のmapped probability平均のargmaxをdebiased predictionとします。
8. 問題別、subject別、全体のmicro accuracyを共通コードで計算します。

標準実験ではGPT-2 tokenizerによる共通2048-token budgetを使います。5-shotを指定していても、4つのpermutationのうち1つでもbudgetを超える問題では、OpenAIとAIkenGPTの両方で同じようにshot数を5から0まで減らします。4配置すべてが収まる最大値を `effective_ntrain` としてmanifestと結果CSVに記録します。これにより、両backendへ同じshot数・同じpromptが渡ります。0-shotでも超過する場合はエラーになります。

text版の `TEXT_REDUCTION="sum"` は候補tokenのlog probability合計で、長い候補ほど不利になりやすい方式です。`"mean"` はtoken数で平均する長さ正規化scoreです。

## Output Files

| ファイル | 役割 |
| --- | --- |
| 結果 `.csv` | 問題、正解、baseline/debiased予測、各choice/permutationのscoreとprobability |
| `.manifest.csv` | 問題順、dataset/config/cases hash、effective ntrain |
| `.prompts.jsonl` | backend直前の全evaluation cases |
| `.run.json` | pipeline、backend、model、runtime、package version |
| `.subjects.csv` | subject別集計 |
| `.overall.json` | 全問題のmicro集計 |

結果は1問ごとにatomic保存します。resume時はrun metadata、manifest、cases hashを検証し、未完了分だけを続行します。

## Entry Points and Validation

- `evaluate_mmlu_openai_permutation.py`: 正式なOpenAI entry point。
- `evaluate_mmlu_permutation.py`: 旧ファイル名の非推奨compatibility wrapper。
- `evaluate_openai.py`: 共通pipeline導入前のlegacy implementation。過去実験の再現専用で、新規評価には使いません。
- `legacy/`: リファクタリング前の原本。新規実験の入口ではありません。

```bash
python -m pip install -r requirements-mmlu.txt
python -m unittest discover -s tests -v
python tests/verify_real_data.py --data_dir ./datasets/mmlu
```

回帰テスト、既存設計との差、API制約、評価値が変化しうる箇所は [MMLU_REFACTOR.md](MMLU_REFACTOR.md) を参照してください。

## References

- [MMLU original repository](https://github.com/hendrycks/test)
- [MMLU dataset](https://huggingface.co/datasets/cais/mmlu)
- [OpenAI API documentation](https://developers.openai.com/api/docs)
