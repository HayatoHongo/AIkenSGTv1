# MMLU Evaluation with OpenAI API

OpenAI APIを用いて、MMLU（Massive Multitask Language Understanding）の zero-shot / 5-shot 評価を行うコードです。

元実装として `hendrycks/test` の `evaluate.py` を参考にし、現在のOpenAI Python SDKおよびAPI仕様で動作するよう変更しました。

## Evaluation Result

評価対象モデル:

`davinci-002`

MMLU test set:

* 57 subjects
* 14,042 questions

| Setting   | Accuracy |
| --------- | -------: |
| zero-shot |   58.86% |
| 5-shot    |   61.22% |

5-shotではzero-shotと比較して、accuracyが2.36 percentage points向上しました。

## Evaluation Method

基本的な評価方法は、元のMMLU実装を踏襲しています。

各問題について、以下の形式のpromptをモデルへ入力します。

```text
Question
A. ...
B. ...
C. ...
D. ...
Answer:
```

モデルが `Answer:` の直後に出力するトークンについて、A / B / C / D のlog probabilityを取得します。

その後、

```text
prediction = argmax(log P(A), log P(B), log P(C), log P(D))
```

として予測ラベルを決定し、MMLUの正解ラベルと比較してaccuracyを計算します。

### zero-shot

評価対象の問題のみをpromptへ入力します。

### 5-shot

各subjectのdev setから5問の解答付き例題をpromptの前に追加し、その後に評価対象の問題を入力します。

## Difference from the Original MMLU Evaluator

元の `hendrycks/test/evaluate.py` では、旧OpenAI Completion APIを用いて

```python
logprobs=100
```

として上位100候補のlog probabilityを取得しています。

現在のCompletions APIでは同一の設定を利用できないため、本評価では

```python
logprobs=5
```

を使用しています。

A / B / C / D のいずれかが返された候補に含まれていない場合は、元のMMLU実装と同様に、その選択肢のlog probabilityを `-100` として扱います。

したがって、本評価は元MMLUの評価方法を可能な限り踏襲していますが、API仕様の違いにより完全に同一の条件ではありません。

## Dataset

MMLU:

https://huggingface.co/datasets/cais/mmlu

Berkeleyの元データ配布先へ接続できなかったため、Hugging Faceの `cais/mmlu` を使用しました。

`download_mmlu_hf.py` によりHugging Face版のデータを取得し、元の `evaluate.py` が期待するCSV形式へ変換しています。

データ自体はGitリポジトリには含めません。

## Setup

Python環境で必要なパッケージをインストールします。

```bash
python -m pip install -U openai pandas numpy datasets
```

OpenAI API keyを環境変数へ設定します。

bash:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

MMLUデータを取得・変換します。

```bash
python download_mmlu_hf.py
```

## Usage

### Test Run

各subjectから1問だけ評価します。

zero-shot:

```bash
python evaluate_openai.py \
  -e davinci-002 \
  -k 0 \
  --limit 1 \
  -d data
```

5-shot:

```bash
python evaluate_openai.py \
  -e davinci-002 \
  -k 5 \
  --limit 1 \
  -d data
```

### Full Evaluation

zero-shot:

```bash
python evaluate_openai.py \
  -e davinci-002 \
  -k 0 \
  --limit 0 \
  -d data
```

5-shot:

```bash
python evaluate_openai.py \
  -e davinci-002 \
  -k 5 \
  --limit 0 \
  -d data
```

`--limit 0` は各subjectのtest setをすべて評価する指定です。

## Output

各subjectについてCSV形式の評価結果を保存します。

例:

```text
results_davinci-002_abstract_algebra_0shot.csv
results_davinci-002_abstract_algebra_5shot.csv
```

本リポジトリでは結果を以下に整理しています。

```text
results/
├── zero_shot/
└── five_shot/
```

各CSVには、MMLUの問題・正解ラベル・正誤判定・A/B/C/Dについて正規化した確率が含まれます。

## References

* MMLU original repository: `hendrycks/test`
* MMLU dataset: `cais/mmlu`
* OpenAI API
