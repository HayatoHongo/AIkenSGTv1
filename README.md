# AIkenGPT / OpenAI MMLU-style Evaluation

まず、
git clone https://github.com/HayatoHongo/AIkenSGTv1.git
cd AIkenSGTv1
git switch tayama
cd ..
git clone https://github.com/hendrycks/test.git mmlu-reference
を実行してください。

このリポジトリは、4択問題データを用いて **OpenAIモデル** と **AIkenGPT** を共通の評価パイプラインで評価するためのものです。

主な用途は次の2つです。

1. **AIkenGPT**
   - AIkenGPTのモデルは基本的に固定し、評価する問題セットを差し替えて評価する。
   - Google Colab上でNotebookを実行する。
2. **OpenAIモデル**
   - 評価する問題セットとモデル名の両方を差し替えて評価する。
   - CLIから実行する。

OpenAIとAIkenGPTではモデルの呼び出し方法は異なりますが、問題の読み込み、sampling、few-shot prompt、選択肢permutation、位置バイアス補正、集計などは共通の `mmlu_eval` パイプラインを使用します。

---

## 1. まず何をすればよいか

初めて使う場合は、いきなり全問を評価せず、次の順番で試すことを推奨します。

### OpenAIを評価したい場合

1. リポジトリをcloneする
2. Python環境を作る
3. 問題データを指定形式で用意する
4. OpenAI API keyを設定する
5. `--dry_run --limit 10` でpromptを確認する
6. 10問だけ実際に評価する
7. 結果を確認する
8. 問題なければ問題数やモデル名を変更して本評価する

### AIkenGPTを評価したい場合

1. Google ColabでNotebookを開く
2. GPU runtimeを選ぶ
3. リポジトリと問題データの場所を設定する
4. `LIMIT = 10` にしてpreflightを確認する
5. 10問だけ実際に評価する
6. 結果を確認する
7. 問題なければ問題セット・問題数を変更して本評価する

### OpenAIとAIkenGPTを同じ問題で比較したい場合

1. 片方のrunでmanifestを作る
2. もう片方のrunで同じmanifestを指定する
3. `prompts.jsonl` を比較し、backend直前の入力が一致していることを確認する
4. そのうえで結果を比較する

---

# 2. リポジトリの主な構成

概念的には次の構成です。

```text
.
├── evaluate_mmlu_openai_permutation.py   # OpenAI評価の正式な入口
├── evaluate_mmlu_permutation.py          # deprecated compatibility wrapper
├── evaluate_aikengpt_mmlu.ipynb          # AIkenGPT: A/B/C/D letter scoring
├── evaluate_aikengpt_mmlu_text.ipynb     # AIkenGPT: 選択肢本文の尤度
│
├── mmlu_eval/
│   ├── core.py                           # 共通評価パイプライン
│   ├── cli.py                            # OpenAI CLI
│   ├── compare.py                        # pre-backend promptの比較
│   └── backends/
│       ├── openai_backend.py             # OpenAI API固有処理
│       ├── aikengpt_backend.py            # AIkenGPT推論・scoring
│       └── aikengpt_model.py              # AIkenGPTモデル定義
│
└── tests/
    └── test_mmlu.py                      # 共通評価パイプラインのテスト
```

`evaluate_mmlu_permutation.py` は互換性のために残されているdeprecated wrapperです。**新しくOpenAI評価を行う場合は `evaluate_mmlu_openai_permutation.py` を使用してください。**

旧来の独立実装が残っている場合も、共通pipelineを使った比較実験では使用しないでください。

---

# 3. 問題データの用意

このコードは任意形式のCSVを自動変換しません。

**評価したい問題データ側を、以下のMMLU-style形式に合わせてください。**

## 3.1 ディレクトリ構成

1つの問題セットは、次のように `dev/` と `test/` を持ちます。

```text
<data_dir>/
├── dev/
│   ├── <subject1>_dev.csv
│   ├── <subject2>_dev.csv
│   └── ...
└── test/
    ├── <subject1>_test.csv
    ├── <subject2>_test.csv
    └── ...
```

例:

```text
datasets/
└── mmlu/
    ├── dev/
    │   ├── abstract_algebra_dev.csv
    │   └── high_school_biology_dev.csv
    └── test/
        ├── abstract_algebra_test.csv
        └── high_school_biology_test.csv
```

`dev` と `test` でsubject名を一致させてください。

## 3.2 CSV形式

各CSVは **ヘッダーなしの6列** です。

```text
question, choice_A, choice_B, choice_C, choice_D, correct_label
```

例:

```csv
What is the capital of Japan?,Tokyo,Osaka,Kyoto,Nagoya,A
Which number is prime?,4,6,7,8,C
```

各列の意味:

| 列 | 内容 |
|---|---|
| 1 | 問題文 |
| 2 | 選択肢A |
| 3 | 選択肢B |
| 4 | 選択肢C |
| 5 | 選択肢D |
| 6 | 正解ラベル `A` / `B` / `C` / `D` |

注意:

- ヘッダー行は付けません。
- 正解ラベルは `A`, `B`, `C`, `D` のいずれかにしてください。
- 問題文や選択肢にカンマが含まれる場合は、通常のCSV規則に従って引用符で囲んでください。
- `ntrain=5` なら、各subjectの `dev` に少なくとも5問必要です。
- zero-shot (`ntrain=0`) の場合でも、現在のloaderは `dev/` と `test/` の両方を読み込むため、対応するdevファイルを用意してください。

---

# 4. 共通の評価方法

OpenAIとAIkenGPTは、モデル固有のscoring部分を除いて同じ `MMLUEvaluator` を通ります。

評価の流れは次のとおりです。

```text
問題データ読み込み
        ↓
sampling / manifest
        ↓
few-shot prompt生成
        ↓
共通context lengthチェック
        ↓
test questionの4 cyclic permutations
        ↓
モデル固有backendでscoring
        ↓
表示位置から元の選択肢へscoreを戻す
        ↓
baseline / debiased prediction
        ↓
accuracy・各種指標を保存
```

## 4.1 Prompt

few-shotの場合、devセットから先頭 `ntrain` 問をdemonstrationとして使用します。

概念的には次の形式です。

```text
The following are multiple choice questions (with answers) about <subject>.

<Question 1>
A. ...
B. ...
C. ...
D. ...
Answer: B

...

<Target question>
A. ...
B. ...
C. ...
D. ...
Answer:
```

few-shot demonstrationの選択肢順は固定し、**評価対象のtest questionだけをpermutationします。**

## 4.2 4 cyclic permutations

各test questionについて、選択肢を4通りのcyclic orderで評価します。

```text
Permutation 0: A B C D
Permutation 1: B C D A
Permutation 2: C D A B
Permutation 3: D A B C
```

これにより、内容ではなく「Aの位置を選びやすい」などの回答位置バイアスの影響を確認・軽減します。

## 4.3 Baselineとdebiased

### Baseline

元の順番 (`Permutation 0`) だけを使ったpredictionです。

### Debiased

1. 各permutation内で4候補のscoreをsoftmaxする
2. 表示位置A/B/C/Dから、元のsemantic choiceへ確率を戻す
3. 4 permutationsの確率を平均する
4. 平均確率が最大の選択肢をpredictionとする

この結果、各問題について `baseline_pred` と `debiased_pred` の両方が保存されます。

---

# 5. Context lengthの扱い

AIkenGPTの最大context長に合わせ、比較実験では標準で次の設定を使用します。

```text
CONTEXT_POLICY = reduce
CONTEXT_TOKENIZER = gpt2
MAX_CONTEXT_LENGTH = 2048
```

## `reduce`

requested `ntrain` から0まで順に減らし、**4つすべてのpermutationが共通GPT-2 tokenizerで2048 tokens以内になる最大のk** を採用します。

例:

```text
requested ntrain = 5
5-shot → 1つのpermutationが2048超過
4-shot → 4つすべて2048以内

→ effective_ntrain = 4
```

その問題で実際に使われたshot数は `effective_ntrain` としてmanifest・resultに記録されます。

この処理はモデルに依存せず共通pipelineで行うため、同じdataset・設定・manifestを使うOpenAIとAIkenGPTでは同じ `effective_ntrain` になります。

## `fixed`

requested `ntrain` をそのまま使用します。ただし4 permutationのうち1つでも共通context budgetを超える場合はエラーにします。

通常のOpenAI / AIkenGPT比較では `reduce` を推奨します。

AIkenGPT backend側にもモデル固有の最終contextチェックを残してあります。

---

# 6. OpenAIモデルを評価する

OpenAIでは、主に次の2つを変更します。

- `--model`: 評価するモデル
- `--data_dir`: 評価する問題セット

正式なentry pointは次です。

```text
evaluate_mmlu_openai_permutation.py
```

## 6.1 リポジトリをclone

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

## 6.2 仮想環境を作る

### Windows + Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

有効になると、shellの先頭などに `(.venv)` と表示されます。

## 6.3 必要なpackageを入れる

リポジトリにrequirementsファイルがある場合は、それを優先してください。

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

requirementsファイルを使わない場合、OpenAI評価の主な依存packageは次です。

```bash
python -m pip install numpy pandas tiktoken openai
```

テストも実行する場合:

```bash
python -m pip install pytest
```

## 6.4 API keyを設定

API keyをコードやGit管理ファイルへ直接書かないでください。

### Bash / Git Bash

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

### PowerShell

```powershell
$env:OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

確認:

```bash
python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

`True` と表示されれば環境変数から読み取れています。

## 6.5 まずdry runする

APIを呼ぶ前に、10問だけ選び、manifestとpromptを生成して確認します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir "./datasets/mmlu" \
  --ntrain 5 \
  --sample_frac 0.1 \
  --seed 42 \
  --limit 10 \
  --context_policy reduce \
  --context_tokenizer gpt2 \
  --max_context_length 2048 \
  --output_dir "./results/openai_trial" \
  --dry_run
```

`--dry_run` ではOpenAI APIを呼びません。

生成されたmanifestと `prompts.jsonl` を開き、次を確認してください。

- 意図した問題が選ばれているか
- 選択肢A-Dが正しく入っているか
- few-shot promptが期待どおりか
- 1問につき4 permutationが生成されているか
- `effective_ntrain` が妥当か

## 6.6 10問だけ実際に評価する

確認できたら `--dry_run` を外します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir "./datasets/mmlu" \
  --ntrain 5 \
  --sample_frac 0.1 \
  --seed 42 \
  --limit 10 \
  --context_policy reduce \
  --context_tokenizer gpt2 \
  --max_context_length 2048 \
  --output_dir "./results/openai_trial"
```

API利用量・rate limitに注意してください。まず少数問で正常終了することを確認してから問題数を増やしてください。

## 6.7 モデルを変える

`--model` だけ変更します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model <MODEL_NAME> \
  --data_dir "./datasets/mmlu" \
  --limit 10
```

使用するモデル・endpointがlogprobsを返せる必要があります。

デフォルトは `--api_mode completions` です。必要な場合は、対応モデルに合わせて次を指定できます。

```bash
--api_mode chat
```

ただしchat modeでは、同じuser-visible promptでもAPI側のchat formattingが加わります。AIkenGPTとの入力条件を最も単純に比較したい場合は、利用可能であればcompletions modeを使用してください。

## 6.8 問題セットを変える

`--data_dir` を別のMMLU-style datasetへ変更します。

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir "./datasets/my_questions" \
  --limit 10
```

コード側を変更する必要はありません。

---

# 7. OpenAI CLIの主要オプション

| option | 意味 | 標準的な値 |
|---|---|---|
| `--model` | OpenAIモデル名 | `gpt-4o-mini` |
| `--data_dir` | `dev/`, `test/` を含むdataset root | 必須 |
| `--subject` | 1 subjectだけ評価 | 未指定なら全subject |
| `--ntrain` | requested few-shot数 | `5` |
| `--sample_frac` | test全体からsamplingする割合 | `0.1` |
| `--seed` | 問題sampling seed | `42` |
| `--limit` | sampling後に評価する最大問題数 | `0` = 制限なし |
| `--manifest` | 既存manifestを再利用 | 未指定なら生成 |
| `--workers` | OpenAI request worker数 | `10` |
| `--output_dir` | 結果保存先 | CLI設定参照 |
| `--context_policy` | `reduce` / `fixed` | `reduce` |
| `--context_tokenizer` | 共通context判定tokenizer | `gpt2` |
| `--max_context_length` | 共通context budget | `2048` |
| `--api_mode` | `completions` / `chat` | `completions` |
| `--dry_run` | APIを呼ばずmanifest/prompt生成 | off |

## `sample_frac` と `limit` の違い

この2つは混同しやすいので注意してください。

- `sample_frac`: test dataset全体から何割をsampling候補として選ぶか
- `limit`: sampling後の問題列から何問まで実際に使うか

例えば:

```text
sample_frac = 0.1
limit = 100
```

なら、まず全test問題の10%をseed固定でsamplingし、その中から最大100問を使います。

### 全問題を評価したい場合

```bash
--sample_frac 1.0 --limit 0
```

### 特定subjectの全問題

```bash
--subject abstract_algebra --sample_frac 1.0 --limit 0
```

---

# 8. AIkenGPTの評価

AIkenGPTはローカルPCのGPUではなく、Google ColabのGPUを利用して評価することを想定しています。

本プロジェクトでは、**VS Code上でNotebookを開き、実行カーネルとしてGoogle Colabを利用する**運用を推奨します。

### 1. リポジトリをローカルにcloneする

まず、ローカルPC上で本リポジトリをcloneします。

```bash
git clone https://github.com/HayatoHongo/AIkenSGTv1.git
cd AIkenSGTv1
git switch tayama
```

現在、共有MMLU評価pipelineは `tayama` ブランチで管理しています。

MMLUの参照リポジトリも、同じ親ディレクトリにcloneしておくことを推奨します。

```bash
cd ..
git clone https://github.com/hendrycks/test.git mmlu-reference
```

ローカルでは、例えば次の構成になります。

```text
workspace/
├── AIkenSGTv1/
└── mmlu-reference/
```

なお、MMLUの `dev/`・`test/` データは `hendrycks/test` のcloneだけでは作成されません。本リポジトリの `download_mmlu_hf.py` を利用して、Hugging Faceの `cais/mmlu` から評価用CSVを生成します。

### 2. VS CodeでNotebookを開く

AIkenGPTの評価には以下のNotebookを使用します。

通常のA/B/C/D letter scoring：

```text
evaluate_aikengpt_mmlu.ipynb
```

選択肢本文のlikelihoodを用いる評価：

```text
evaluate_aikengpt_mmlu_text.ipynb
```

VS Codeで目的のNotebookを開きます。

### 3. Google ColabのGPUカーネルへ接続する

VS CodeからGoogle Colabのruntimeを実行カーネルとして選択し、GPU runtimeへ接続します。

Notebookファイル自体はローカルのVS Code上で編集しますが、**セル内のPythonコードはColab側の環境で実行されます。**

そのため、ローカルPCにcloneしたリポジトリは、ColabのPythonから直接参照できません。

Notebookのsetupセルでは、Colab runtimeの一時領域 `/content/` に必要なリポジトリをcloneします。

```python
from pathlib import Path

PROJECT_DIR = Path("/content/AIkenSGTv1_New")
MMLU_ROOT = Path("/content/mmlu-reference")
DATA_DIR = MMLU_ROOT / "data"

# Evaluation repository
if not (PROJECT_DIR / "mmlu_eval").is_dir():
    !rm -rf /content/AIkenSGTv1_New
    !git clone https://github.com/HayatoHongo/AIkenSGTv1.git /content/AIkenSGTv1_New
    !cd /content/AIkenSGTv1_New && git switch tayama

# MMLU reference repository
if not MMLU_ROOT.exists():
    !git clone https://github.com/hendrycks/test.git /content/mmlu-reference

# Create MMLU dev/test CSV files if they do not exist
if not (DATA_DIR / "dev").is_dir() or not (DATA_DIR / "test").is_dir():
    !pip install -q datasets pandas
    !cd /content/mmlu-reference && python /content/AIkenSGTv1_New/download_mmlu_hf.py
```

Colab runtimeは一時的な環境なので、runtimeを新しくすると `/content/` 以下のcloneや生成済みMMLUデータは消える場合があります。

その場合でも、Notebookを上から実行すれば必要なものが再度準備されます。

### 4. Google Driveをマウントする

AIkenGPTのcheckpointや評価結果など、runtime終了後も保持したいファイルにはGoogle Driveを利用します。

```python
from google.colab import drive
drive.mount("/content/drive")
```

役割分担は次のようになります。

```text
ローカルPC / VS Code
    └── Notebookの編集・Git操作

GitHub
    └── 評価コードの共有・バージョン管理

Colab /content/
    ├── AIkenSGTv1_New/     # 実行用の一時clone
    └── mmlu-reference/     # 実行用のMMLUデータ

Google Drive
    ├── AIkenGPT checkpoint
    └── 評価結果
```

Google Driveにリポジトリそのものをコピーしておく必要はありません。

### 5. 評価条件を設定する

通常変更する設定はNotebook冒頭の設定セルにまとめています。

例：

```python
OUTPUT_DIR = Path("/content/drive/MyDrive/aikengpt_results")

SUBJECT = None
NTRAIN = 0
SAMPLE_FRAC = 0.10
SEED = 42
LIMIT = 10
MANIFEST_PATH = None
```

主な設定：

| 設定              | 意味                                     |
| --------------- | -------------------------------------- |
| `SUBJECT`       | 特定subjectだけ評価する場合に指定。`None` なら全subject |
| `NTRAIN`        | few-shot例の数。`0` はzero-shot             |
| `SAMPLE_FRAC`   | 全問題から抽出する割合                            |
| `SEED`          | sampling seed                          |
| `LIMIT`         | 実際に評価する最大問題数。`0` なら抽出された問題をすべて使用       |
| `MANIFEST_PATH` | 既存manifestを使って同じ問題集合を再評価する場合に指定        |

最初の動作確認では、

```python
NTRAIN = 0
LIMIT = 10
```

程度の小規模評価を推奨します。

### 6. scoring方式

#### 通常版

`evaluate_aikengpt_mmlu.ipynb` では、

```python
SCORING_METHOD = "letter"
```

として、prompt末尾に続く `" A"`, `" B"`, `" C"`, `" D"` のlogitを比較します。

#### choice-text版

`evaluate_aikengpt_mmlu_text.ipynb` では、

```python
SCORING_METHOD = "choice_text"
TEXT_REDUCTION = "mean"
```

などとして、各選択肢本文のteacher-forced likelihoodを比較します。

`TEXT_REDUCTION` は次の2種類です。

```text
sum
```

候補tokenのlog probabilityを合計します。長い選択肢ほど不利になりやすい方式です。

```text
mean
```

候補token数で平均します。選択肢長に対するlength normalizationを行います。

### 7. context length

AIkenGPTの最大context長に合わせて、OpenAIとAIkenGPTの比較では共通して次の設定を使用します。

```python
CONTEXT_POLICY = "reduce"
CONTEXT_TOKENIZER = "gpt2"
MAX_CONTEXT_LENGTH = 2048
```

requested `NTRAIN` のpromptが2048 tokensを超える場合、4つのpermutationすべてが収まるまでfew-shot例を減らします。

実際に使用されたshot数は、

```text
effective_ntrain
```

として結果に記録されます。

これにより、OpenAIだけ長いpromptを使用し、AIkenGPTだけshot数を減らす、といった評価条件の不一致を防ぎます。

### 8. 実行する

設定後、Notebookを上から順番に実行します。

大まかな流れは次のとおりです。

```text
repository / dataset setup
        ↓
Google Drive mount
        ↓
package installation
        ↓
AIkenGPT checkpoint load
        ↓
manifest / prompt generation
        ↓
MMLU evaluation
        ↓
result output
```

初回は10問程度で最後まで実行できることを確認してから、評価対象を増やすことを推奨します。

### 9. GPUメモリ不足が発生した場合

AIkenGPTのモデル読み込み中などに、

```text
OutOfMemoryError: CUDA out of memory
```

が発生した場合、以前のモデルやTensorがGPUメモリに残っている可能性があります。

まず、

```python
!nvidia-smi
```

でGPU使用量を確認してください。

ColabのL4 GPUで、モデル実行前にもかかわらずPythonプロセスが大量のVRAMを使用している場合は、カーネルを再起動してから再実行します。

クリーンな状態では、例えば次のようにGPUメモリがほぼ空になります。

```text
Memory-Usage: 数MiB / 約23GiB
No running processes found
```

カーネル再起動後は、Notebookを上から順番に実行し直してください。


# 9. AIkenGPT letter scoring

`evaluate_aikengpt_mmlu.ipynb` は、prompt末尾の次tokenとして

```text
" A"
" B"
" C"
" D"
```

のlogitを直接取得します。

各ラベルがtokenizer上で1 tokenであることをbackendが検査します。

生成 (`generate`) やsamplingは使用せず、4候補のlogitを直接比較します。

OpenAIのletter scoringと比較する際の基本モードです。

---

# 10. AIkenGPT choice-text likelihood scoring

`evaluate_aikengpt_mmlu_text.ipynb` は、A/B/C/Dという文字ではなく、**選択肢本文そのものの尤度**をteacher forcingで評価します。

User settingsで次を指定します。

```python
TEXT_REDUCTION = "mean"
```

選択肢は次の2つです。

### `sum`

candidate tokenのlog probabilityを合計します。

```text
score = Σ log P(token_i | prompt, previous candidate tokens)
```

候補が長いほど負のlog probabilityを多く足すため、長い選択肢が不利になりやすい性質があります。

### `mean`

tokenごとのlog probabilityを平均します。

```text
score = mean(log P(token_i | ...))
```

candidate lengthで正規化したscoreです。

`sum` と `mean` は異なる評価方式なので、結果を保存・比較するときはどちらを使ったか必ず明記してください。

letter scoringとchoice-text scoringも異なる評価方式です。直接同一のscoreとして扱わないでください。

---

# 11. Manifestを使って同じ問題を評価する

複数モデルを比較する場合は、**同じmanifestを使うことを強く推奨します。**

manifestには、少なくとも各問題のsubjectとtest index、および再現性確認用の情報が保存されます。

既存manifestを指定した場合、そのmanifestが評価対象として優先されます。

## OpenAI

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir "./datasets/mmlu" \
  --manifest "./manifests/mmlu_trial.manifest.csv"
```

## AIkenGPT

Notebookで:

```python
MANIFEST_PATH = Path("/content/drive/MyDrive/manifests/mmlu_trial.manifest.csv")
```

同じmanifestを使う場合は、datasetとEvalConfigも同じにしてください。

特に次を揃えます。

```text
NTRAIN
CONTEXT_POLICY
CONTEXT_TOKENIZER
MAX_CONTEXT_LENGTH
PERMUTATION_COUNT
```

---

# 12. OpenAIとAIkenGPTのpromptが同じか確認する

各runでは、モデルに渡る前のcaseを `.prompts.jsonl` に保存します。

OpenAI runとAIkenGPT runのtraceを比較するには:

```bash
python -m mmlu_eval.compare \
  path/to/openai.prompts.jsonl \
  path/to/aikengpt.prompts.jsonl
```

完全一致すれば、例えば次のように表示されます。

```text
Identical: 40 permutation cases
```

`compare.py` はwhitespaceを含むcase内容の差を検出します。

モデル比較を行う場合、結果scoreを見る前にこの確認を行うと安全です。

---

# 13. 出力ファイル

通常のrunでは、結果CSVのほかに再現性確認用ファイルが生成されます。

例えば結果本体が

```text
results_xxx.csv
```

なら、同じstemで次のようなファイルが作られます。

| ファイル | 内容 |
|---|---|
| `results_xxx.csv` | 各問題のprediction・score・probability |
| `results_xxx.manifest.csv` | 評価した問題集合と再現性情報 |
| `results_xxx.prompts.jsonl` | backend直前の全prompt/case |
| `results_xxx.run.json` | pipeline、package、model等のrun metadata |
| `results_xxx.subjects.csv` | subjectごとの集計 |
| `results_xxx.overall.json` | 全問題の集計 |

AIkenGPT NotebookではEvaluation前にもpreflight用のmanifest・prompt traceを保存します。

---

# 14. 結果の主な見方

## Accuracy

主に次の2つを確認します。

```text
baseline_accuracy
```

元の選択肢順だけで評価したaccuracyです。

```text
debiased_accuracy
```

4 permutationをsemantic choiceへ戻して平均した後のaccuracyです。

## Prediction change

位置バイアス補正によってpredictionが変わった影響を見るため、summaryには次の情報も含まれます。

- wrong → correct
- correct → wrong
- baselineとdebiasedのaccuracy差

## `effective_ntrain`

各問題で実際に使用されたfew-shot数です。

requested `NTRAIN=5` でもcontext budgetのため4以下になることがあります。

---

# 15. OpenAI top-logprobsに関する注意

OpenAI backendはA/B/C/D候補のscoreをAPIのtop logprobsから取得します。

候補がtop-kに現れない場合があるため、結果metadataには次の情報を保存します。

```text
missing_count
error_bound
fifth_logprob
```

4候補すべてが取得できない場合は、そのまま誤答として処理せずエラーにします。

OpenAIモデルをsanity checkとして使用する場合、accuracyだけでなく `missing_count` や `error_bound` も確認してください。

特に異なるOpenAIモデルへ差し替える場合、logprobの提供方法・tokenization・endpoint仕様が同じとは限りません。

---

# 16. Resume

評価は1問単位で結果を保存します。

同じoutput pathで再実行した場合、run metadata・manifest・model・settingsが一致していれば、完了済み問題を再利用して途中から再開できます。

設定やモデルが異なる既存結果へ誤って追記しようとするとエラーになります。

モデル・dataset・設定を変更した場合は、新しいoutput pathを使用してください。

---

# 17. テスト

共通MMLU評価pipelineのテストだけを実行する場合:

```bash
python -m pytest tests/test_mmlu.py -q
```

PyTorchがインストールされていないlocal環境では、local AIkenGPT backendを使うテストがskipされることがあります。これは想定された挙動です。

重要なのは `failed` / `error` がないことです。

PyTorchが利用できるColab環境ではlocal backendのテストも実行してください。

---

# 18. 推奨する試験運用手順

README自体とコードが一致していることを確認するため、初回は次の順番で実際に試してください。

## Phase A: OpenAI dry run

```bash
python evaluate_mmlu_openai_permutation.py \
  --model gpt-4o-mini \
  --data_dir "<DATA_DIR>" \
  --ntrain 5 \
  --sample_frac 0.1 \
  --seed 42 \
  --limit 10 \
  --dry_run
```

確認:

- manifestが作られる
- 10 questions / 40 permutation promptsになる
- promptが正しい
- `effective_ntrain` が妥当

## Phase B: OpenAI 10問

同じ条件から `--dry_run` を外して実行します。

確認:

- result CSVが生成される
- baseline / debiased predictionが入る
- overall summaryが生成される
- missing_count / error_boundに異常がない

## Phase C: AIkenGPT letter 10問

OpenAIで作成したmanifestをNotebookの `MANIFEST_PATH` に指定します。

```python
LIMIT = 10
MANIFEST_PATH = Path("<OPENAI_MANIFEST_PATH>")
```

Run allします。

確認:

- GPUでmodelがloadされる
- checkpoint SHA256が表示される
- 10 questions / 40 prompts
- 結果CSVが生成される

## Phase D: prompt一致

OpenAIとAIkenGPTの `.prompts.jsonl` を比較します。

```bash
python -m mmlu_eval.compare \
  <OPENAI_PROMPTS_JSONL> \
  <AIKENGPT_PROMPTS_JSONL>
```

`Identical` が出ることを確認します。

## Phase E: AIkenGPT choice-text

同じmanifestで `evaluate_aikengpt_mmlu_text.ipynb` を実行します。

まず:

```python
TEXT_REDUCTION = "mean"
```

で試し、必要に応じて `sum` も別runとして評価します。

## Phase F: 本評価

小規模試験が正常なら、目的に応じて設定を変更します。

### 10% sample全体

```text
SAMPLE_FRAC = 0.10
LIMIT = 0
```

### 全test問題

```text
SAMPLE_FRAC = 1.0
LIMIT = 0
```

モデル比較では同じmanifestを再利用してください。

---

# 19. よくある問題

## `pytest: command not found`

```bash
python -m pip install pytest
python -m pytest tests/test_mmlu.py -q
```

`python -m pytest` を使うと、現在有効なPython環境のpytestを明示的に使えます。

## OpenAI API keyがない

```text
openai.OpenAIError: Missing credentials
```

環境変数 `OPENAI_API_KEY` を設定してください。

## `PROJECT_DIR must point to the repository containing mmlu_eval/`

Colab Notebookの `PROJECT_DIR` がリポジトリrootを指していません。

```python
PROJECT_DIR = Path("/content/drive/MyDrive/<YOUR_REPOSITORY_DIRECTORY>")
```

を確認してください。

## `DATA_DIR must contain dev/ and test/`

`DATA_DIR` がdataset rootではない可能性があります。

正しい構成:

```text
DATA_DIR/
├── dev/
└── test/
```

## `Prompt exceeds context`

標準比較では `CONTEXT_POLICY="reduce"` を使用してください。

0-shotでもcontextを超える問題はそのまま評価できません。

## 同じoutput pathなのにresumeできない

model、dataset、manifest、source、settingsなどが前回runと異なる可能性があります。

別のoutput pathを使用してください。

---

# 20. 比較実験で固定すべきもの

モデル性能を比較したい場合は、少なくとも次を固定してください。

```text
dataset
manifest
NTRAIN
CONTEXT_POLICY
CONTEXT_TOKENIZER
MAX_CONTEXT_LENGTH
PERMUTATION_COUNT
prompt pipeline
scoring method
```

そのうえで、比較したい変数だけを変更します。

### OpenAIモデル同士の比較

```text
変える: --model
固定: dataset + manifest + EvalConfig
```

### OpenAIとAIkenGPT letter scoringの比較

```text
変える: backend / model
固定: dataset + manifest + EvalConfig + prompt + aggregation
```

### AIkenGPTのletterとchoice-textの比較

```text
変える: scoring_method
固定: dataset + manifest + EvalConfig
```

ただしletterとchoice-textはscoreの定義そのものが異なるため、同一指標として解釈しないでください。

---

# 21. 実験結果を共有するときに残すもの

再現性のため、最低限次を残してください。

- 使用したmodel名 / checkpoint
- checkpoint SHA256（AIkenGPT）
- dataset
- manifest
- `NTRAIN`
- `effective_ntrain` の分布
- `SAMPLE_FRAC`
- `SEED`
- context settings
- scoring method
- result CSV
- `.overall.json`
- `.run.json`
- `.prompts.jsonl`

OpenAIの場合は、可能ならresolved modelやsystem fingerprintなどrun metadataに記録された情報も保持してください。

API keyや個人PCの絶対pathはGitHubへcommitしないでください。

---

# 22. この評価pipelineの目的

この実装では、OpenAIとAIkenGPTを比較するときに、モデル以外の違いを可能な限り共通化することを重視しています。

つまり、

```text
同じ問題
同じfew-shot examples
同じeffective_ntrain
同じprompt文字列
同じ4 permutations
同じsemantic mapping
同じaggregation
        ↓
モデル固有のscoringだけを差し替える
```

という構造です。

特にOpenAIモデルで評価pipelineをsanity checkした後、同じmanifest・同じpre-backend inputをAIkenGPTへ与えることで、評価実装の差ではなくモデルの差を検討しやすくしています。

---

## 最初のおすすめ設定

迷った場合は、まず次で10問だけ試してください。

```text
NTRAIN = 5
SAMPLE_FRAC = 0.10
SEED = 42
LIMIT = 10
CONTEXT_POLICY = reduce
CONTEXT_TOKENIZER = gpt2
MAX_CONTEXT_LENGTH = 2048
PERMUTATION_COUNT = 4
```

これが正常に動いてから、`LIMIT=0` や `SAMPLE_FRAC=1.0` に広げてください。
