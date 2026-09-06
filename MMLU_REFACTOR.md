# MMLU共通評価pipeline

## 変更前に確認した内容

対象の実ファイルは evaluate_mmlu_permutation.py と2つのNotebook。
依頼文の evaluate_mmlu_openai_permutation.py は存在しなかったため、今回aliasを追加した。
変更前の3ファイルは legacy/ にそのまま保存した。既存の結果CSVには変更を加えていない。

| 項目 | OpenAI Python | 通常Notebook | text Notebook（実際の内容） |
|---|---|---|---|
| データ | headerなし6列CSV、test/dev | 同じCSV、0-shotはtestのみ | 同じCSV、testのみ |
| subjects | testファイル名をsort、subject指定可 | manifestのsubject | abstract_algebra試行のみ。全体runなし |
| sample | 全subject結合、default_rng(42)、round(N*.1)、非復元抽出、最後にlimit | OpenAIの2列manifest、行順そのまま | 先頭10問の試行 |
| train/dev | dev先頭ntrain、train未使用 | 5-shotはdev先頭、train未使用 | dev few-shot実装なし |
| ntrain | default 5、-1は全dev | 前半/後半0-shotと末尾5-shotが別々 | 0-shot |
| prompt | subject導入文、質問、改行A.〜D.、改行Answer: | 前半はQuestion:付き導入文なし、後半はPython同一 | 同じ前半・後半0-shot定義 |
| few-shot | 回答後に改行2つ、順序固定 | 同じ。長い問題はshot削減 | なし |
| context | ローカルの長さチェックなし | gpt2/2048、5→0で4配置すべて入る最大shot | gpt2/2048、超過は例外 |
| permutation | 4 cyclic、identityから開始、乱数なし | 同じ | 同じ |
| score | 空白付き文字token、top5欠落=-100、4択softmax | 空白付き文字tokenの全logitsから4択softmax | **本文尤度ではなく同じ文字token scoring** |
| baseline/debiased | identity argmax / 元のchoiceへ戻した確率の平均argmax | 同じ | 同じ |
| tie | A→Dの最初を選ぶ | 同じ | 同じ |
| resume | subject/indexだけで再開、10問ごと保存 | subject/indexのみ、1問ごと追記 | 全体resumeなし |
| CSV | test_index、各perm確率、token/top5診断 | 試行はquestion_index、後半test_index、5-shotにeffective_ntrain | 試行CSVのみ |
| manifest | subject,test_indexを生成（読み込みなし） | 2列manifest読み込み | なし |
| 集計 | micro accuracy、label recall/RStd | micro、位置確率平均、subject別、correct/wrong遷移 | 試行accuracyのみ |

モデルはHF形式のAutoModelではなく、Notebookに定義された独自GPTと
HayatoHongo/AIkenSGTv1 の model_clean.safetensors、tiktoken gpt2、FP32 GPUだった。
このcheckpointが「AIken-GPT-Base-2.5B」に該当するかは、ファイル名だけからは検証できない。
実行記録にはcheckpointのSHA256も保存する。モデル本体の構造は変更していない。

## ファイルの役割

- mmlu_eval/core.py: CSV読み込み、sampling、prompt、few-shot、4 cyclic、manifest検証、
  backend前Case、softmax、semantic remapping、平均、判定、resume、共通CSV/集計。
- mmlu_eval/backends/openai_backend.py: key、SDK、API request、retry、rate limit、
  concurrency、top5候補抽出と診断。SDK/client生成は遅延しimport時にkey不要。
- mmlu_eval/backends/aikengpt_backend.py: 独自GPT/HF forward、letter logits、
  choice_text sum/mean、候補batch、checkpoint load。
- mmlu_eval/backends/aikengpt_model.py: 通常NotebookのGPT/Configを抽出した元のモデル定義。
- mmlu_eval/cli.py: 旧CLI引数と追加の明示設定、APIなしdry-run。
- evaluate_mmlu_permutation.py / evaluate_mmlu_openai_permutation.py: 共通CLIの入口。
- 2つのNotebook: install、Drive、パス/設定、GPU、checkpoint/tokenizer load、Evaluator呼び出し。
- mmlu_eval/compare.py: 2つのprompt JSONLの全Caseを厳密比較。
- tests/test_mmlu.py: 旧関数への回帰、10問×4配置一致、mapping、resume、manifest、
  whitespace、mock API、PyTorchによる小型モデルscoringテスト。
- requirements-mmlu.txt: 必要なライブラリ。各runの実際のversionはrun.jsonに記録。
- legacy/: 調査時の原本。ここを新実験の入口にしない。

## 共通条件と意図的な差

3方式すべて同じMMLUEvaluatorを通る。backendのscore_choices(prompt, displayed_choices)は
表示位置順の4つの未正規化log scoreを返す。softmaxは共通側で行う。
baselineはidentity、debiasedは各配置を元choiceへ戻した4確率の算術平均。
位置確率は全問題×4配置の表示位置確率の平均。overallは問題数で重み付けされたmicro accuracy。
subject別も同じ関数で集計する。正誤はpred == labelで算出し、CSV文字列"False"のbool変換をしない。

letterは空白付きA-Dの次token logits。APIではtop5欠落=-100という旧近似を維持。
choice_textは prompt token列 + encode(" " + 候補本文) をteacher forcingし、
**候補部分のみ**のlog probabilityをsumまたはmeanにする。EOSは加えない。
sumは長さの影響を受ける。meanはtoken数で除した長さ正規化。
元text Notebookに本文方式は存在しなかったため、既存挙動の保存とは主張しない。
ユーザーの選択に従い両方式を実装し、text_reductionの明示指定を必須とした。
letterと本文尤度のスコア差はモデル差ではなく方式差を含む。

context_policy=reduce を比較実験の標準とし、requested ntrainから0まで減らして、
4配置すべてが共通budgetに収まる最大値をeffective_ntrainにする。
context_policy=fixed はrequested ntrainだけを使うが、上限を無視せず、
1配置でも超過すれば明示的にエラーにする。
両モデルで同じcontext_tokenizer=gpt2とmax_context_length=2048を使う。
モデル独自にshot削減/切り捨て/問題スキップは行わず、超過はエラーにする。
本文尤度は候補分もcontextを使うため、超過する場合は共通設定のshot数/予算を減らし、
**全方式のmanifestを作り直して再実行**する。本文backendだけを変更して続行しない。

## OpenAI APIの注意点（設計上の限界）

現状: 旧コードはgpt-4o-miniを指定しながらlegacy Completionsを呼ぶ。
問題: chatモデルの利用にはendpoint整合性の確認が必要で、失敗を黙ってfallbackしてはいけない。
対応: --api_mode completions（旧標準）と --api_mode chat を明示的に分ける。
gpt-4o-miniの実行例はchatを指定する。chatのuser.contentは共通promptそのものだが、
サーバー側chat wrapperとtokenizerはローカル生promptと同一にはできない。

現状: 旧APIは" A"等の空白付きtokenしか採用しない。
問題: chat応答先頭では"A"等が使われ、4候補が欠落しうる。
対応: --answer_prefix bare を明示選択できる。空白/非空白を黙って合算しない。
top5打ち切り近似も残り、error_boundを保存する。all-missingは失敗として停止。
将来の推奨: 同じraw completion条件で4候補の正確な尤度を得られるモデル/endpointを用意し、
別実験として比較する。今回その方式へ勝手に変更していない。

公式資料:
- https://developers.openai.com/api/docs/models/gpt-4o-mini
- https://developers.openai.com/api/reference/cli/resources/chat

従って保証するのは「同じデータ・few-shot・permutation・prompt文字列・集計pipeline」。
chat wrapper、tokenizer、top5近似まで完全一致する、あるいは既知のMMLU公表値と
一致するとは主張しない。10% sampling/shot削減/permutation方式は公表評価条件とも異なりうる。
固定model snapshotの指定を推奨する。aliasを使う場合はAPIが返したmodel/fingerprintも保存する。

## 実行方法

このディレクトリをカレントにし、Python 3.10以上を使う。

~~~sh
pip install -r requirements-mmlu.txt
python -m unittest discover -s tests -v
~~~

まずAPIを使わず10問の共通manifestとpromptを作る（固定5-shot例）。

~~~sh
python evaluate_mmlu_openai_permutation.py --data_dir "../mmlu-reference/data" --ntrain 5 --limit 10 --dry_run --output_dir runs/prepare
~~~

生成ファイル:
runs/prepare/results_gpt-4o-mini_5shot_seed42_frac0p1_all.manifest.csv
同名の.prompts.jsonlには10問×4配置の完全な文字列が入る。

API keyはOPENAI_API_KEY環境変数で指定。以下はchat/非空白回答tokenを明示した別API設定。

~~~sh
python evaluate_mmlu_openai_permutation.py --data_dir "../mmlu-reference/data" --ntrain 5 --limit 10 --manifest runs/prepare/results_gpt-4o-mini_5shot_seed42_frac0p1_all.manifest.csv --api_mode chat --answer_prefix bare --output_dir runs/openai
~~~

Colabではコピー側フォルダ（mmlu_evalを含む）、同じMMLUデータ、同じmanifestをDriveへ置く。
NotebookのPROJECT_DIR、DATA_DIR、MANIFEST_PATHを設定する。
NTRAIN=5、SAMPLE_FRAC=.1、SEED=42、LIMIT=10等をmanifest生成時と揃える。
全文比較には両方のprompts.jsonlを使う。

~~~sh
python -m mmlu_eval.compare runs/openai/results_gpt-4o-mini_5shot_seed42_frac0p1_all.prompts.jsonl path/to/aikengpt.prompts.jsonl
~~~

0-shotは両側ntrain=0で別manifest/出力を作る。
標準の可変shotは両側context_policy=reduceを使う。固定shotとの比較時は別manifestを作る。
全サンプル実験はlimit=0で新manifestを作成する。
既存manifestを渡すとその全行が正本であり、limit/subjectで黙って行を落とさない。
enriched manifestでは設定hashも比較するため、実行時も同じ設定を指定する。
legacyの2列manifestは読めるが元データhashはないため、読み込んだ時点から検証情報を追加する。
入力manifestは上書きせず各runの.manifest.csvを保存する。

Notebookの主要実行は以下。

~~~python
backend = AIkenGPTBackend(model, tokenizer, model_identifier=MODEL_ID, ...)
evaluator = MMLUEvaluator(DATA_DIR, eval_config)
results = evaluator.run(backend, OUTPUT_PATH, manifest_path=MANIFEST_PATH)
~~~

## 出力とresume

結果CSVはsubject/test_index/question/A-D/label、baseline/debiased_pred/correct、
baseline/debiased_A-D_prob、perm0-3_A-D_score/prob、
effective_ntrain/model_identifier/scoring_method/permutations/cases_hash、
sample_order/run_id、token/top5診断、permutation別backend metadataを保存。
モデル固有情報も列名は共通で、非該当は空欄。
.manifest.csvはsubject/test_index/sample_order/dataset_hash/config_hash/
pipeline_version/effective_ntrain/cases_hash。
.run.jsonは評価設定、backend設定、manifest hash、model/checkpoint識別、
runtime versionを保存する。pathの違いだけでdataset同一性を崩さない。
.prompts.jsonlはquestion/choices/correct_answer/few_shot_examples/
effective_ntrain/permutation/promptを全件保存し、backend直前と同じCase生成を使う。
.subjects.csvと.overall.jsonも共通schema。
1問完了ごとに一時ファイルからatomic replaceし、manifest順で保存する。
同じ出力への並行実行はしないこと。

条件/model/manifestが違うresume、重複行、dataset変更はエラー。
旧CSVには条件の証拠がないため新runへのresumeはしない（既存ファイルは保持）。
旧CLI引数は維持。新default output_dirはresults_permutation_shared。
旧Notebookセルの関数名は共通Evaluatorへ置換した。
workersはbackend内の4配置の並行request数を制御する。
batch_sizeはローカル本文候補のbatch数を制御し、letterは1 prompt/forwardである。

## 評価値が変わりうる点

1. 前半のQuestion:付き試行を正式評価から除外したため、その試行結果とはpromptが違う。
2. 固定5-shotと旧Notebookの可変shotは違う。必要なら共通reduceを明示選択する。
3. chat/answer_prefix変更はAPI条件の変更。既存completionsとは別run。
4. 本文sum/meanは新しい方式であり、文字評価とは別実験。
5. ローカルfloat32 softmaxから共通NumPy float64 softmaxに統一したため、
   極端な同点付近の丸めでargmaxが変わる可能性がある。
6. デバイス/dtype/batch/ライブラリ、API alias更新・非決定性は残る。
7. 不適切なresume混在が拒否されるため、以前の混在結果とは集計が違いうる。

再現性の中心は乱数seedだけではなく、同じmanifest・設定・データhash・Case hashの一致。
実モデル/APIでの精度測定は別途必要であり、offlineテストだけでモデルのMMLU妥当性を認定しない。

## GitHub利用者向けinterfaceの整理

正式なOpenAI entry pointを `evaluate_mmlu_openai_permutation.py` に一本化してREADMEに明記した。
`evaluate_mmlu_permutation.py` は共通pipelineを呼ぶ非推奨wrapper、
`evaluate_openai.py` は過去実験再現用のlegacy implementationとして警告を表示する。

2つのAIkenGPT Notebookは、Settings、Environment setup、Model loading、
Dataset / evaluator setup、Preflight / manifest、Evaluation、Resultsの順に整理した。
通常変更するdataset・sampling設定は先頭のUser settingsに集約し、
checkpoint等はProject settingsへ分離した。`MANIFEST_PATH=None` の場合も、
preflightで生成・保存したmanifestを評価本体が再利用する。

このinterface整理では共通評価アルゴリズム、prompt、sampling、permutation、mapping、
baseline/debiased定義、集計、結果schemaを変更していない。

その後、比較条件を揃える修正としてcontext判定だけをmmlu-shared-v2へ更新した。
`reduce / gpt2 / 2048` を標準とし、`fixed` でも全4配置の上限を検査する。
sampling、prompt文面、permutation、scoring、mapping、aggregationには変更していない。

## この環境での検証結果

- 25件のunit/integrationテスト: 全件成功、skipなし。
- tests/verify_real_data.py: 実MMLU 57 subjectsからseed42で抽出した10問を、
  mock OpenAI / 小型local letter / 小型local text sum / 小型local text meanの4経路で実行。
  10問×4配置のCase全フィールドとbackend直前引数が完全一致。
  各経路のresult schemaも一致。verification/real_data/report.json と4つのprompt JSONLを保存。
- 旧OpenAI関数との0/5-shot prompt、samplingの回帰テスト成功。
- 本文sum/meanは既知のtoken log probabilityによる期待値と一致。
  batch size 1と4が一致。中断後resumeで未完了問だけ再実行されることも確認。
- 両Notebookのコードセルを構文検証。古い実行outputはlegacyに保存し、新Notebookは未実行状態。
- 変更禁止のAIkenSGTv1は131ファイルのSHA256が作業前と一致（.git管理内部を除く）。
- **gpt-4o-mini実APIとAIkenGPT実checkpointの精度評価は未実行。**
  mock/小型モデルの結果をMMLUモデル性能として扱わないこと。

CPUテスト用PyTorchはコピー側の.test-depsにのみ導入し、.gitignoreで除外した。
通常環境/Colabではrequirementsを導入してunittestを実行できる。
このローカル環境で同じ依存を使う例:

~~~sh
python -B -c "import sys,unittest; sys.path.insert(0,'.test-deps'); unittest.main(module=None,argv=['test','discover','-s','tests','-v'])"
python -B tests/verify_real_data.py --data_dir "../mmlu-reference/data"
~~~

補足: run.jsonには共通コードのsource hashとライブラリversionも含め、
再開時のコード・実行環境変更も検出する。
データに含まれるN/A等は旧pandas読み込みと同じ欠損解釈を保持するため、
promptでは旧挙動同様にnanとなりうる。これを勝手に修正してデータ条件を変えていない。
新規追加の主なファイルとしてtests/verify_real_data.py、verification/検証出力、
tests/original_folder_sha256.jsonを含む。.gitignoreには.test-depsのみ追加した。
