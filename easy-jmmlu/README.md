# Easy JCommonsenseQA（test 30問）

JCommonsenseQAのvalidationから、元の正解indexが4ではない30問をtestに選びました。devはtrainから5問です。

## 変換方法

`convert.py`を実行します。Python標準機能だけを使っています。

1. `source.jsonl`を1行ずつ読む。
2. 正解indexが4の問題を除外する。
3. 残った問題から選択肢4を削除する。
4. 元の選択肢0〜3をそのままA〜Dにし、正解indexもA〜Dに置き換える。

並べ替えやランダム処理はありません。出力はMMLU評価コード用のヘッダーなし6列CSVです。

```sh
python convert.py
```

## フォルダ構成

```text
easy-jmmlu/
├── dev/jcommonsenseqa_dev.csv
└── test/jcommonsenseqa_test.csv
```

AIkenGPT評価Notebookの `DATA_DIR` をこのフォルダにし、`SUBJECT = "jcommonsenseqa"`、`NTRAIN = 5` にします。

## 全体版JSONL

`source_4.jsonl`はJCommonsenseQAのtrainとvalidation全件を4択に変換したファイルです。正解index 4のサンプルを除外し、残りの選択肢4を削除しています。合計8,141件（train 7,220件、validation 921件）です。

## 出典

- [sbintuitions/JCommonsenseQA](https://huggingface.co/datasets/sbintuitions/JCommonsenseQA)
- [yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE)
- ライセンス: CC BY-SA 4.0
- testの30問はvalidation splitから取得

Google Drive: [easy-jmmlu フォルダ](https://drive.google.com/drive/folders/1rHQlrRu--_y9KnyBkqLvc_unW7_y0C8K)
