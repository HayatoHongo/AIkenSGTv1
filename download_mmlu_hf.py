from pathlib import Path

import pandas as pd
from datasets import load_dataset


CHOICES = ["A", "B", "C", "D"]


print("Downloading MMLU from Hugging Face...")

dataset = load_dataset("cais/mmlu", "all")

data_dir = Path("data")
(data_dir / "dev").mkdir(parents=True, exist_ok=True)
(data_dir / "test").mkdir(parents=True, exist_ok=True)

subjects = sorted(set(dataset["test"]["subject"]))

print(f"Found {len(subjects)} subjects.")


for subject in subjects:
    for split in ["dev", "test"]:

        rows = []

        for example in dataset[split]:
            if example["subject"] != subject:
                continue

            row = [
                example["question"],
                *example["choices"],
                CHOICES[example["answer"]],
            ]

            rows.append(row)

        output_path = (
            data_dir
            / split
            / f"{subject}_{split}.csv"
        )

        pd.DataFrame(rows).to_csv(
            output_path,
            index=False,
            header=False,
        )

    print(f"Created: {subject}")


print("Done.")