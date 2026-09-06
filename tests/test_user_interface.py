import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UserInterfaceTests(unittest.TestCase):
    def notebook(self, name):
        return json.loads((ROOT / name).read_text(encoding="utf-8"))

    def test_notebooks_follow_documented_sections_and_are_clean(self):
        expected = [
            "## 1. Settings",
            "## 2. Environment setup",
            "## 3. Model loading",
            "## 4. Dataset / evaluator setup",
            "## 5. Preflight / manifest",
            "## 6. Evaluation",
            "## 7. Results",
        ]
        for name in (
            "evaluate_aikengpt_mmlu.ipynb",
            "evaluate_aikengpt_mmlu_text.ipynb",
        ):
            notebook = self.notebook(name)
            sources = ["".join(cell["source"]) for cell in notebook["cells"]]
            positions = [next(i for i, source in enumerate(sources) if title in source)
                         for title in expected]
            self.assertEqual(positions, sorted(positions))
            self.assertIn("MANIFEST_PATH = None", sources[2])
            self.assertIn("DATA_DIR =", sources[2])
            combined = "\n".join(sources)
            self.assertIn('CONTEXT_POLICY = "reduce"', combined)
            self.assertIn('CONTEXT_TOKENIZER = "gpt2"', combined)
            self.assertIn("MAX_CONTEXT_LENGTH = 2048", combined)
            self.assertLess(positions[0], next(i for i, source in enumerate(sources)
                                              if "%pip install" in source))
            for cell in notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                if not source.lstrip().startswith("%"):
                    ast.parse(source)

    def test_text_reduction_is_explained_and_explicit(self):
        notebook = self.notebook("evaluate_aikengpt_mmlu_text.ipynb")
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn('TEXT_REDUCTION = "mean"', source)
        self.assertIn('"sum"', source)
        self.assertIn("length-normalized", source)
        self.assertIn("text_reduction=TEXT_REDUCTION", source)

    def test_shared_context_defaults(self):
        from mmlu_eval import EvalConfig
        from mmlu_eval.cli import parser

        config = EvalConfig()
        self.assertEqual(config.context_policy, "reduce")
        self.assertEqual(config.context_tokenizer, "gpt2")
        self.assertEqual(config.max_context_length, 2048)

        cli = parser().parse_args(["--data_dir", "dataset"])
        self.assertEqual(cli.context_policy, "reduce")
        self.assertEqual(cli.context_tokenizer, "gpt2")
        self.assertEqual(cli.max_context_length, 2048)

    def test_preflight_manifest_is_reused_when_no_manifest_given(self):
        for name in (
            "evaluate_aikengpt_mmlu.ipynb",
            "evaluate_aikengpt_mmlu_text.ipynb",
        ):
            source = "\n".join(
                "".join(cell["source"]) for cell in self.notebook(name)["cells"]
            )
            self.assertIn(
                "active_manifest_path = Path(MANIFEST_PATH) if MANIFEST_PATH is not None "
                "else preflight_manifest_path",
                source,
            )
            self.assertIn("manifest_path=active_manifest_path", source)

    def test_readme_identifies_interface_and_dataset_contract(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for value in (
            "evaluate_mmlu_openai_permutation.py",
            "Quick Start: OpenAI",
            "Quick Start: AIkenGPT",
            "<subject1>_dev.csv",
            "question,choice_A,choice_B,choice_C,choice_D,correct_label",
            "同じmanifest",
            "cases_hash",
            ".prompts.jsonl",
            "effective_ntrain",
        ):
            self.assertIn(value, readme)


if __name__ == "__main__":
    unittest.main()
