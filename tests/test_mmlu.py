import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from mmlu_eval.core import (
    EvalConfig, MMLUEvaluator, ScoreResult, aggregate, atomic_csv, cyclic_order,
    format_example, gen_prompt, make_sample, softmax, summarize,
)
from mmlu_eval.compare import compare
from mmlu_eval.backends.openai_backend import OpenAIBackend, extract_scores, MissingChoiceLogprobError

ROOT = Path(__file__).resolve().parents[1]


class RecordingBackend:
    def __init__(self, name):
        self.metadata = {"model": name, "scoring_method": "letter"}
        self.calls = []

    def score_choices(self, prompt, choices):
        self.calls.append((prompt, choices))
        # Unequal scores catch mapping errors.
        return ScoreResult([0.0, -1.0, -2.0, -3.0], {})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("dev", "test"):
            (self.root / split).mkdir()
            for subject in ("abstract_algebra", "world_history"):
                rows = [[f"{subject} {split} Q{i}\nsecond line", "one", "two words", "三", "  four  ", "ABCD"[i % 4]]
                        for i in range(5 if split == "dev" else 12)]
                pd.DataFrame(rows).to_csv(self.root / split / f"{subject}_{split}.csv", header=False, index=False)
        self.config = EvalConfig(sample_frac=1, limit=10, ntrain=5)
        self.evaluator = MMLUEvaluator(self.root, self.config)

    def test_legacy_prompt_and_sampling_golden(self):
        source = (ROOT / "legacy/evaluate_mmlu_permutation.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        names = {"format_subject", "format_example", "gen_prompt", "cyclic_order", "make_sample"}
        code = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        legacy = {"np": np, "choices": list("ABCD")}
        exec(compile(code, "legacy", "exec"), legacy)
        for k in (0, 5):
            for subject, (dev, test) in self.evaluator.data.items():
                self.assertEqual(gen_prompt(dev, subject, k), legacy["gen_prompt"](dev, subject, k))
                for shift in range(4):
                    self.assertEqual(format_example(test, 0, False, cyclic_order(shift)),
                                     legacy["format_example"](test, 0, False, cyclic_order(shift)))
        self.assertEqual(make_sample(self.evaluator.data, .5, 42, 10),
                         legacy["make_sample"](self.evaluator.data, .5, 42, 10))

    def test_ten_questions_exact_pre_backend_identity_and_resume(self):
        manifest = self.evaluator.manifest()
        path = self.root / "manifest.csv"
        atomic_csv(path, manifest)
        other = MMLUEvaluator(self.root, self.config)
        a, b = RecordingBackend("openai-fixture"), RecordingBackend("aikengpt-fixture")
        one = self.evaluator.run(a, self.root / "openai.csv", manifest_path=path)
        two = other.run(b, self.root / "aikengpt.csv", manifest_path=path)
        self.assertEqual(len(a.calls), 40)
        self.assertEqual(a.calls, b.calls)
        self.assertEqual(compare(self.root / "openai.prompts.jsonl", self.root / "aikengpt.prompts.jsonl"), 40)
        self.assertEqual(one.baseline_pred.tolist(), two.baseline_pred.tolist())
        self.evaluator.run(a, self.root / "openai.csv", manifest_path=path)
        self.assertEqual(len(a.calls), 40)

    def test_mapping_hand_calculated_and_tie(self):
        probabilities = np.array([[.7,.1,.1,.1], [.1,.6,.2,.1], [.1,.1,.7,.1], [.1,.1,.1,.7]])
        baseline, debiased = aggregate(probabilities)
        np.testing.assert_allclose(baseline, [.7,.1,.1,.1])
        np.testing.assert_allclose(debiased, [.4,.1,.375,.125])
        self.assertEqual(int(np.argmax(aggregate(np.full((4,4), .25))[1])), 0)

    def test_legacy_manifest_order_authoritative(self):
        old = self.root / "old.csv"
        pd.DataFrame([["world_history", 9], ["abstract_algebra", 2]], columns=["subject", "test_index"]).to_csv(old, index=False)
        manifest = self.evaluator.manifest(old)
        self.assertEqual(manifest.subject.tolist(), ["world_history", "abstract_algebra"])
        self.assertEqual(manifest.test_index.tolist(), [9, 2])

    def test_reject_duplicates_stale_data_and_wrong_config(self):
        path = self.root / "manifest.csv"
        manifest = self.evaluator.manifest()
        atomic_csv(path, pd.concat([manifest, manifest.head(1)]))
        with self.assertRaises(ValueError):
            self.evaluator.manifest(path)
        atomic_csv(path, manifest)
        with self.assertRaises(ValueError):
            MMLUEvaluator(self.root, replace(self.config, ntrain=0)).manifest(path)
        data_path = self.root / "test/abstract_algebra_test.csv"
        data_path.write_text(data_path.read_text(encoding="utf-8").replace("Q0", "changed"), encoding="utf-8")
        with self.assertRaises(ValueError):
            MMLUEvaluator(self.root, self.config).manifest(path)

    def test_five_shot_within_budget_keeps_five(self):
        evaluator = MMLUEvaluator(
            self.root,
            replace(self.config, context_policy="reduce", max_context_length=2048),
            context_encoder=lambda prompt: range(2048),
        )
        cases = evaluator.cases("abstract_algebra", 0)
        self.assertEqual({case.effective_ntrain for case in cases}, {5})

    def test_reduce_uses_four_when_five_exceeds_budget(self):
        def encode(prompt):
            shots = prompt.count("Answer:") - 1
            return range(1000 + shots * 250)

        evaluator = MMLUEvaluator(
            self.root,
            replace(self.config, context_policy="reduce", max_context_length=2048),
            context_encoder=encode,
        )
        cases = evaluator.cases("abstract_algebra", 0)
        self.assertEqual({case.effective_ntrain for case in cases}, {4})
        self.assertTrue(all(len(encode(case.prompt)) <= 2048 for case in cases))

    def test_reduce_checks_all_four_permutations(self):
        def encode(prompt):
            shots = prompt.count("Answer:") - 1
            test_prompt = prompt.rsplit("\n\n", 1)[-1]
            one_permutation_exceeds = shots == 5 and "\nA. two words" in test_prompt
            return range(2049 if one_permutation_exceeds else 2048)

        evaluator = MMLUEvaluator(
            self.root,
            replace(self.config, context_policy="reduce", max_context_length=2048),
            context_encoder=encode,
        )
        cases = evaluator.cases("abstract_algebra", 0)
        self.assertEqual({case.effective_ntrain for case in cases}, {4})

    def test_fixed_raises_when_any_permutation_exceeds(self):
        def encode(prompt):
            test_prompt = prompt.rsplit("\n\n", 1)[-1]
            return range(2049 if "\nA. two words" in test_prompt else 2048)

        evaluator = MMLUEvaluator(
            self.root,
            replace(self.config, context_policy="fixed", max_context_length=2048),
            context_encoder=encode,
        )
        with self.assertRaisesRegex(ValueError, "Fixed context budget exceeded"):
            evaluator.cases("abstract_algebra", 0)

    def test_reduce_passes_identical_prompts_to_both_backends(self):
        def encode(prompt):
            shots = prompt.count("Answer:") - 1
            return range(1000 + shots * 250)

        evaluator = MMLUEvaluator(
            self.root,
            replace(self.config, context_policy="reduce", max_context_length=2048),
            context_encoder=encode,
        )
        openai = RecordingBackend("openai")
        aikengpt = RecordingBackend("aikengpt")
        evaluator.evaluate_question(openai, "abstract_algebra", 0)
        evaluator.evaluate_question(aikengpt, "abstract_algebra", 0)
        self.assertEqual(openai.calls, aikengpt.calls)
        self.assertEqual(len(openai.calls), 4)


    def test_interrupted_run_saves_only_complete_question(self):
        class Failing(RecordingBackend):
            def score_choices(self, prompt, choices):
                if len(self.calls) == 5:
                    raise RuntimeError("simulated failure")
                return super().score_choices(prompt, choices)
        path = self.root / "interrupted.csv"
        with self.assertRaises(RuntimeError):
            self.evaluator.run(Failing("same"), path)
        self.assertEqual(len(pd.read_csv(path)), 1)
        backend = RecordingBackend("same")
        result = self.evaluator.run(backend, path)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(backend.calls), 36)

    def test_resume_rejects_model_and_unidentified_legacy(self):
        output = self.root / "result.csv"
        self.evaluator.run(RecordingBackend("one"), output)
        with self.assertRaises(ValueError):
            self.evaluator.run(RecordingBackend("two"), output)
        output.with_suffix(".run.json").unlink()
        with self.assertRaises(ValueError):
            self.evaluator.run(RecordingBackend("one"), output)

    def test_micro_metrics_ignore_string_boolean_trap(self):
        rows = []
        for i in range(3):
            rows.append({"subject": "x", "baseline_pred": "A", "debiased_pred": "B",
                         "label": "A" if i < 2 else "B", "baseline_correct": "False",
                         **{f"perm{s}_{l}_prob": .25 for s in range(4) for l in "ABCD"}})
        result = summarize(pd.DataFrame(rows))
        self.assertEqual(result["baseline_accuracy"], 2 / 3)
        self.assertEqual(result["wrong_to_correct"], 1)

    def test_bad_scores_and_permutation_configuration(self):
        for values in ([0, 1], [0, 1, 2, float("nan")]):
            with self.assertRaises(ValueError):
                softmax(values)
        with self.assertRaises(ValueError):
            EvalConfig(permutation_count=3)


    def test_legacy_na_string_hashing(self):
        path = self.root / "test/abstract_algebra_test.csv"
        frame = pd.read_csv(path, header=None)
        frame.iloc[0, 1] = "N/A"
        frame.to_csv(path, header=False, index=False)
        evaluator = MMLUEvaluator(self.root, self.config)
        case = evaluator.cases("abstract_algebra", 0)[0]
        self.assertEqual(case.choices[0], "nan")
        self.assertIn("A. nan", case.prompt)
        self.assertEqual(len(evaluator.dataset_hash), 64)

    def test_trace_detects_whitespace(self):
        manifest = self.evaluator.manifest()
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        self.evaluator.export_debug(manifest, a, limit=1)
        rows = [json.loads(x) for x in a.read_text(encoding="utf-8").splitlines()]
        rows[0]["prompt"] += " "
        b.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
        with self.assertRaises(AssertionError):
            compare(a, b)


class OpenAITests(unittest.TestCase):
    def test_missing_probability_legacy(self):
        result = extract_scores({" A": -1, " B": -2, "x": -3, "y": -4, "z": -5})
        self.assertEqual(result.scores, [-1, -2, -100, -100])
        expected = 2*np.exp(-5)/(np.exp(-1)+np.exp(-2)+2*np.exp(-5))
        self.assertAlmostEqual(result.metadata["error_bound"], expected)
        with self.assertRaises(MissingChoiceLogprobError):
            extract_scores({"x": -1})

    def test_transport_prompt_and_scores(self):
        top = {f" {l}": -i-1.0 for i,l in enumerate("ABCD")}
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(logprobs=SimpleNamespace(
                top_logprobs=[top], content=[SimpleNamespace(top_logprobs=[
                    SimpleNamespace(token=k, logprob=v) for k,v in top.items()])]))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1), model="snapshot")
        client = SimpleNamespace(completions=SimpleNamespace(create=create),
                                 chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        for mode in ("completions", "chat"):
            backend = OpenAIBackend(api_mode=mode, client=client, request_interval=0)
            result = backend.score_choices("exact\nAnswer:", ("x","y","z","w"))
            self.assertEqual(result.scores, [-1,-2,-3,-4])
            actual = calls[-1].get("prompt") if mode == "completions" else calls[-1]["messages"][0]["content"]
            self.assertEqual(actual, "exact\nAnswer:")


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed; run these tests in Colab too")
class LocalBackendTests(unittest.TestCase):
    def test_text_sum_mean_and_batch_equivalence(self):
        from mmlu_eval.backends.aikengpt_backend import AIkenGPTBackend
        class Tokenizer:
            def encode(self, text):
                return [ord(c) % 8 for c in text]
        class TinyModel(torch.nn.Module):
            def forward(self, ids, target, use_cache=False):
                logits = torch.arange(8, device=ids.device, dtype=torch.float32).expand(*ids.shape, 8)
                return logits, None
        choices = ("a", "bb", "ccc", "dddd")
        outputs = {}
        for reduction in ("sum", "mean"):
            for batch in (1, 4):
                backend = AIkenGPTBackend(TinyModel(), Tokenizer(), model_identifier="tiny",
                    scoring_method="choice_text", text_reduction=reduction,
                    device="cpu", batch_size=batch, max_context_length=100)
                outputs[reduction, batch] = backend.score_choices("prompt", choices).scores
                expected = []
                lp = torch.arange(8, dtype=torch.float32).log_softmax(-1)
                for text in choices:
                    values = lp[Tokenizer().encode(" " + text)]
                    expected.append(float(values.sum() if reduction == "sum" else values.mean()))
                np.testing.assert_allclose(outputs[reduction,batch], expected, rtol=1e-6)
        np.testing.assert_allclose(outputs["sum",1], outputs["sum",4])

    def test_letter_logits(self):
        from mmlu_eval.backends.aikengpt_backend import AIkenGPTBackend
        class Tokenizer:
            def encode(self, text):
                return [list("ABCD").index(text[1])] if text in (" A"," B"," C"," D") else [0, 1]
        class TinyModel(torch.nn.Module):
            def forward(self, ids, target, use_cache=False):
                return torch.tensor([1., 2., 3., 4.]).expand(*ids.shape,4), None
        backend = AIkenGPTBackend(TinyModel(), Tokenizer(), model_identifier="tiny", device="cpu")
        self.assertEqual(backend.score_choices("prompt", ("x",)*4).scores, [1.,2.,3.,4.])


if __name__ == "__main__":
    unittest.main()
