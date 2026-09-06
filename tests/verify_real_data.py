"""Offline integration on REAL MMLU data, with FAKE models. No API calls."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / ".test-deps").exists():
    sys.path.insert(0, str(ROOT / ".test-deps"))

import torch
from mmlu_eval import EvalConfig, MMLUEvaluator
from mmlu_eval.core import atomic_csv, atomic_text, dumps
from mmlu_eval.compare import compare
from mmlu_eval.backends.openai_backend import OpenAIBackend
from mmlu_eval.backends.aikengpt_backend import AIkenGPTBackend


class TinyTokenizer:
    def encode(self, text):
        if text in (" A", " B", " C", " D"):
            return ["ABCD".index(text[1])]
        return [ord(c) % 8 for c in text]


class TinyModel(torch.nn.Module):
    def forward(self, ids, target, use_cache=False):
        return -torch.arange(1, 9, device=ids.device, dtype=torch.float32).expand(*ids.shape, 8), None


def fake_response(**kwargs):
    return SimpleNamespace(
        model="FAKE-openai",
        choices=[SimpleNamespace(logprobs=SimpleNamespace(
            top_logprobs=[{f" {label}": -float(i + 1) for i, label in enumerate("ABCD")}]))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def run(data_dir, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = EvalConfig(ntrain=5, sample_frac=.1, seed=42, limit=10)
    reference = MMLUEvaluator(data_dir, config)
    manifest_path = output / "sample_manifest.csv"
    atomic_csv(manifest_path, reference.manifest())
    backends = {
        "mock_openai": OpenAIBackend(
            model="FAKE-openai", client=SimpleNamespace(completions=SimpleNamespace(create=fake_response)),
            request_interval=0, workers=1),
        "tiny_letter": AIkenGPTBackend(
            TinyModel(), TinyTokenizer(), model_identifier="FAKE-local", device="cpu", max_context_length=100000),
    }
    for reduction in ("sum", "mean"):
        backends[f"tiny_text_{reduction}"] = AIkenGPTBackend(
            TinyModel(), TinyTokenizer(), model_identifier="FAKE-local",
            device="cpu", max_context_length=100000, scoring_method="choice_text",
            text_reduction=reduction, batch_size=4)
    all_calls = {}
    result_columns = []
    reference_cases = [asdict(c) for row in reference.manifest(manifest_path).itertuples()
                       for c in reference.cases(row.subject, int(row.test_index))]
    for name, backend in backends.items():
        evaluator = MMLUEvaluator(data_dir, config)
        manifest = evaluator.manifest(manifest_path)
        calls = []
        original_score = backend.score_choices
        def recording_score(prompt, choices, _score=original_score, _calls=calls):
            _calls.append((prompt, choices))
            return _score(prompt, choices)
        backend.score_choices = recording_score
        cases = []
        for row in manifest.itertuples():
            cases.extend(asdict(c) for c in evaluator.cases(row.subject, int(row.test_index)))
            result = evaluator.evaluate_question(backend, row.subject, int(row.test_index))
            result_columns.append(set(result))
        assert cases == reference_cases, name
        assert len(calls) == 40, name
        all_calls[name] = calls
        evaluator.export_debug(manifest, output / f"{name}.prompts.jsonl")
    first = all_calls["mock_openai"]
    assert all(calls == first for calls in all_calls.values())
    assert all(columns == result_columns[0] for columns in result_columns)
    assert all(compare(output / "mock_openai.prompts.jsonl", output / f"{name}.prompts.jsonl") == 40
               for name in backends)
    report = {
        "real_model_accuracy_measured": False,
        "purpose": "Real MMLU data, fake API and tiny local model; pipeline equivalence only",
        "subjects_in_dataset": len(reference.data),
        "questions": 10, "permutations_per_question": 4,
        "backends_checked": list(backends), "all_pre_backend_calls_identical": True,
        "question_choices_labels_few_shots_permutations_prompts_identical": True,
        "result_schema_identical": True,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
    }
    atomic_text(output / "report.json", dumps(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT.parent / "mmlu-reference/data"))
    parser.add_argument("--output_dir", default=str(ROOT / "verification/real_data"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
