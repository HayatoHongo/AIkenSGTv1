"""Model-independent MMLU experiment. Never import a model SDK here."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import importlib.metadata

import numpy as np
import pandas as pd

LABELS = ("A", "B", "C", "D")
VERSION = "mmlu-shared-v2"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, frame):
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


@dataclass(frozen=True)
class EvalConfig:
    ntrain: int = 5
    sample_frac: float = 0.1
    seed: int = 42
    limit: int = 0
    subject: str | None = None
    permutation_count: int = 4
    permutation_seed: int = 0  # Recorded; cyclic permutations do not use randomness.
    context_policy: str = "reduce"
    context_tokenizer: str = "gpt2"
    max_context_length: int = 2048

    def __post_init__(self):
        if self.ntrain < -1 or not 0 < self.sample_frac <= 1 or self.limit < 0:
            raise ValueError("Require ntrain >= -1, 0 < sample_frac <= 1, limit >= 0")
        if self.permutation_count != 4:
            raise ValueError("Existing experiment requires exactly four cyclic permutations")
        if self.context_policy not in ("fixed", "reduce"):
            raise ValueError("context_policy must be fixed or reduce")
        if self.max_context_length < 1:
            raise ValueError("max_context_length must be positive")


def cyclic_order(shift):
    return [(i + shift) % 4 for i in range(4)]


def format_subject(subject):
    return " " + " ".join(subject.split("_"))


def format_example(df, idx, include_answer=True, order=None):
    order = cyclic_order(0) if order is None else list(order)
    row = df.iloc[idx]
    prompt = str(row.iloc[0])
    for position, original_index in enumerate(order):
        prompt += "\n{}. {}".format(LABELS[position], row.iloc[original_index + 1])
    prompt += "\nAnswer:"
    if include_answer:
        label = str(row.iloc[5]).strip()
        prompt += " {}\n\n".format(LABELS[order.index(LABELS.index(label))])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = ("The following are multiple choice questions "
              "(with answers) about{}.\n\n").format(format_subject(subject))
    k = len(train_df) if k == -1 else k
    if k > len(train_df):
        raise ValueError(f"{subject}: requested {k} demonstrations, only {len(train_df)} available")
    return prompt + "".join(format_example(train_df, i) for i in range(k))


def load_data(data_dir, subject=None):
    root = Path(data_dir)
    subjects = sorted(p.name[:-9] for p in (root / "test").glob("*_test.csv"))
    if not subjects or (subject is not None and subject not in subjects):
        raise ValueError(f"No test subjects or unknown subject: {subject}")
    data = {}
    for name in ([subject] if subject else subjects):
        frames = []
        for split in ("dev", "test"):
            # Match legacy pandas parsing, including its NA interpretation.
            frame = pd.read_csv(root / split / f"{name}_{split}.csv", header=None)
            if frame.shape[1] != 6 or not frame.iloc[:, 5].astype(str).str.strip().isin(LABELS).all():
                raise ValueError(f"Invalid MMLU {name}/{split}: require six columns and A-D labels")
            frames.append(frame)
        data[name] = tuple(frames)
    return data


def make_sample(data, sample_frac, seed, limit):
    candidates = [(subject, i) for subject, (_, test) in data.items() for i in range(len(test))]
    selected = np.random.default_rng(seed).choice(
        len(candidates), size=round(len(candidates) * sample_frac), replace=False)
    sampled = [candidates[int(i)] for i in selected]
    return sampled[:limit] if limit > 0 else sampled


@dataclass(frozen=True)
class Case:
    subject: str
    test_index: int
    question: str
    choices: tuple
    correct_answer: str
    few_shot_examples: tuple
    effective_ntrain: int
    permutation: tuple
    prompt: str


@dataclass
class ScoreResult:
    # Four unnormalized log scores in DISPLAY position order.
    scores: list
    metadata: dict


def softmax(scores):
    values = np.asarray(scores, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Backend must return four finite log scores")
    values = np.exp(values - np.max(values))
    return values / values.sum()


def aggregate(permutation_probs):
    probs = np.asarray(permutation_probs)
    if probs.shape != (4, 4):
        raise ValueError("Four probability vectors required")
    mapped = np.zeros((4, 4))
    for shift in range(4):
        mapped[shift, cyclic_order(shift)] = probs[shift]
    return probs[0], mapped.mean(axis=0)


def summarize(frame):
    """Micro accuracy, not a macro average of subject accuracies."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty experiment")
    result = {"n": len(frame)}
    for kind in ("baseline", "debiased"):
        correct = frame[f"{kind}_pred"] == frame["label"]
        result[f"{kind}_accuracy"] = float(correct.mean())
        recalls = []
        for label in LABELS:
            subset = frame[frame.label == label]
            recall = None if subset.empty else float((subset[f"{kind}_pred"] == label).mean())
            result[f"{kind}_recall_{label}"] = recall
            recalls.append(recall)
        result[f"{kind}_rstd"] = None if None in recalls else float(np.std(recalls))
    result["difference"] = result["debiased_accuracy"] - result["baseline_accuracy"]
    for label in LABELS:
        result[f"position_{label}_prob"] = float(
            frame[[f"perm{s}_{label}_prob" for s in range(4)]].to_numpy(dtype=float).mean())
    b = frame.baseline_pred == frame.label
    d = frame.debiased_pred == frame.label
    result["wrong_to_correct"] = int((~b & d).sum())
    result["correct_to_wrong"] = int((b & ~d).sum())
    return result


class MMLUEvaluator:
    def __init__(self, data_dir, config=None, *, context_encoder=None):
        self.config = config or EvalConfig()
        self.data = load_data(data_dir, self.config.subject)
        self.encoder = context_encoder
        if self.encoder is None:
            import tiktoken
            self.encoder = tiktoken.get_encoding(self.config.context_tokenizer).encode
        # All loaded test/dev bytes as interpreted by the legacy CSV reader.
        self.dataset_hash = digest({
            subject: [[[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
                      for frame in frames]
            for subject, frames in self.data.items()
        })

    def cases(self, subject, index):
        if subject not in self.data or not 0 <= index < len(self.data[subject][1]):
            raise ValueError(f"Unknown manifest question: {subject}[{index}]")
        dev, test = self.data[subject]
        requested = len(dev) if self.config.ntrain == -1 else self.config.ntrain
        row = test.iloc[index]
        candidate_ntrains = (
            [requested]
            if self.config.context_policy == "fixed"
            else range(requested, -1, -1)
        )
        for k in candidate_ntrains:
            prefix = gen_prompt(dev, subject, k)
            prompts = [prefix + format_example(test, index, False, cyclic_order(s)) for s in range(4)]
            token_lengths = [len(self.encoder(prompt)) for prompt in prompts]
            if all(length <= self.config.max_context_length for length in token_lengths):
                return [Case(subject, int(index), str(row.iloc[0]),
                             tuple(str(x) for x in row.iloc[1:5]), str(row.iloc[5]).strip(),
                             tuple(format_example(dev, i) for i in range(k)), k,
                             tuple(cyclic_order(s)), prompts[s]) for s in range(4)]
            if self.config.context_policy == "fixed":
                raise ValueError(
                    "Fixed context budget exceeded for "
                    f"{subject}[{index}] at ntrain={requested}: "
                    f"permutation_token_lengths={token_lengths}, "
                    f"max_context_length={self.config.max_context_length}, "
                    f"context_tokenizer={self.config.context_tokenizer}"
                )
        raise ValueError(f"Even zero-shot exceeds common context budget: {subject}[{index}]")

    def manifest(self, path=None):
        existing = None
        if path is not None and Path(path).exists():
            existing = pd.read_csv(path, keep_default_na=False)
            if not {"subject", "test_index"}.issubset(existing.columns):
                raise ValueError("Manifest requires subject,test_index")
            sampled = []
            for row in existing.to_dict("records"):
                raw = str(row["test_index"])
                if not raw.isdigit():
                    raise ValueError("Manifest test_index must be a nonnegative integer")
                sampled.append((str(row["subject"]), int(raw)))
            # Input manifest is authoritative: never silently resample/filter/limit it.
        else:
            sampled = make_sample(self.data, self.config.sample_frac, self.config.seed, self.config.limit)
        if not sampled or len(set(sampled)) != len(sampled):
            raise ValueError("Manifest is empty or contains duplicate questions")
        config_hash = digest(asdict(self.config))
        records = []
        for order, (subject, index) in enumerate(sampled):
            cases = self.cases(subject, index)
            records.append({
                "subject": subject, "test_index": index, "sample_order": order,
                "dataset_hash": self.dataset_hash, "config_hash": config_hash,
                "pipeline_version": VERSION, "effective_ntrain": cases[0].effective_ntrain,
                "cases_hash": digest([asdict(c) for c in cases]),
            })
        manifest = pd.DataFrame(records)
        if existing is not None:
            # Legacy two-column manifests are supported; enriched columns must match.
            for column in set(existing.columns) & set(manifest.columns):
                if existing[column].astype(str).tolist() != manifest[column].astype(str).tolist():
                    raise ValueError(f"Manifest mismatch: {column}")
        return manifest

    def export_debug(self, manifest, path, limit=None):
        rows = []
        selected = manifest if limit is None else manifest.head(limit)
        for row in selected.itertuples():
            for case in self.cases(row.subject, int(row.test_index)):
                rows.append(dumps(asdict(case)))
        atomic_text(path, "\n".join(rows) + "\n")

    def evaluate_question(self, backend, subject, index):
        cases = self.cases(subject, index)
        requests = [(c.prompt, tuple(c.choices[i] for i in c.permutation)) for c in cases]
        responses = backend.score_many(requests) if hasattr(backend, "score_many") else [
            backend.score_choices(*request) for request in requests]
        if len(responses) != 4:
            raise ValueError("Backend must return one response per permutation")
        position_probs = [softmax(r.scores) for r in responses]
        baseline, debiased = aggregate(position_probs)
        case = cases[0]
        result = {
            "subject": subject, "test_index": index, "question": case.question,
            **dict(zip(LABELS, case.choices)), "label": case.correct_answer,
            "effective_ntrain": case.effective_ntrain,
            "model_identifier": backend.metadata["model"],
            "scoring_method": backend.metadata["scoring_method"],
            "permutations": dumps([c.permutation for c in cases]),
            "cases_hash": digest([asdict(c) for c in cases]),
        }
        for name, values in (("baseline", baseline), ("debiased", debiased)):
            result[f"{name}_pred"] = LABELS[int(np.argmax(values))]
            result[f"{name}_correct"] = result[f"{name}_pred"] == case.correct_answer
            for label, value in zip(LABELS, values):
                result[f"{name}_{label}_prob"] = float(value)
        for s, response in enumerate(responses):
            for j, label in enumerate(LABELS):
                result[f"perm{s}_{label}_score"] = float(response.scores[j])
                result[f"perm{s}_{label}_prob"] = float(position_probs[s][j])
            result[f"perm{s}_backend_metadata"] = dumps(response.metadata)
            for key in ("missing_count", "error_bound", "fifth_logprob"):
                result[f"perm{s}_{key}"] = response.metadata.get(key)
        for key in ("input_tokens", "output_tokens"):
            result[key] = sum(r.metadata.get(key, 0) for r in responses)
        bounds = [r.metadata.get("error_bound") for r in responses]
        result["mean_error_bound"] = float(np.mean(bounds)) if all(x is not None for x in bounds) else None
        result["max_error_bound"] = max(bounds) if all(x is not None for x in bounds) else None
        result["has_all_missing"] = any(r.metadata.get("missing_count") == 4 for r in responses)
        return result

    def run(self, backend, output_path, *, manifest_path=None, resume=True):
        output = Path(output_path)
        manifest = self.manifest(manifest_path)
        records = manifest.to_dict("records")
        packages = {}
        for package in ("numpy", "pandas", "torch", "tiktoken", "openai"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        source_hash = digest({str(p.relative_to(Path(__file__).parent)): p.read_text(encoding="utf-8")
                              for p in sorted(Path(__file__).parent.rglob("*.py"))})
        run_info = {"source_hash": source_hash, "python": platform.python_version(),
                    "packages": packages, "pipeline_version": VERSION, "evaluation": asdict(self.config),
                    "dataset_hash": self.dataset_hash, "manifest_hash": digest(records),
                    "backend": backend.metadata}
        run_id = digest(run_info)
        metadata_path = output.with_suffix(".run.json")
        results = []
        if output.exists():
            if not resume:
                raise FileExistsError(f"Use a new output path: {output}")
            if not metadata_path.exists() or json.loads(metadata_path.read_text(encoding="utf-8"))["run_id"] != run_id:
                raise ValueError("Resume settings/model/manifest mismatch or legacy CSV; use a new output path")
            existing = pd.read_csv(output, keep_default_na=False)
            allowed = {(r["subject"], r["test_index"]): r["cases_hash"] for r in records}
            seen = set()
            for row in existing.to_dict("records"):
                key = (row["subject"], int(row["test_index"]))
                if key in seen or key not in allowed or row.get("run_id") != run_id or row.get("cases_hash") != allowed[key]:
                    raise ValueError("Invalid/duplicate/stale resume row")
                for kind in ("baseline", "debiased"):
                    if row[f"{kind}_pred"] not in LABELS:
                        raise ValueError("Invalid prediction in resume file")
                    row[f"{kind}_correct"] = row[f"{kind}_pred"] == row["label"]
                seen.add(key)
                results.append(row)
        atomic_text(metadata_path, dumps({**run_info, "run_id": run_id,
                                         "python": platform.python_version(), "packages": packages}))
        atomic_csv(output.with_suffix(".manifest.csv"), manifest)
        # Full, model-independent pre-inference strings; never truncate prompts.
        self.export_debug(manifest, output.with_suffix(".prompts.jsonl"))
        completed = {(r["subject"], int(r["test_index"])) for r in results}
        order = {(r["subject"], r["test_index"]): r["sample_order"] for r in records}
        for record in records:
            key = (record["subject"], record["test_index"])
            if key in completed:
                continue
            result = self.evaluate_question(backend, *key)
            result["sample_order"] = record["sample_order"]
            result["run_id"] = run_id
            results.append(result)
            results.sort(key=lambda r: order[(r["subject"], int(r["test_index"]))])
            atomic_csv(output, pd.DataFrame(results))  # One complete question per checkpoint.
            print(f"{len(results)}/{len(records)} {key[0]}[{key[1]}]", flush=True)
        frame = pd.DataFrame(results)
        atomic_csv(output.with_suffix(".subjects.csv"), pd.DataFrame([
            {"subject": subject, **summarize(group)} for subject, group in frame.groupby("subject")
        ]))
        atomic_text(output.with_suffix(".overall.json"), dumps(summarize(frame)))
        return frame
