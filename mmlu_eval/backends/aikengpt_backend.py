"""Exact local letter logits or teacher-forced choice-text likelihood."""
from pathlib import Path
import hashlib
from ..core import LABELS, ScoreResult


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


class AIkenGPTBackend:
    def __init__(self, model, tokenizer, *, model_identifier,
                 tokenizer_identifier="gpt2", scoring_method="letter",
                 text_reduction=None, device="cuda", dtype="float32", batch_size=1,
                 max_context_length=2048, model_format="custom",
                 checkpoint_sha256=None):
        import torch
        if scoring_method not in ("letter", "choice_text"):
            raise ValueError("scoring_method must be letter or choice_text")
        if scoring_method == "choice_text" and text_reduction not in ("sum", "mean"):
            raise ValueError("Explicitly select text_reduction='sum' or 'mean'; no historical default exists")
        if scoring_method == "letter" and text_reduction is not None:
            raise ValueError("text_reduction applies only to choice_text")
        if batch_size < 1 or max_context_length < 1 or model_format not in ("custom", "hf"):
            raise ValueError("Invalid local inference settings")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        self.model = model.to(device=device, dtype=getattr(torch, dtype)).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.model_format = model_format
        self.method = scoring_method
        self.reduction = text_reduction
        self.answer_ids = []
        if scoring_method == "letter":
            for label in LABELS:
                ids = self.encode(" " + label)
                if len(ids) != 1:
                    raise ValueError(f"Expected single answer token: {label}")
                self.answer_ids.append(ids[0])
        self.metadata = {
            "model": model_identifier, "tokenizer": tokenizer_identifier,
            "scoring_method": scoring_method, "text_reduction": text_reduction,
            "device": str(device), "dtype": dtype, "batch_size": batch_size,
            "max_context_length": max_context_length, "model_format": model_format,
            "checkpoint_sha256": checkpoint_sha256, "answer_prefix": " ",
            "candidate_tokenization": "encode(prompt) + encode(' ' + text)",
            "add_special_tokens": False,
        }

    def encode(self, text):
        if self.model_format == "hf":
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text)

    def _logits(self, ids, mask):
        if self.model_format == "hf":
            return self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return self.model(ids, None, use_cache=False)[0]

    def score_choices(self, prompt, choices):
        import torch
        prompt_ids = self.encode(prompt)
        if not prompt_ids:
            raise ValueError("Empty tokenized prompt")
        with torch.inference_mode():
            if self.method == "letter":
                if len(prompt_ids) > self.max_context_length:
                    raise ValueError("Prompt exceeds context; use the SAME common reduce policy for BOTH models")
                ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                logits = self._logits(ids, torch.ones_like(ids))
                scores = logits[0, -1, self.answer_ids].float().cpu().tolist()
                return ScoreResult(scores, {"input_tokens": len(prompt_ids), "output_tokens": 0})
            candidates = [self.encode(" " + text) for text in choices]
            if any(not ids for ids in candidates):
                raise ValueError("Empty tokenized candidate")
            # Fixed, explicit token boundary. No EOS, truncation or automatic shot reduction.
            sequences = [prompt_ids + ids for ids in candidates]
            if any(len(ids) > self.max_context_length for ids in sequences):
                raise ValueError("Prompt plus candidate exceeds context; create a shared shorter-shot experiment")
            scores = []
            for start in range(0, len(sequences), self.batch_size):
                chunk = sequences[start:start + self.batch_size]
                max_len = max(map(len, chunk))
                ids = torch.zeros((len(chunk), max_len), dtype=torch.long, device=self.device)
                mask = torch.zeros_like(ids)
                for i, seq in enumerate(chunk):
                    ids[i, :len(seq)] = torch.tensor(seq, device=self.device)
                    mask[i, :len(seq)] = 1
                logits = self._logits(ids, mask)
                for i, seq in enumerate(chunk):
                    begin = len(prompt_ids) - 1
                    end = len(seq) - 1
                    log_probs = logits[i, begin:end].float().log_softmax(-1)
                    targets = ids[i, begin + 1:end + 1]
                    values = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                    value = values.sum() if self.reduction == "sum" else values.mean()
                    scores.append(float(value.cpu()))
            return ScoreResult(scores, {
                "input_tokens": sum(map(len, sequences)), "output_tokens": 0,
                "candidate_token_counts": list(map(len, candidates)),
                "text_reduction": self.reduction,
            })


def load_custom_checkpoint(path, *, device="cuda", dtype="float32", max_context_length=2048):
    """Original notebook architecture and parameter-by-parameter loading."""
    import torch
    from safetensors import safe_open
    from .aikengpt_model import Config, GPT
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype")
    config = Config()
    config.max_sequence_length = max_context_length
    with torch.device(device):
        model = GPT(config).to(dtype=getattr(torch, dtype))
    parameters = dict(model.named_parameters())
    with safe_open(str(path), framework="pt", device="cpu") as stream:
        if set(stream.keys()) != set(parameters):
            raise ValueError("Checkpoint keys do not match original GPT architecture")
        with torch.no_grad():
            for name in stream.keys():
                value = stream.get_tensor(name)
                if value.shape != parameters[name].shape:
                    raise ValueError(f"Checkpoint shape mismatch: {name}")
                parameters[name].copy_(value)
    return model.eval()
