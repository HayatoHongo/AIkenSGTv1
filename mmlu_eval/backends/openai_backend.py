"""OpenAI-only transport, concurrency, rate limiting, retry and top-k extraction."""
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

from ..core import LABELS, ScoreResult


class MissingChoiceLogprobError(ValueError):
    pass


def extract_scores(top, prefix=" "):
    if not top:
        raise MissingChoiceLogprobError("No top logprobs returned")
    scores = [float(top.get(prefix + label, -100.0)) for label in LABELS]
    missing = sum(prefix + label not in top for label in LABELS)
    if missing == 4:
        raise MissingChoiceLogprobError(
            "All four candidate tokens are absent. Check endpoint/answer_prefix; "
            "do not silently count this as an incorrect answer.")
    floor = min(top.values())
    visible = sum(math.exp(top[prefix + label]) for label in LABELS if prefix + label in top)
    upper = missing * math.exp(floor)
    return ScoreResult(scores, {
        "missing_count": missing,
        "error_bound": upper / (visible + upper) if missing else 0.0,
        "fifth_logprob": float(floor),
    })


class OpenAIBackend:
    def __init__(self, model="gpt-4o-mini", *, api_mode="completions",
                 answer_prefix=" ", workers=10, request_interval=1.0, retries=5,
                 tokenizer="server-managed", max_context_length=128000,
                 client=None):
        if api_mode not in ("completions", "chat") or answer_prefix not in (" ", ""):
            raise ValueError("Invalid api_mode or answer_prefix")
        if workers < 1 or request_interval < 0 or retries < 1 or max_context_length < 1:
            raise ValueError("Invalid transport settings")
        if client is None:
            from openai import OpenAI
            # The SDK must not add another hidden retry layer.
            client = OpenAI(max_retries=0)
        self.client = client
        self.model = model
        self.api_mode = api_mode
        self.answer_prefix = answer_prefix
        self.workers = workers
        self.interval = request_interval
        self.retries = retries
        self.max_context_length = max_context_length
        self._lock = threading.Lock()
        self._next_request = 0.0
        self.metadata = {
            "model": model, "scoring_method": "letter", "api_mode": api_mode,
            "answer_prefix": answer_prefix, "tokenizer": tokenizer,
            "max_context_length": max_context_length, "batch_size": None,
            "device": "remote", "dtype": "server-managed", "workers": workers,
            "request_interval": request_interval, "retries": retries,
            "top_logprobs": 5, "missing_score": -100.0, "temperature": 0,
            "max_tokens": 1, "chat_wrapper": api_mode == "chat",
        }

    def _wait(self):
        with self._lock:
            now = time.monotonic()
            delay = max(0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self.interval
        if delay:
            time.sleep(delay)

    def score_many(self, requests):
        # requests contain only prompt and displayed candidate text.
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.score_choices, prompt, choices) for prompt, choices in requests]
            return [future.result() for future in futures]

    def score_choices(self, prompt, choices):
        from openai import APIConnectionError, APITimeoutError, APIStatusError
        for attempt in range(self.retries):
            self._wait()
            try:
                if self.api_mode == "completions":
                    response = self.client.completions.create(
                        model=self.model, prompt=prompt, max_tokens=1,
                        temperature=0, logprobs=5)
                    top = response.choices[0].logprobs.top_logprobs[-1]
                else:
                    response = self.client.chat.completions.create(
                        model=self.model, messages=[{"role": "user", "content": prompt}],
                        max_tokens=1, temperature=0, logprobs=True, top_logprobs=5)
                    content = response.choices[0].logprobs.content
                    if not content:
                        raise MissingChoiceLogprobError("No first-token logprobs (possibly refusal)")
                    top = {item.token: item.logprob for item in content[0].top_logprobs}
                result = extract_scores(top, self.answer_prefix)
                if response.usage.prompt_tokens + response.usage.completion_tokens > self.max_context_length:
                    raise ValueError("API usage exceeds configured context budget")
                result.metadata.update(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    resolved_model=getattr(response, "model", self.model),
                    system_fingerprint=getattr(response, "system_fingerprint", None),
                )
                return result
            except (APIConnectionError, APITimeoutError, APIStatusError) as error:
                status = getattr(error, "status_code", None)
                if status is not None and status not in (408, 409, 429) and status < 500:
                    raise
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
