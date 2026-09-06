"""Deprecated compatibility wrapper; use evaluate_mmlu_openai_permutation.py."""
import warnings

warnings.warn(
    "evaluate_mmlu_permutation.py is deprecated; use "
    "evaluate_mmlu_openai_permutation.py. Both call the shared mmlu_eval pipeline.",
    FutureWarning,
    stacklevel=2,
)

from mmlu_eval.core import (
    LABELS as choices, cyclic_order, format_example, format_subject,
    gen_prompt, make_sample, softmax, summarize,
)
from mmlu_eval.cli import main

if __name__ == "__main__":
    main()
