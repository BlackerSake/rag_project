"""结构化输出统一入口。"""

from .fallbacks import fallback_judge_items, fallback_rewrite_items, fallback_structured_state
from .parser import (
    StructuredOutputParseError,
    extract_json_object,
    parse_json_array_or_object,
    parse_json_output,
)
from .repair import repair_json_text, repair_judge_output_with_llm, should_attempt_repair
from .schemas import (
    JudgeItemSchema,
    JudgeOutputSchema,
    RewriteItemSchema,
    RewriteOutputSchema,
    StructuredStateSchema,
)
from .validators import (
    dedupe_candidate_assignments,
    recover_missing_candidates,
    sanitize_judge_items,
    sanitize_rewrite_items,
    validate_candidate_ids,
    validate_intent_id,
)
from .workflow import parse_and_validate_judge_output, parse_and_validate_judge_output_async

__all__ = [
    "StructuredOutputParseError",
    "JudgeItemSchema",
    "JudgeOutputSchema",
    "RewriteItemSchema",
    "RewriteOutputSchema",
    "StructuredStateSchema",
    "dedupe_candidate_assignments",
    "extract_json_object",
    "fallback_judge_items",
    "fallback_rewrite_items",
    "fallback_structured_state",
    "parse_and_validate_judge_output",
    "parse_and_validate_judge_output_async",
    "parse_json_array_or_object",
    "parse_json_output",
    "recover_missing_candidates",
    "repair_json_text",
    "repair_judge_output_with_llm",
    "sanitize_judge_items",
    "sanitize_rewrite_items",
    "should_attempt_repair",
    "validate_candidate_ids",
    "validate_intent_id",
]
