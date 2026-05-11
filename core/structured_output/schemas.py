"""结构化输出 Pydantic Schema 定义。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JudgeItemSchema(BaseModel):
    """单个查询拆解裁决项。"""

    model_config = ConfigDict(extra="ignore")

    candidate_ids: list[str] = Field(default_factory=list)
    intent_id: str = ""
    reason: str = ""

    @field_validator("candidate_ids", mode="before")
    @classmethod
    def _coerce_candidate_ids(cls, value: Any) -> Any:
        """将候选 ID 统一转换为字符串列表。"""
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @field_validator("intent_id", "reason", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        """将文本字段统一转换为字符串。"""
        return "" if value is None else str(value).strip()


class JudgeOutputSchema(BaseModel):
    """查询拆解裁决输出。"""

    model_config = ConfigDict(extra="ignore")

    items: list[JudgeItemSchema] = Field(default_factory=list)


class RewriteItemSchema(BaseModel):
    """单个查询改写项。"""

    model_config = ConfigDict(extra="ignore")

    original: str = ""
    intent_id: str = ""
    rewritten: str = ""

    @field_validator("original", "intent_id", "rewritten", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        """将改写文本字段统一转换为字符串。"""
        return "" if value is None else str(value).strip()


class RewriteOutputSchema(BaseModel):
    """查询改写输出。"""

    model_config = ConfigDict(extra="ignore")

    rewritten: list[RewriteItemSchema] = Field(default_factory=list)


class StructuredStateSchema(BaseModel):
    """摘要结构化状态输出。"""

    model_config = ConfigDict(extra="ignore")

    goal: str = ""
    constraints: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    discarded_context: list[str] = Field(default_factory=list)
