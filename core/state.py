from typing import Dict, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
import operator

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_topic: str
    history: List[Dict[str, str]]
    knowledge_results: List[tuple]
    highest_score: float
    confidence_decision: Optional[str]
    confidence_score: Optional[float]
    conversation_rounds: int
    summary: Optional[str]
    intent_id: Optional[str]
    intent_score: Optional[float]
    sub_questions: List[str]
    is_complex_query: bool
    multi_intent_results: List[tuple]
    decompose_skipped: bool
    processed_results: List[Dict]
    judge_items: List[Dict]
    candidate_by_id: Dict[str, tuple]
    tool_execution_history: List[Dict]
