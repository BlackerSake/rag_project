from typing import Dict, List, TypedDict, Optional
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
import operator

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    query: Optional[str]
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
    intent_candidates: List[Dict]
    intent_candidate_results: List[Dict]
    intent_margin: Optional[float]
    intent_confidence_level: Optional[str]
    intent_gate_action: Optional[str]
    intent_gate_reason: Optional[str]
    clarification_question: Optional[str]
    final_route: Optional[str]
    sub_questions: List[str]
    is_complex_query: bool
    multi_intent_results: List[tuple]
    decompose_skipped: bool
    processed_results: List[Dict]
    judge_items: List[Dict]
    candidate_by_id: Dict[str, tuple]
    tool_execution_history: List[Dict]
