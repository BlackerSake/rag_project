import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.prompt_loader import prompt_loader


PROJECT_ROOT = Path(__file__).parent.parent


def test_clarify_prompt_accepts_conversation_history():
    prompt = prompt_loader.get_prompt("clarify_question_prompt")

    assert "messages" in prompt.input_variables
    assert "conversation_history" in prompt.input_variables


def test_clarify_node_passes_history_to_prompt():
    source = (PROJECT_ROOT / "core" / "nodes.py").read_text(encoding="utf-8")

    assert '"messages": messages' in source
    assert '"conversation_history": conversation_history' in source
    assert "历史消息数" in source


def test_logistics_clarification_uses_correct_parameter_key():
    source = (PROJECT_ROOT / "core" / "nodes.py").read_text(encoding="utf-8")

    assert '"clarify_parameters": "請提供需要查詢的物流單號。"' in source
    assert '"clarify_paremeters": "請提供需要查詢的物流單號。"' not in source


def test_tracking_number_pattern_supports_alphanumeric_ids():
    source = (PROJECT_ROOT / "core" / "nodes.py").read_text(encoding="utf-8")

    assert "[A-Za-z]{1,4}" in source
    assert "\\d{3,}" in source


def test_decomposition_and_rewrite_prompts_accept_context():
    judge_prompt = prompt_loader.get_prompt("query_decomposition_judge")
    rewrite_prompt = prompt_loader.get_prompt("query_rewrite")

    assert "summary" in judge_prompt.input_variables
    assert "conversation_history" in judge_prompt.input_variables
    assert "summary" in rewrite_prompt.input_variables
    assert "conversation_history" in rewrite_prompt.input_variables


def test_decomposition_and_rewrite_nodes_pass_context_to_llm():
    source = (PROJECT_ROOT / "core" / "nodes.py").read_text(encoding="utf-8")

    assert "conversation_history = _format_conversation_history(messages)" in source
    assert "conversation_history = _format_conversation_history(state.get(\"messages\", []))" in source
    assert "summary=summary" in source
    assert '"summary": summary' in source
    assert '"conversation_history": conversation_history' in source
