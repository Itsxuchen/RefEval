"""Frozen single-criterion prompts, context serialization and label parsing.

Prompt wording and context serialization adapt THU-KEG/RuVerBench revision
0af0256870d6322490c7ab15eb6370779b898a37 (Copyright 2026 Yangda Peng).
See licenses/RuVerBench-CODE_LICENSE and docs/DATA_SOURCES.md. Extracted from
the study's generation runner; network, credentials and API dispatch are omitted.
Parsing rejects unresolved responses instead of replacing them with guessed bits.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Sequence

class StrictParseError(ValueError):
    """A response has no unambiguous label for the requested criterion."""

def _deepresearch_single_prompt(question: str, response: str, point: dict[str, Any]) -> str:
    # Exact standard single-rubric template from the pinned upstream commit.
    return f"""# Rubric Coverage Evaluation

## Task
Determine whether the AI response adequately covers the specific criterion provided. Answer with "yes" or "no" followed by a brief justification.

## Input Materials
<Question>: {question}
<AI Response>: {response}
<criterion>: {point['point']}

## Evaluation Criteria
- Answer "yes" if the AI response clearly includes or adequately expresses the main content of the criterion
- Answer "yes" if the response conveys the same meaning as the criterion, even if using different terminology or phrasing
- Answer "no" if the AI response only partially addresses or completely fails to mention the content of the criterion
- Consider semantic equivalence, not just keyword matching
- Pay special attention to technical details, numerical values, and specific claims in the criterion

## Output Format
Your answer must begin with either "yes" or "no" followed by a brief justification.

Example format:
"yes: The response clearly addresses this criterion by explaining [specific detail]..."
"no: While the response mentions [related concept], it fails to address [specific aspect] of the criterion..."

Note: Your assessment will be used to calculate recall metrics, so accuracy is critical."""

AGENTIC_SINGLE_TEMPLATE = """
你是一个「Agent 指令遵循」评审模型。

你的任务是：根据给定的 **单个 Check 项**，评估 assistant 在对话中的表现。

=====INPUT CONVERSATION=====
====TOOLS===
{tools}
====TOOLS===
====MESSAGES===
{messages}
====MESSAGES===
=====INPUT CONVERSATION=====

=====CHECK TO EVALUATE=====
Category: {category}
Check ID: {check_id}
Description: {check_description}
=====CHECK TO EVALUATE=====

--------------------------------------------------
评估规则
--------------------------------------------------

1. **评估依据**：检查所有 `role == "assistant"` 的消息，包括：
   - 自然语言输出
   - 内部推理（reasoning_content，如有）
   - 工具调用（tool_calls）

2. **判定标准**：
   - `"success"`：assistant 明确遵循了该要求
   - `"fail"`：assistant 明显违背了该要求，或在应该遵循时未遵循

3. **reasoning 字段**：
   - 对于 "fail"：简要说明 assistant 如何违背该要求
   - 对于 "success"：简要说明 assistant 如何遵循该要求

--------------------------------------------------
输出格式（必须为合法 JSON）
--------------------------------------------------

输出一个 JSON 对象，包含 check_id, reasoning 和 result 字段：

{{
  "check_id": "{check_id}",
  "reasoning": "简要说明判定依据",
  "result": "success 或 fail"
}}

--------------------------------------------------
注意事项
--------------------------------------------------

1. result 只能是 "success" 或 "fail"（小写），不允许其他值
2. 输出必须是合法 JSON，不要在 JSON 外添加任何文字
3. 确保输出的 check_id 与输入一致

请严格按照 Check 描述进行评估。{prompt_style_instruction}
"""

def _agenticcoding_single_prompt(
    tools: str,
    messages: str,
    check: dict[str, Any],
) -> str:
    return AGENTIC_SINGLE_TEMPLATE.format(
        tools=tools,
        messages=messages,
        category=check["category"],
        check_id=check["check_id"],
        check_description=check["description"],
        prompt_style_instruction="",
    )

def _limit_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or limit <= 0 or len(value) <= limit:
        return value
    half = max(1, (limit - 50) // 2)
    return value[:half] + "\n[content truncated]\n" + value[-half:]

def _context_strings(record: dict[str, Any], max_chars: int = 0) -> tuple[str, str]:
    # Exact serialization/truncation behavior from the pinned upstream runner.
    tools = record.get("tools", [])
    messages = []
    for original in record.get("messages", []):
        message = deepcopy(original)
        if message.get("role") == "tool":
            message["content"] = _limit_text(message.get("content", ""), 5000)
        elif message.get("role") == "assistant":
            message["content"] = _limit_text(message.get("content", ""), 50000)
            message["reasoning_content"] = _limit_text(message.get("reasoning_content", ""), 50000)
        messages.append(message)
    tools_text = "\n".join(json.dumps(item, ensure_ascii=False) for item in tools)
    messages_text = "\n".join(json.dumps(item, ensure_ascii=False) for item in messages)
    if max_chars > 0 and len(tools_text) + len(messages_text) > max_chars:
        tool_budget = min(len(tools_text), max_chars // 3)
        message_budget = max_chars - tool_budget
        tools_text = _limit_text(tools_text, tool_budget)
        messages_text = _limit_text(messages_text, message_budget)
    return tools_text, messages_text

def _extract_json(text: str) -> Any:
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        raise StrictParseError("response contains no JSON")
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}"))
    if end < start:
        raise StrictParseError("response contains incomplete JSON")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise StrictParseError("response JSON is invalid") from exc

def parse_deepresearch_output(raw: str | None) -> tuple[bool, str]:
    text = (raw or "").strip()
    lowered = text.lower()
    if re.match(r"^yes(?:\s|:|$)", lowered):
        return True, text[3:].lstrip(": ")
    if re.match(r"^no(?:\s|:|$)", lowered):
        return False, text[2:].lstrip(": ")
    if text.startswith('"') and text.endswith('"'):
        try:
            scalar = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StrictParseError("DeepResearch JSON string is invalid") from exc
        if not isinstance(scalar, str):
            raise StrictParseError("DeepResearch quoted output is not a string")
        scalar = scalar.strip()
        lowered_scalar = scalar.lower()
        if re.match(r"^yes(?:\s|:|$)", lowered_scalar):
            return True, scalar[3:].lstrip(": ")
        if re.match(r"^no(?:\s|:|$)", lowered_scalar):
            return False, scalar[2:].lstrip(": ")
        raise StrictParseError("DeepResearch JSON string has no explicit binary prefix")
    parsed = _extract_json(text)
    item = parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else parsed
    if not isinstance(item, dict):
        raise StrictParseError("DeepResearch output is not exactly one object")
    value = item.get("covered", item.get("answer"))
    normalized = str(value).strip().lower()
    if value is True or normalized in {"yes", "true", "1", "positive"}:
        return True, str(item.get("justification", item.get("reasoning", "")))
    if value is False or normalized in {"no", "false", "0", "negative"}:
        return False, str(item.get("justification", item.get("reasoning", "")))
    raise StrictParseError("DeepResearch output has no valid binary label")

def parse_agenticcoding_output(raw: str | None, expected_check_id: str) -> tuple[bool, str]:
    parsed = _extract_json((raw or "").strip())
    values = parsed if isinstance(parsed, list) else [parsed]
    matches = [
        value
        for value in values
        if isinstance(value, dict) and str(value.get("check_id")) == expected_check_id
    ]
    if len(matches) != 1:
        raise StrictParseError("AgenticCoding output does not contain exactly one expected check_id")
    result = str(matches[0].get("result", "")).strip().lower()
    if result not in {"success", "fail"}:
        raise StrictParseError("AgenticCoding output has no valid success/fail label")
    return result == "success", str(matches[0].get("reasoning", ""))

def _prompt_digest(messages: Sequence[dict[str, str]]) -> str:
    rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()

def deepresearch_messages(question: str, response: str, criterion: str) -> list[dict[str, str]]:
    """Build one English criterion request; released labels are not inputs."""
    return [{"role": "system", "content": "You are a helpful assistant that helps determine if a rubric is covered in an AI response."},
            {"role": "user", "content": _deepresearch_single_prompt(question, response, {"point": criterion})}]


def agenticcoding_messages(trajectory: dict[str, Any], check: dict[str, Any]) -> list[dict[str, str]]:
    """Build one Chinese check request from the selected upstream trajectory."""
    tools, messages = _context_strings(trajectory, max_chars=0)
    return [{"role": "user", "content": _agenticcoding_single_prompt(tools, messages, check)}]
