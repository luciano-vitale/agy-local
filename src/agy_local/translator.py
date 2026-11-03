"""Bidirectional protocol translation between Cloud Code v1internal and OpenAI /v1/chat/completions."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Iterator, Optional


SCHEMA_TYPE_MAP = {
    "OBJECT": "object",
    "STRING": "string",
    "INTEGER": "integer",
    "NUMBER": "number",
    "BOOLEAN": "boolean",
    "ARRAY": "array",
}


def normalize_json_schema(schema: Any) -> Any:
    """Recursively normalize a Gemini FunctionDeclaration parameter schema to JSON Schema."""
    if not isinstance(schema, dict):
        return schema

    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            out["type"] = SCHEMA_TYPE_MAP.get(v.upper(), v.lower())
        elif k == "properties" and isinstance(v, dict):
            out["properties"] = {prop_k: normalize_json_schema(prop_v) for prop_k, prop_v in v.items()}
        elif k == "items":
            out["items"] = normalize_json_schema(v)
        elif k in ("anyOf", "oneOf", "allOf") and isinstance(v, list):
            out[k] = [normalize_json_schema(item) for item in v]
        elif k == "nullable":
            continue
        else:
            out[k] = v

    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}

    return out


def convert_gemini_tools_to_openai(gemini_tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Convert Gemini tools[].functionDeclarations[] into OpenAI tools[]."""
    if not gemini_tools:
        return []

    openai_tools: list[dict[str, Any]] = []
    for tool_group in gemini_tools:
        if not isinstance(tool_group, dict):
            continue
        decls = tool_group.get("functionDeclarations") or []
        for decl in decls:
            if not isinstance(decl, dict) or not decl.get("name"):
                continue
            params = normalize_json_schema(decl.get("parameters") or {"type": "object", "properties": {}})
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl["name"],
                        "description": decl.get("description", ""),
                        "parameters": params,
                    },
                }
            )
    return openai_tools


def _serialize_function_response_payload(resp_obj: Any) -> str:
    if isinstance(resp_obj, str):
        return resp_obj
    if isinstance(resp_obj, dict):
        if set(resp_obj.keys()) == {"output"} and isinstance(resp_obj["output"], str):
            return resp_obj["output"]
        if set(resp_obj.keys()) == {"error"} and isinstance(resp_obj["error"], str):
            return f"Error: {resp_obj['error']}"
    try:
        return json.dumps(resp_obj, ensure_ascii=False)
    except Exception:
        return str(resp_obj)


def convert_gemini_request_to_openai(
    inner_request: dict[str, Any],
    target_model: str,
    stream: bool = True,
) -> dict[str, Any]:
    """Translate a Cloud Code v1internal inner GenerateContentRequest to OpenAI /v1/chat/completions."""
    messages: list[dict[str, Any]] = []

    # 1. System Instruction
    sys_inst = inner_request.get("systemInstruction")
    if isinstance(sys_inst, dict):
        sys_parts = sys_inst.get("parts") or []
        sys_texts = [p.get("text", "") for p in sys_parts if isinstance(p, dict) and p.get("text")]
        if sys_texts:
            messages.append({"role": "system", "content": "\n\n".join(sys_texts)})

    # 2. Conversation Contents
    pending_call_ids_by_name: dict[str, list[str]] = {}
    call_counter = 0

    for content_item in inner_request.get("contents") or []:
        if not isinstance(content_item, dict):
            continue
        raw_role = content_item.get("role", "user")
        parts = content_item.get("parts") or []

        user_parts: list[dict[str, Any]] = []
        assistant_text_chunks: list[str] = []
        assistant_tool_calls: list[dict[str, Any]] = []
        tool_response_messages: list[dict[str, Any]] = []

        for part in parts:
            if not isinstance(part, dict):
                continue

            if "functionCall" in part and isinstance(part["functionCall"], dict):
                fc = part["functionCall"]
                call_counter += 1
                fn_name = fc.get("name", "unknown_tool")
                call_id = fc.get("id") or f"call_{call_counter}_{fn_name}"
                pending_call_ids_by_name.setdefault(fn_name, []).append(call_id)
                args_obj = fc.get("args")
                if isinstance(args_obj, str):
                    args_str = args_obj
                else:
                    args_str = json.dumps(args_obj or {}, ensure_ascii=False)
                assistant_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": args_str,
                        },
                    }
                )
                continue

            if "functionResponse" in part and isinstance(part["functionResponse"], dict):
                fr = part["functionResponse"]
                fn_name = fr.get("name", "unknown_tool")
                call_id = fr.get("id")
                if not call_id:
                    queue = pending_call_ids_by_name.get(fn_name)
                    call_id = queue.pop(0) if queue else f"call_{fn_name}"
                else:
                    queue = pending_call_ids_by_name.get(fn_name)
                    if queue and call_id in queue:
                        queue.remove(call_id)

                tool_response_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _serialize_function_response_payload(fr.get("response", {})),
                    }
                )
                continue

            if part.get("thought") is True or "thoughtSignature" in part and not part.get("text"):
                # Skip internal Gemini thought signatures or prior thought blocks in prompt history
                continue

            if "inlineData" in part and isinstance(part["inlineData"], dict):
                inline = part["inlineData"]
                mime = inline.get("mimeType", "image/png")
                b64 = inline.get("data", "")
                if b64:
                    user_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
                continue

            if "text" in part and part["text"] is not None:
                text_val = str(part["text"])
                if not text_val:
                    continue
                if raw_role == "model":
                    assistant_text_chunks.append(text_val)
                else:
                    user_parts.append({"type": "text", "text": text_val})

        # Flush collected messages for this content block in valid OpenAI order:
        if user_parts:
            if len(user_parts) == 1 and user_parts[0].get("type") == "text":
                user_content: Any = user_parts[0]["text"]
            else:
                user_content = user_parts

            # Merge with previous user text message if consecutive
            if (
                messages
                and messages[-1]["role"] == "user"
                and isinstance(messages[-1]["content"], str)
                and isinstance(user_content, str)
            ):
                messages[-1]["content"] += "\n\n" + user_content
            else:
                messages.append({"role": "user", "content": user_content})

        if assistant_text_chunks or assistant_tool_calls:
            joined_text = "".join(assistant_text_chunks)
            # If the previous message is already an assistant message and we only have tool_calls, merge
            if (
                messages
                and messages[-1]["role"] == "assistant"
                and not tool_response_messages
                and assistant_tool_calls
                and not messages[-1].get("tool_calls")
            ):
                if joined_text:
                    prev_c = messages[-1].get("content") or ""
                    messages[-1]["content"] = (prev_c + "\n" + joined_text) if prev_c else joined_text
                messages[-1]["tool_calls"] = assistant_tool_calls
            else:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": joined_text if joined_text else ("" if assistant_tool_calls else " "),
                }
                if assistant_tool_calls:
                    msg["tool_calls"] = assistant_tool_calls
                messages.append(msg)

        for tr_msg in tool_response_messages:
            messages.append(tr_msg)

    if not messages:
        messages.append({"role": "user", "content": "Hello"})

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": messages,
        "stream": stream,
    }

    openai_tools = convert_gemini_tools_to_openai(inner_request.get("tools"))
    if openai_tools:
        payload["tools"] = openai_tools

    gen_cfg = inner_request.get("generationConfig") or {}
    if isinstance(gen_cfg, dict):
        if "temperature" in gen_cfg:
            payload["temperature"] = gen_cfg["temperature"]
        if "topP" in gen_cfg:
            payload["top_p"] = gen_cfg["topP"]
        if "maxOutputTokens" in gen_cfg and isinstance(gen_cfg["maxOutputTokens"], int):
            payload["max_tokens"] = min(gen_cfg["maxOutputTokens"], 32768)

    return payload


def _format_gemini_sse_event(
    parts: list[dict[str, Any]],
    model_version: str,
    response_id: str,
    trace_id: str,
    finish_reason: Optional[str] = None,
    usage_metadata: Optional[dict[str, int]] = None,
) -> bytes:
    candidate: dict[str, Any] = {
        "content": {
            "role": "model",
            "parts": parts,
        }
    }
    if finish_reason:
        candidate["finishReason"] = finish_reason

    response_obj: dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": model_version,
        "responseId": response_id,
    }
    if usage_metadata:
        response_obj["usageMetadata"] = usage_metadata

    envelope = {
        "response": response_obj,
        "traceId": trace_id,
        "metadata": {},
    }
    return f"data: {json.dumps(envelope, ensure_ascii=False)}\n\n".encode("utf-8")


_TOOL_CALL_XML_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_WORKSPACE_CWD_RE = re.compile(r"Command Working Directory:\s*(.+)")
_ACTIVE_WORKSPACE_RE = re.compile(r"Active Workspaces:\s*\n-\s*(.+)")


def extract_workspace_root(inner_request: dict[str, Any]) -> Optional[str]:
    """Extract the active workspace directory from agy's systemInstruction."""
    sys_inst = inner_request.get("systemInstruction")
    if not isinstance(sys_inst, dict):
        return None
    for part in sys_inst.get("parts") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text") or ""
        m = _WORKSPACE_CWD_RE.search(text)
        if m:
            return m.group(1).strip()
        m2 = _ACTIVE_WORKSPACE_RE.search(text)
        if m2:
            return m2.group(1).strip()
    return None


def _normalize_tool_call_args(
    tool_name: str,
    args_dict: dict[str, Any],
    workspace_root: Optional[str],
) -> dict[str, Any]:
    """Normalize relative or @-prefixed Path arguments emitted by local models."""
    if not workspace_root or tool_name not in ("read_file", "write_file", "edit_file"):
        return args_dict

    out = dict(args_dict)
    raw_path = out.get("Path")
    if isinstance(raw_path, str) and raw_path.strip():
        cleaned = raw_path.strip()
        if cleaned.startswith("@"):
            cleaned = cleaned[1:].strip()
        if cleaned and not os.path.isabs(cleaned):
            out["Path"] = os.path.normpath(os.path.join(workspace_root, cleaned))
        else:
            out["Path"] = cleaned
    return out


def _parse_safe_json_args(raw_args: str) -> dict[str, Any]:
    raw_args = (raw_args or "").strip()
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {}


def stream_openai_to_gemini_sse(
    openai_lines: Iterator[bytes],
    model_version: str,
    workspace_root: Optional[str] = None,
) -> Iterator[bytes]:
    """Consume an OpenAI /v1/chat/completions SSE stream and yield Cloud Code v1internal SSE events."""
    response_id = f"local-{uuid.uuid4().hex[:16]}"
    trace_id = uuid.uuid4().hex[:16]

    tool_calls_by_index: dict[int, dict[str, str]] = {}
    emitted_any_text = False
    emitted_any_thought = False
    in_think_tag = False
    think_buffer = ""
    full_text_accum = ""

    prompt_tokens = 0
    completion_tokens = 0

    for raw_line in openai_lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except Exception:
            continue

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
            completion_tokens = int(usage.get("completion_tokens") or completion_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}

        # 1. Explicit reasoning_content / reasoning field from LM Studio
        reasoning_chunk = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning_chunk:
            emitted_any_thought = True
            yield _format_gemini_sse_event(
                parts=[{"text": str(reasoning_chunk), "thought": True}],
                model_version=model_version,
                response_id=response_id,
                trace_id=trace_id,
            )

        # 2. Streaming tool_calls
        tc_deltas = delta.get("tool_calls") or []
        for tc in tc_deltas:
            if not isinstance(tc, dict):
                continue
            idx = int(tc.get("index", 0))
            entry = tool_calls_by_index.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                entry["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["name"] += str(fn["name"])
            if fn.get("arguments"):
                entry["arguments"] += str(fn["arguments"])

        # 3. Content (with <think>...</think> tag detection)
        content_piece = delta.get("content")
        if content_piece:
            text_str = str(content_piece)
            think_buffer += text_str

            while think_buffer:
                if not in_think_tag:
                    if "<think>" in think_buffer:
                        before, after = think_buffer.split("<think>", 1)
                        if before:
                            emitted_any_text = True
                            full_text_accum += before
                            yield _format_gemini_sse_event(
                                parts=[{"text": before}],
                                model_version=model_version,
                                response_id=response_id,
                                trace_id=trace_id,
                            )
                        in_think_tag = True
                        think_buffer = after
                        continue
                    # Hold back if buffer ends with a partial prefix of "<think>"
                    partial_hold = False
                    for plen in range(1, len("<think>")):
                        if think_buffer.endswith("<think>"[:plen]):
                            emit_now = think_buffer[:-plen]
                            if emit_now:
                                emitted_any_text = True
                                full_text_accum += emit_now
                                yield _format_gemini_sse_event(
                                    parts=[{"text": emit_now}],
                                    model_version=model_version,
                                    response_id=response_id,
                                    trace_id=trace_id,
                                )
                            think_buffer = think_buffer[-plen:]
                            partial_hold = True
                            break
                    if partial_hold:
                        break
                    emitted_any_text = True
                    full_text_accum += think_buffer
                    yield _format_gemini_sse_event(
                        parts=[{"text": think_buffer}],
                        model_version=model_version,
                        response_id=response_id,
                        trace_id=trace_id,
                    )
                    think_buffer = ""
                else:
                    if "</think>" in think_buffer:
                        thought_text, after = think_buffer.split("</think>", 1)
                        if thought_text:
                            emitted_any_thought = True
                            yield _format_gemini_sse_event(
                                parts=[{"text": thought_text, "thought": True}],
                                model_version=model_version,
                                response_id=response_id,
                                trace_id=trace_id,
                            )
                        in_think_tag = False
                        think_buffer = after.lstrip("\r\n")
                        continue
                    partial_hold = False
                    for plen in range(1, len("</think>")):
                        if think_buffer.endswith("</think>"[:plen]):
                            emit_now = think_buffer[:-plen]
                            if emit_now:
                                emitted_any_thought = True
                                yield _format_gemini_sse_event(
                                    parts=[{"text": emit_now, "thought": True}],
                                    model_version=model_version,
                                    response_id=response_id,
                                    trace_id=trace_id,
                                )
                            think_buffer = think_buffer[-plen:]
                            partial_hold = True
                            break
                    if partial_hold:
                        break
                    emitted_any_thought = True
                    yield _format_gemini_sse_event(
                        parts=[{"text": think_buffer, "thought": True}],
                        model_version=model_version,
                        response_id=response_id,
                        trace_id=trace_id,
                    )
                    think_buffer = ""

    # Flush any residual buffer
    if think_buffer:
        if in_think_tag:
            emitted_any_thought = True
            yield _format_gemini_sse_event(
                parts=[{"text": think_buffer, "thought": True}],
                model_version=model_version,
                response_id=response_id,
                trace_id=trace_id,
            )
        else:
            emitted_any_text = True
            full_text_accum += think_buffer
            yield _format_gemini_sse_event(
                parts=[{"text": think_buffer}],
                model_version=model_version,
                response_id=response_id,
                trace_id=trace_id,
            )

    # Check if model emitted raw <tool_call>{...}</tool_call> in text without OpenAI tool_calls
    if not tool_calls_by_index and "<tool_call>" in full_text_accum:
        for idx, match in enumerate(_TOOL_CALL_XML_RE.finditer(full_text_accum)):
            try:
                tc_json = json.loads(match.group(1))
                if isinstance(tc_json, dict) and tc_json.get("name"):
                    tool_calls_by_index[idx] = {
                        "id": f"call_xml_{idx + 1}",
                        "name": str(tc_json["name"]),
                        "arguments": json.dumps(tc_json.get("arguments") or tc_json.get("args") or {}),
                    }
            except Exception:
                pass

    final_parts: list[dict[str, Any]] = []
    for idx in sorted(tool_calls_by_index.keys()):
        tc_entry = tool_calls_by_index[idx]
        if not tc_entry.get("name"):
            continue
        call_id = tc_entry.get("id") or f"call_{idx + 1}_{tc_entry['name']}"
        args_dict = _normalize_tool_call_args(
            tc_entry["name"],
            _parse_safe_json_args(tc_entry.get("arguments", "")),
            workspace_root,
        )
        final_parts.append(
            {
                "functionCall": {
                    "id": call_id,
                    "name": tc_entry["name"],
                    "args": args_dict,
                }
            }
        )

    # Ensure agy never gets an empty turn with neither output text nor tool calls
    if not emitted_any_text and not final_parts:
        final_parts.append({"text": " "})
    elif not final_parts:
        final_parts.append({"text": ""})

    usage_meta = {
        "promptTokenCount": prompt_tokens or 128,
        "candidatesTokenCount": completion_tokens or 16,
        "totalTokenCount": (prompt_tokens or 128) + (completion_tokens or 16),
    }

    yield _format_gemini_sse_event(
        parts=final_parts,
        model_version=model_version,
        response_id=response_id,
        trace_id=trace_id,
        finish_reason="STOP",
        usage_metadata=usage_meta,
    )
