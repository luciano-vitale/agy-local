"""Unit tests for Cloud Code v1internal <-> OpenAI protocol translation."""

from __future__ import annotations

import json
import unittest

from agy_local.translator import (
    _normalize_tool_call_args,
    convert_gemini_request_to_openai,
    convert_gemini_tools_to_openai,
    extract_workspace_root,
    normalize_json_schema,
    stream_openai_to_gemini_sse,
)


class TranslatorTest(unittest.TestCase):
    def test_normalize_json_schema_converts_uppercase_types(self) -> None:
        raw = {
            "type": "OBJECT",
            "properties": {
                "Path": {"type": "STRING", "description": "File path"},
                "StartLine": {"type": "INTEGER", "nullable": True},
                "Tags": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["Path"],
        }
        norm = normalize_json_schema(raw)
        self.assertEqual(norm["type"], "object")
        self.assertEqual(norm["properties"]["Path"]["type"], "string")
        self.assertEqual(norm["properties"]["StartLine"]["type"], "integer")
        self.assertNotIn("nullable", norm["properties"]["StartLine"])
        self.assertEqual(norm["properties"]["Tags"]["type"], "array")
        self.assertEqual(norm["properties"]["Tags"]["items"]["type"], "string")

    def test_convert_gemini_request_with_tool_call_and_response(self) -> None:
        inner = {
            "systemInstruction": {"parts": [{"text": "You are Antigravity."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Read AGENTS.md"}]},
                {
                    "role": "model",
                    "parts": [
                        {"text": "Thinking...", "thought": True},
                        {
                            "functionCall": {
                                "id": "call_1",
                                "name": "read_file",
                                "args": {"Path": "/repo/AGENTS.md"},
                            }
                        },
                    ],
                },
                {
                    # agy sends functionResponse under role="model"
                    "role": "model",
                    "parts": [
                        {
                            "functionResponse": {
                                "id": "call_1",
                                "name": "read_file",
                                "response": {"output": "# Agents doc"},
                            }
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {"Path": {"type": "STRING"}},
                                "required": ["Path"],
                            },
                        }
                    ]
                }
            ],
        }

        openai_req = convert_gemini_request_to_openai(inner, target_model="qwen3.6-35b-a3b", stream=True)
        self.assertEqual(openai_req["model"], "qwen3.6-35b-a3b")
        msgs = openai_req["messages"]
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0], {"role": "system", "content": "You are Antigravity."})
        self.assertEqual(msgs[1], {"role": "user", "content": "Read AGENTS.md"})
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[2]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(msgs[3], {"role": "tool", "tool_call_id": "call_1", "content": "# Agents doc"})
        self.assertEqual(len(openai_req["tools"]), 1)

    def test_stream_openai_to_gemini_sse_handles_think_tags_and_tool_calls(self) -> None:
        raw_sse_lines = [
            b'data: {"choices":[{"delta":{"content":"<think>Step 1"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"</think>Hello world"}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\\"Path\\": "}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"/a.txt\\"}"}}]}}],"usage":{"prompt_tokens":50,"completion_tokens":12}}\n',
            b"data: [DONE]\n",
        ]

        events = [
            json.loads(chunk.decode("utf-8").strip()[5:].strip())
            for chunk in stream_openai_to_gemini_sse(iter(raw_sse_lines), model_version="gemma-4-12b")
        ]

        thoughts = []
        texts = []
        fn_calls = []
        for ev in events:
            for part in ev["response"]["candidates"][0]["content"]["parts"]:
                if part.get("thought"):
                    thoughts.append(part["text"])
                elif "functionCall" in part:
                    fn_calls.append(part["functionCall"])
                elif part.get("text"):
                    texts.append(part["text"])

        self.assertEqual("".join(thoughts), "Step 1")
        self.assertEqual("".join(texts), "Hello world")
        self.assertEqual(len(fn_calls), 1)
        self.assertEqual(fn_calls[0], {"id": "c1", "name": "read_file", "args": {"Path": "/a.txt"}})
        self.assertEqual(events[-1]["response"]["candidates"][0]["finishReason"], "STOP")
        self.assertEqual(events[-1]["response"]["usageMetadata"]["promptTokenCount"], 50)

    def test_extract_workspace_root_and_normalize_relative_tool_paths(self) -> None:
        inner = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "Active Workspaces:\n- /Users/test/project\nCommand Working Directory: /Users/test/project\n"
                    }
                ]
            }
        }
        ws = extract_workspace_root(inner)
        self.assertEqual(ws, "/Users/test/project")
        norm = _normalize_tool_call_args("read_file", {"Path": "@GLOSSARY.md"}, ws)
        self.assertEqual(norm["Path"], "/Users/test/project/GLOSSARY.md")


if __name__ == "__main__":
    unittest.main()
