# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate final JSON answers and completed Chat streams."""

import json

_TRUNCATED = {"length", "max_tokens", "max_new_tokens"}


def validate_output(schema, text, finish_reason, *, tool_calls=False):
    if schema is None or finish_reason in _TRUNCATED:
        return
    if tool_calls and not text.strip():
        return  # A function-only turn has no response text to constrain.
    from jsonschema import Draft202012Validator

    try:
        value = json.loads(
            text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
        )
        Draft202012Validator(json.loads(schema)).validate(value)
    except Exception as exc:
        raise RuntimeError(
            "Generated output did not satisfy the requested JSON Schema"
        ) from exc


def validate_tool_calls(strict_schemas, tool_calls, finish_reason):
    """Each call to a strict function must carry arguments its schema accepts."""
    if not strict_schemas or finish_reason in _TRUNCATED:
        return
    from jsonschema import Draft202012Validator

    for call in tool_calls or ():
        name = call["function"]["name"]
        if name not in strict_schemas:
            continue
        try:
            arguments = json.loads(call["function"]["arguments"])
            Draft202012Validator(
                strict_schemas[name],
                format_checker=Draft202012Validator.FORMAT_CHECKER,
            ).validate(arguments)
        except Exception as exc:
            raise RuntimeError(
                f"Tool call {name} did not satisfy its strict parameters schema"
            ) from exc


async def validate_chat_stream(source, schema, strict_schemas=None):
    from atom.entrypoints.openai.sse import data_frame, iter_sse_data

    texts, calls = {}, {}
    try:
        async for frame in source:
            for data in iter_sse_data(frame):
                if data == "[DONE]":
                    continue
                payload = json.loads(data)
                for choice in payload.get("choices", []):
                    index = choice.get("index", 0)
                    delta = choice.get("delta", {})
                    texts.setdefault(index, []).append(delta.get("content") or "")
                    for call in delta.get("tool_calls") or ():
                        fn = call.get("function") or {}
                        acc = calls.setdefault(index, {}).setdefault(
                            call.get("index", 0),
                            {"function": {"name": "", "arguments": ""}},
                        )["function"]
                        acc["name"] += fn.get("name") or ""
                        acc["arguments"] += fn.get("arguments") or ""
                    if choice.get("finish_reason"):
                        validate_output(
                            schema,
                            "".join(texts[index]),
                            choice["finish_reason"],
                            tool_calls=choice["finish_reason"] == "tool_calls",
                        )
                        validate_tool_calls(
                            strict_schemas,
                            calls.get(index, {}).values(),
                            choice["finish_reason"],
                        )
            yield frame
    except RuntimeError as exc:
        yield data_frame(
            {"error": {"type": "structured_output_error", "message": str(exc)}}
        )
        yield "data: [DONE]\n\n"
    finally:
        await source.aclose()
