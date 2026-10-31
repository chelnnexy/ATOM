import asyncio
import json
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("jsonschema")

from atom.entrypoints.openai.sse import data_frame
from atom.structured_output.request_validation import (
    schema_from_format,
    structured_reasoning_marker,
)
from atom.structured_output.response_validation import (
    validate_chat_stream,
    validate_output,
)


@pytest.mark.parametrize(
    "name", ["$ref", "$dynamicRef", "const", "enum", "default", "examples"]
)
@pytest.mark.parametrize("position", ["properties", "$defs", "dependentSchemas"])
def test_schema_map_names_are_not_keywords(name, position):
    schema = {position: {name: {"type": "string"}}}
    fmt = {"type": "json_schema", "name": "test", "schema": schema}
    assert json.loads(schema_from_format(fmt, responses=True)) == schema
    schema[position][name] = {"$ref": "https://invalid.example/schema"}
    with pytest.raises(ValueError, match="Only local"):
        schema_from_format(fmt, responses=True)


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_literal_objects_do_not_trigger_reference_checks(keyword):
    literal = {"$ref": "https://invalid.example/literal"}
    value = [literal] if keyword in {"enum", "examples"} else literal
    fmt = {"type": "json_schema", "name": "literal", "schema": {keyword: value}}
    schema_from_format(fmt, responses=True)


@pytest.mark.parametrize(
    "prompt", ["assistant", "assistant<think></think>", "assistant<think>"]
)
@pytest.mark.parametrize("tokenized", [False, True])
def test_grammar_activation_uses_rendered_prompt(prompt, tokenized):
    from atom.entrypoints.openai.reasoning import (
        ReasoningChannel,
        prompt_starts_in_reasoning,
        prompt_tokens_start_in_reasoning,
    )

    tokenizer = NS(
        encode=lambda value, **kw: list(value.encode()),
        decode=lambda ids: bytes(ids).decode(),
    )
    opened = (
        prompt_tokens_start_in_reasoning(list(prompt.encode()), tokenizer.decode)
        if tokenized
        else prompt_starts_in_reasoning(prompt)
    )
    channel = ReasoningChannel(starts_open=opened)
    assert structured_reasoning_marker(channel, tokenizer) == (
        tuple(b"</think>") if prompt.endswith("<think>") else ()
    )


@pytest.mark.parametrize("opened", [False, True])
def test_framed_channels_fail_admission_even_without_explicit_think_close(opened):
    from atom.entrypoints.openai.reasoning import ReasoningChannel
    from atom.entrypoints.openai.reasoning_dialects import resolve_dialect

    dialect, _ = resolve_dialect("<|open|>think<|sep|>")
    with pytest.raises(ValueError, match="framed reasoning channels"):
        structured_reasoning_marker(ReasoningChannel(dialect, opened), object())


@pytest.mark.parametrize(
    "choice",
    [
        "auto",
        "required",
        {"function": {"name": "files::lookup"}},
        {"function": {"name": "lookup"}, "namespace": "files"},
    ],
)
def test_namespaced_tools_keep_dispatch_identity(choice):
    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
    )
    from atom.structured_output.request_validation import (
        structured_tool_format,
    )

    tools = [
        {"function": {"name": "lookup"}, "namespace": ns} for ns in ("search", "files")
    ]
    fmt = json.loads(structured_tool_format(DsmlV41Parser, tools, choice))
    assert [tool["name"] for tool in fmt["tools"]] == (
        ["files::lookup"]
        if isinstance(choice, dict)
        else ["search::lookup", "files::lookup"]
    )


@pytest.mark.parametrize("responses", [False, True])
def test_nested_schema(responses):
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"enum": ["red", "blue"]}},
            "ok": {"type": "boolean"},
        },
        "required": ["items", "ok"],
        "additionalProperties": False,
    }
    spec = {"name": "answer", "schema": schema}
    fmt = (
        {"type": "json_schema", **spec}
        if responses
        else {"type": "json_schema", "json_schema": spec}
    )
    compiled = schema_from_format(fmt, responses=responses)
    validate_output(compiled, '{"items":["red"],"ok":true}', "stop")
    for bad in [
        "{}",
        '{"items":["green"],"ok":true}',
        '{"items":[],"ok":true,"extra":1}',
    ]:
        with pytest.raises(RuntimeError):
            validate_output(compiled, bad, "stop")


@pytest.mark.parametrize(
    "fmt",
    [
        {},
        {"type": "other"},
        {"type": "json_schema"},
        {"type": "json_schema", "name": "x", "schema": {"type": "bogus"}},
        {
            "type": "json_schema",
            "name": "x",
            "schema": {"$ref": "https://example.com/schema"},
        },
    ],
)
def test_invalid_schema_rejected(fmt):
    with pytest.raises(ValueError):
        schema_from_format(fmt, responses=True)


def test_length_is_incomplete_not_a_false_success():
    validate_output('{"type":"object"}', '{"x":', "length")
    with pytest.raises(RuntimeError):
        validate_output('{"type":"object"}', '{"x":', "stop")


def test_tool_only_turn_has_no_text_but_text_still_must_conform():
    validate_output('{"type":"object"}', "", "tool_calls", tool_calls=True)
    with pytest.raises(RuntimeError):
        validate_output('{"type":"object"}', "not JSON", "tool_calls", tool_calls=True)


def test_tool_format_matches_the_selected_parser_and_tool_name():
    import json

    from atom.entrypoints.openai.tool_parser.deepseekv41_tool_parser import (
        DsmlV41Parser,
    )
    from atom.structured_output.request_validation import (
        structured_tool_format,
    )

    tools = [{"type": "function", "function": {"name": "weather"}}]
    assert structured_tool_format(DsmlV41Parser, tools, "none") is None
    spec = json.loads(
        structured_tool_format(
            DsmlV41Parser, tools, {"type": "function", "function": {"name": "weather"}}
        )
    )
    assert spec["prefix"] == "｜DSML｜ " and spec["section"] == "calls"
    assert spec["tools"] == [{"name": "weather", "parameters": None}]
    assert spec["required"]
    with pytest.raises(ValueError, match="declared"):
        structured_tool_format(
            DsmlV41Parser, tools, {"type": "function", "function": {"name": "unknown"}}
        )


def test_stream_failure_withholds_success_and_closes_source():
    closed = []

    async def source():
        try:
            yield data_frame(
                {"choices": [{"index": 0, "delta": {"content": "invalid"}}]}
            )
            yield data_frame(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            )
        finally:
            closed.append(True)

    async def run():
        return [
            item async for item in validate_chat_stream(source(), '{"type":"object"}')
        ]

    frames = "".join(asyncio.run(run()))
    assert "structured_output_error" in frames
    assert "finish_reason" not in frames
    assert frames.count("[DONE]") == 1
    assert closed == [True]


STRICT = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


def _function(parameters=STRICT, strict=True):
    return {
        "type": "function",
        "function": {"name": "weather", "strict": strict, "parameters": parameters},
    }


def test_strict_tool_schemas_keep_only_strict_functions():
    from atom.structured_output.request_validation import strict_tool_schemas

    tools = [
        _function(),
        {
            **_function(strict=False),
            "function": {**_function(strict=False)["function"], "name": "other"},
        },
    ]
    assert strict_tool_schemas(tools) == {"weather": STRICT}


@pytest.mark.parametrize(
    "parameters, strict, message",
    [
        ({**STRICT, "required": []}, True, "required"),
        (
            {k: v for k, v in STRICT.items() if k != "additionalProperties"},
            True,
            "additionalProperties",
        ),
        (
            {**STRICT, "properties": {"city": {"type": "string", "maxLength": 3}}},
            True,
            "maxLength",
        ),
        (
            {
                **STRICT,
                "properties": {
                    "city": {"type": "object", "properties": {"x": {"type": "integer"}}}
                },
            },
            True,
            "additionalProperties",
        ),
        ({"type": "array"}, True, "object"),
        (STRICT, "yes", "boolean"),
    ],
)
def test_strict_parameters_follow_deepseek_rules(parameters, strict, message):
    from atom.structured_output.request_validation import strict_tool_schemas

    with pytest.raises(ValueError, match=message):
        strict_tool_schemas([_function(parameters, strict)])


def test_property_named_like_a_keyword_is_not_a_keyword():
    from atom.structured_output.request_validation import strict_tool_schemas

    parameters = {
        **STRICT,
        "properties": {"maxLength": {"type": "integer"}},
        "required": ["maxLength"],
    }
    assert strict_tool_schemas([_function(parameters)])


def _call(arguments):
    return {"function": {"name": "weather", "arguments": arguments}}


def test_strict_tool_calls_are_validated_after_generation():
    from atom.structured_output.response_validation import validate_tool_calls

    schemas = {"weather": STRICT}
    validate_tool_calls(schemas, [_call('{"city": "Paris"}')], "tool_calls")
    # A truncated reply is reported as such, not as a schema violation.
    validate_tool_calls(schemas, [_call('{"city": ')], "length")
    emails = {
        "weather": {
            **STRICT,
            "properties": {"city": {"type": "string", "format": "email"}},
        }
    }
    with pytest.raises(RuntimeError, match="strict"):
        validate_tool_calls(emails, [_call('{"city": "not an address"}')], "tool_calls")
    for bad in ('{"city": 3}', '{"city": "a", "x": 1}', "not json"):
        with pytest.raises(RuntimeError, match="strict"):
            validate_tool_calls(schemas, [_call(bad)], "tool_calls")
