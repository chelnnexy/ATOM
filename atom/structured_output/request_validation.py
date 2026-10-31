# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate response formats, tool alternatives and reasoning activation."""

import json


def schema_from_format(fmt, *, responses=False):
    if fmt is None:
        return None
    if not isinstance(fmt, dict) or fmt.get("type") not in {
        "text",
        "json_object",
        "json_schema",
    }:
        raise ValueError("format.type must be text, json_object, or json_schema")
    if fmt["type"] == "text":
        return None
    if fmt["type"] == "json_object":
        return json.dumps({"type": "object"})
    spec = fmt if responses else fmt.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise ValueError("json_schema requires a schema object")  # noqa: TRY004
    if not isinstance(spec.get("name"), str) or not spec["name"].strip():
        raise ValueError("json_schema requires a nonempty name")
    schema = spec["schema"]
    from jsonschema import Draft202012Validator, SchemaError

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc

    # Resolve only document-local references. Never fetch schemas from URLs.
    def check(value):
        if not isinstance(value, dict):
            return
        for key in ("$ref", "$dynamicRef"):
            if key in value and not value[key].startswith("#"):
                raise ValueError("Only local JSON Schema references are supported")
        # Names in these maps are user data, not schema keywords.
        for key in (
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependentSchemas",
        ):
            for child in value.get(key, {}).values():
                check(child)
        for key in (
            "items",
            "additionalProperties",
            "contains",
            "propertyNames",
            "unevaluatedProperties",
            "unevaluatedItems",
            "not",
            "if",
            "then",
            "else",
        ):
            check(value.get(key))
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            for child in value.get(key, []):
                check(child)

    check(schema)
    return json.dumps(schema, ensure_ascii=False)


def structured_tool_format(parser_cls, tools, tool_choice, strict_schemas=None):
    """Describe the native tool alternative without accepting client grammars.

    A function named in ``strict_schemas`` (see :func:`strict_tool_schemas`)
    also has its arguments constrained by that schema.
    """
    if not tools or tool_choice == "none":
        return None
    from atom.entrypoints.openai.tool_parser.deepseekv4_tool_parser import DsmlParser
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    if parser_cls is None or not issubclass(parser_cls, DsmlParser):
        raise ValueError("Structured output with tools requires a DSML tool parser")
    names = [qualified_tool_name(tool) for tool in tools]
    required = tool_choice == "required" or isinstance(tool_choice, dict)
    if isinstance(tool_choice, dict):
        name = qualified_tool_name(tool_choice)
        if name not in names:
            raise ValueError("The selected tool must be declared in tools")
        names = [name]
    return json.dumps(
        {
            "prefix": parser_cls.DSML_PREFIX,
            "section": parser_cls.SECTION,
            "tools": [
                {"name": name, "parameters": (strict_schemas or {}).get(name)}
                for name in names
            ],
            "required": required,
        }
    )


def structured_reasoning_marker(channel, tokenizer):
    """Only suspend the JSON grammar when generation starts inside reasoning."""
    from atom.entrypoints.openai.reasoning_dialects import FALLBACK_DIALECT

    dialect = channel.dialect or FALLBACK_DIALECT
    if dialect.content_framing or len(dialect.end_markers) != 1:
        raise ValueError(
            "Structured output does not yet support framed reasoning channels"
        )
    if not channel.starts_open:
        return ()
    marker = dialect.think_end_marker
    tokens = tuple(tokenizer.encode(marker, add_special_tokens=False)) if marker else ()
    if not tokens:
        raise ValueError(
            "Structured thinking requires an explicit reasoning end marker"
        )
    return tokens


def strict_tool_schemas(tools):
    """``{name: parameters}`` for each function declaring ``strict: true``.

    Keyed by the qualified name, the identity calls are parsed and validated by.
    """
    from atom.entrypoints.openai.tool_parser.tool_parser import qualified_tool_name

    schemas = {}
    for tool in tools or ():
        parameters = strict_parameters(tool["function"])
        if parameters is not None:
            schemas[qualified_tool_name(tool)] = parameters
    return schemas


# Keywords DeepSeek's strict mode rejects rather than enforces.
_STRICT_UNSUPPORTED = ("minLength", "maxLength", "minItems", "maxItems")


def strict_parameters(function):
    """A strict function's ``parameters`` checked against DeepSeek's rules.

    ``None`` for a function that is not strict. Every object must list all of
    its properties in ``required`` and set ``additionalProperties: false``, so
    the arguments are exactly the declared ones, in declared order.
    """
    strict = function.get("strict", False)
    if not isinstance(strict, bool):
        message = f"{function['name']}: strict must be a boolean"
        raise ValueError(message)  # noqa: TRY004
    if not strict:
        return None
    parameters = function.get("parameters") or {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(f"{function['name']}: strict parameters must be an object")
    schema_from_format(
        {"type": "json_schema", "name": function["name"], "schema": parameters},
        responses=True,
    )
    _check_strict(function["name"], parameters)
    return parameters


def _check_strict(name, schema):
    if isinstance(schema, list):
        for child in schema:
            _check_strict(name, child)
        return
    if not isinstance(schema, dict):
        return
    for keyword in _STRICT_UNSUPPORTED:
        if keyword in schema:
            raise ValueError(f"{name}: {keyword} is not supported by strict tools")
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            raise ValueError(
                f"{name}: every strict object must set additionalProperties: false"
            )
        if set(schema.get("required", ())) != set(properties):
            raise ValueError(
                f"{name}: every property of a strict object must be required"
            )
    for key, child in schema.items():
        if key in {"properties", "$defs", "definitions"} and isinstance(child, dict):
            for subschema in child.values():
                _check_strict(name, subschema)
        elif key not in {"const", "enum", "default", "examples"}:
            _check_strict(name, child)
