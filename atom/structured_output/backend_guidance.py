# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strict JSON Schema compilation: unsupported constraints must never disappear."""

import json

import torch


class GrammarMatcher:
    def __init__(self, compiled):
        self.matcher = compiled.deep_copy()

    def copy(self):
        return GrammarMatcher(self.matcher)

    def accept_token(self, token):
        return self.matcher.consume_token(int(token))

    def is_terminated(self):
        return self.matcher.is_stopped()

    def fill_next_token_bitmask(self, mask):
        from llguidance.torch import fill_next_token_bitmask

        fill_next_token_bitmask(self.matcher, mask)
        if self.matcher.is_error():
            raise ValueError(self.matcher.get_error())


def allocate_token_bitmask(batch_size, vocab_size):
    return torch.full((batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32)


class GrammarCompiler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compile_json_schema(self, schema, tool_format=None):
        """Compile a JSON Schema answer, DSML tool calls, or either of them.

        ``schema`` is ``None`` when only strict tools constrain the reply.
        """
        from llguidance import LLMatcher

        from .dsml_grammar import dsml_grammar
        from .schema_normalization import normalize_schema

        value = None
        if schema is not None:
            value = normalize_schema(json.loads(schema))
            if isinstance(value, dict):
                # This is an API JSON Schema, not a backend configuration
                # object. In particular, clients must not enable
                # lenient/coerce_one_of or replace JSON punctuation through an
                # x-guidance annotation.
                value.pop("x-guidance", None)
        if tool_format:
            # Only the response text is constrained by response_format; a
            # function call uses the parser's native envelope, its arguments
            # constrained only for strict tools.
            lark = dsml_grammar(json.loads(tool_format), value, normalize_schema)
            grammar = {"grammars": [{"lark_grammar": lark}]}
        else:
            grammar = LLMatcher.grammar_from_json_schema(
                value, overrides={"lenient": False, "coerce_one_of": False}
            )
        matcher = LLMatcher(self.tokenizer, grammar, log_level=0)
        if matcher.is_error():
            raise ValueError(matcher.get_error())
        warnings = matcher.get_grammar_warnings()
        if warnings:
            raise ValueError("; ".join(warnings))
        return matcher
