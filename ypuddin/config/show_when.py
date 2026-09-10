"""Tiny expression language for conditional field visibility (``x-ui.show_when``).

Grammar (precedence low -> high)::

    expr    := or
    or      := and ("||" and)*
    and     := not ("&&" not)*
    not     := "!" not | cmp
    cmp     := atom (("==" | "!=" | "<" | "<=" | ">" | ">=" | "in") atom)?
    atom    := path | string | number | "true" | "false" | "null" | "[" (atom ("," atom)*)? "]" | "(" expr ")"
    path    := ident ("." ident)*

The frontend implements the same grammar; keep both in sync.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>-?\d+(?:\.\d+)?)"
    r"|(?P<str>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
    r"|(?P<op>==|!=|<=|>=|&&|\|\||[<>!()\[\],])"
    r"|(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*))"
)


class ShowWhenError(ValueError):
    pass


@dataclass(frozen=True)
class _Tok:
    kind: str
    value: Any


def _tokenize(src: str) -> list[_Tok]:
    pos, out = 0, []
    src = src.strip()
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise ShowWhenError(f"bad token at {pos}: {src[pos : pos + 10]!r}")
        pos = m.end()
        if m.group("num") is not None:
            text = m.group("num")
            out.append(_Tok("num", float(text) if "." in text else int(text)))
        elif m.group("str") is not None:
            raw = m.group("str")[1:-1]
            out.append(_Tok("str", bytes(raw, "utf-8").decode("unicode_escape")))
        elif m.group("op") is not None:
            out.append(_Tok("op", m.group("op")))
        else:
            word = m.group("path")
            if word in ("true", "false"):
                out.append(_Tok("bool", word == "true"))
            elif word == "null":
                out.append(_Tok("null", None))
            elif word == "in":
                out.append(_Tok("op", "in"))
            else:
                out.append(_Tok("path", word))
    return out


class _Parser:
    def __init__(self, tokens: list[_Tok]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, kind: str | None = None, value: Any = None) -> _Tok:
        tok = self.peek()
        if tok is None or (kind and tok.kind != kind) or (value is not None and tok.value != value):
            raise ShowWhenError(f"expected {value or kind}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> tuple:
        node = self.p_or()
        if self.peek() is not None:
            raise ShowWhenError(f"trailing tokens: {self.toks[self.i :]}")
        return node

    def p_or(self) -> tuple:
        node = self.p_and()
        while (t := self.peek()) and t.kind == "op" and t.value == "||":
            self.take()
            node = ("or", node, self.p_and())
        return node

    def p_and(self) -> tuple:
        node = self.p_not()
        while (t := self.peek()) and t.kind == "op" and t.value == "&&":
            self.take()
            node = ("and", node, self.p_not())
        return node

    def p_not(self) -> tuple:
        t = self.peek()
        if t and t.kind == "op" and t.value == "!":
            self.take()
            return ("not", self.p_not())
        return self.p_cmp()

    def p_cmp(self) -> tuple:
        left = self.p_atom()
        t = self.peek()
        if t and t.kind == "op" and t.value in ("==", "!=", "<", "<=", ">", ">=", "in"):
            self.take()
            right = self.p_atom()
            return ("cmp", t.value, left, right)
        return left

    def p_atom(self) -> tuple:
        t = self.take()
        if t.kind == "op" and t.value == "(":
            node = self.p_or()
            self.take("op", ")")
            return node
        if t.kind == "op" and t.value == "[":
            items = []
            while not ((n := self.peek()) and n.kind == "op" and n.value == "]"):
                items.append(self.p_atom())
                if (n := self.peek()) and n.kind == "op" and n.value == ",":
                    self.take()
            self.take("op", "]")
            return ("list", items)
        if t.kind in ("num", "str", "bool", "null"):
            return ("lit", t.value)
        if t.kind == "path":
            return ("path", t.value)
        raise ShowWhenError(f"unexpected token {t}")


def parse(expr: str) -> tuple:
    return _Parser(_tokenize(expr)).parse()


def _lookup(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _eval(node: tuple, data: Any) -> Any:
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "path":
        return _lookup(data, node[1])
    if kind == "list":
        return [_eval(n, data) for n in node[1]]
    if kind == "not":
        return not _eval(node[1], data)
    if kind == "and":
        return bool(_eval(node[1], data)) and bool(_eval(node[2], data))
    if kind == "or":
        return bool(_eval(node[1], data)) or bool(_eval(node[2], data))
    if kind == "cmp":
        op, left, right = node[1], _eval(node[2], data), _eval(node[3], data)
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "in":
            return right is not None and left in right
        if left is None or right is None:
            return False
        return {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right}[op]
    raise ShowWhenError(f"bad node {node!r}")


def evaluate(expr: str, data: Any) -> bool:
    """Evaluate ``expr`` against a config dict / model. Missing paths read as ``null``."""
    return bool(_eval(parse(expr), data))
