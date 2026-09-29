"""Validation failures as the service reports them.

Each issue names its field (``loc``) and keeps pydantic's own wording (``msg``); ``type`` and ``ctx`` carry the
error kind and its bound (``less_than`` with ``{"lt": 1}``), so a client can phrase the message itself and show a
bound in the field's unit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

# The member of a union a failed value was tried against, such as "constrained-int" or "literal['full']".
_UNION_MEMBER = re.compile(r"^(constrained-)?(int|float|str|bool|bytes|none)$|[\[\-]")


def plain_context(ctx: dict[str, Any] | None) -> dict[str, Any]:
    """The values of an error context a client can read: bounds, lengths and expected choices."""
    return {key: value for key, value in (ctx or {}).items() if isinstance(value, (str, int, float, bool))}


def validation_issues(exc: ValidationError, prefix: Iterable[str] = ()) -> list[dict[str, Any]]:
    """``{loc, msg, type, ctx}`` for each error; ``prefix`` leads the dotted field path.

    A value no member of a union accepts (a rank that is neither a positive number nor "full") is one issue, the
    first member's.
    """
    head = [str(part) for part in prefix]
    issues, unions = [], set()
    for error in exc.errors():
        parts = [part for part in error["loc"] if isinstance(part, int) or not _UNION_MEMBER.search(part)]
        loc = ".".join([*head, *(str(part) for part in parts)])
        if len(parts) < len(error["loc"]):
            if loc in unions:
                continue
            unions.add(loc)
        issue = {"loc": loc, "msg": error["msg"], "type": error["type"]}
        if ctx := plain_context(error.get("ctx")):
            issue["ctx"] = ctx
        issues.append(issue)
    return issues
