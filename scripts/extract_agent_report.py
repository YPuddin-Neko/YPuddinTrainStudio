"""Extract the final assistant text message from a Claude agent JSONL transcript."""

import json
import sys


def extract(path: str) -> str:
    last_text = ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "assistant":
                continue
            msg = rec.get("message") or {}
            parts = []
            for block in msg.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = "\n".join(p for p in parts if p).strip()
            if len(text) > 2000:  # only keep substantial reports
                last_text = text
    return last_text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    text = extract(src)
    if not text:
        print(f"no report found in {src}", file=sys.stderr)
        sys.exit(1)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")
    print(f"wrote {len(text)} chars -> {dst}")
