"""Regenerate the API specification and training schemas from the code."""

import argparse
import json
import tempfile
from pathlib import Path

from ypuddin.config import TrainConfig
from ypuddin.server import create_app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openapi-only", action="store_true", help="Export only docs/api/openapi.json."
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1] / "docs" / "api"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        app = create_app(temporary, frontend_dist="/nonexistent")
        try:
            (root / "openapi.json").write_text(
                json.dumps(app.openapi(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        finally:
            app.state.ctx.db.close()
    if not args.openapi_only:
        schema = json.dumps(TrainConfig.json_schema(), indent=2, ensure_ascii=False)
        (root / "train-schema.example.json").write_text(schema, encoding="utf-8")
        # The bundled form must carry the same optimizer capabilities and UI metadata
        # as GET /schema/train, not the lower-level Pydantic model_json_schema output.
        (root.parents[1] / "frontend/src/schema/train-schema.json").write_text(schema, encoding="utf-8")
    print("exported", root)


if __name__ == "__main__":
    main()
