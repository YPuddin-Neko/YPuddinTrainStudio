"""Regenerate docs/api/openapi.json and docs/api/train-schema.example.json from the code."""

import json
import tempfile
from pathlib import Path

from ypuddin.config import TrainConfig
from ypuddin.server import create_app

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
schema = json.dumps(TrainConfig.json_schema(), indent=2, ensure_ascii=False)
(root / "train-schema.example.json").write_text(schema, encoding="utf-8")
# The bundled form must carry the same optimizer capabilities and UI metadata
# as GET /schema/train, not the lower-level Pydantic model_json_schema output.
(root.parents[1] / "frontend/src/schema/train-schema.json").write_text(schema, encoding="utf-8")
print("exported", root)
