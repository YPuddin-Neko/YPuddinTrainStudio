"""Publish the typed TTS payload carried by the existing event stream."""

from ypuddin.tts.source_models import TtsSourceChanged


def install(app) -> None:
    original = app.openapi

    def openapi():
        schema = original()
        event = TtsSourceChanged.model_json_schema(ref_template="#/components/schemas/{model}")
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for name, model in event.pop("$defs", {}).items():
            components.setdefault(name, model)
        components["TtsSourceChanged"] = event
        schema.setdefault("x-event-payloads", {})["tts.source.changed"] = {
            "$ref": "#/components/schemas/TtsSourceChanged"
        }
        return schema

    app.openapi = openapi
