"""Engine-specific listening settings stored with each sample request."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SpeechLanguage = Literal["zh", "en", "ja", "ko", "yue", "auto"]


class GptSovitsSampleOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    text_language: SpeechLanguage = "zh"
    reference_language: SpeechLanguage = "zh"
    top_k: int = Field(15, ge=1, le=100)
    top_p: float = Field(1.0, gt=0, le=1)
    temperature: float = Field(1.0, gt=0, le=2)
    speed: float = Field(1.0, ge=0.5, le=2)
    repetition_penalty: float = Field(1.35, ge=1, le=2)
    fragment_interval: float = Field(0.3, ge=0, le=2)
    sample_steps: Literal[4, 8, 16, 32] | None = None
    cfg_scale: float | None = Field(None, ge=0, le=2)

    def resolved(self, variant: str) -> "GptSovitsSampleOptions":
        values = self.model_dump()
        if values["sample_steps"] is None:
            values["sample_steps"] = 4 if variant == "v5turbo" else 32
        if values["cfg_scale"] is None:
            values["cfg_scale"] = 0.0 if variant == "v5turbo" else 1.3
        return self.model_validate(values)
