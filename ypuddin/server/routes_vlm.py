"""Vision model services for image tagging: addresses, stored keys (state only) and model lists."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from . import vlm
from .model_credentials import CredentialState, CredentialUpdate, ModelCredentials, VlmProvider
from .network import ProxyPolicy

router = APIRouter()


class VlmService(BaseModel):
    id: VlmProvider
    base_url: str
    editable: bool
    key_configured: bool


class VlmServices(BaseModel):
    services: list[VlmService]


class VlmKeyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr

    @field_validator("api_key")
    @classmethod
    def valid_key(cls, value: SecretStr) -> SecretStr:
        return CredentialUpdate.valid_token(value)


class VlmModelQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str | None = Field(None, max_length=500)


class VlmModelList(BaseModel):
    models: list[str]


def credentials(request: Request) -> ModelCredentials:
    return request.app.state.model_downloads.credentials


@router.get("/vlm/services", response_model=VlmServices)
def services(response: Response, store: ModelCredentials = Depends(credentials)) -> dict:
    response.headers["Cache-Control"] = "no-store"
    configured = store.vlm_state()
    return {
        "services": [
            {
                "id": provider,
                "base_url": entry["base_url"],
                "editable": entry["editable"],
                "key_configured": configured.get(provider, False),
            }
            for provider, entry in vlm.PROVIDERS.items()
        ]
    }


@router.put("/vlm/services/{provider}/key", response_model=CredentialState)
def save_key(provider: VlmProvider, body: VlmKeyUpdate, store: ModelCredentials = Depends(credentials)):
    return store.save_vlm_key(provider, body.api_key.get_secret_value())


@router.delete("/vlm/services/{provider}/key", response_model=CredentialState)
def clear_key(provider: VlmProvider, store: ModelCredentials = Depends(credentials)):
    return store.save_vlm_key(provider, "")


@router.post("/vlm/services/{provider}/models", response_model=VlmModelList)
def models(
    provider: VlmProvider,
    body: VlmModelQuery,
    request: Request,
    store: ModelCredentials = Depends(credentials),
) -> dict:
    policy = ProxyPolicy.from_context(request.app.state.ctx)
    return {"models": vlm.list_models(provider, body.base_url, store.vlm_key(provider), policy=policy)}
