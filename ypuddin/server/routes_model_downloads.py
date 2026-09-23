"""Model downloads are service tasks; completed files join the same model registry as local paths."""

from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, ConfigDict

from .errors import NotFound
from .model_credentials import CredentialState, CredentialStates, CredentialUpdate, Provider
from .model_downloads import ModelDownload, ModelDownloadRequest, ModelDownloads

router = APIRouter()


def downloads(request: Request) -> ModelDownloads:
    return request.app.state.model_downloads


@router.get("/models/downloads", response_model=list[ModelDownload])
def list_downloads(service: ModelDownloads = Depends(downloads)):
    return service.list()


@router.post("/models/downloads", response_model=ModelDownload, status_code=202)
def start_download(body: ModelDownloadRequest, service: ModelDownloads = Depends(downloads)):
    return service.start(body)


@router.post("/models/downloads/{download_id}/cancel", response_model=ModelDownload)
def cancel_download(download_id: str, service: ModelDownloads = Depends(downloads)):
    return service.cancel(download_id)


class RetryDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Provider | None = None


@router.post("/models/downloads/{download_id}/retry", response_model=ModelDownload, status_code=202)
def retry_download(
    download_id: str,
    body: RetryDownloadRequest = RetryDownloadRequest(),
    service: ModelDownloads = Depends(downloads),
):
    return service.retry(download_id, provider=body.provider)


@router.get("/models/credentials", response_model=CredentialStates)
def credential_states(response: Response, service: ModelDownloads = Depends(downloads)):
    response.headers["Cache-Control"] = "no-store"
    return service.credentials.state()


@router.put("/models/credentials/{provider}", response_model=CredentialState)
def save_credential(provider: Provider, body: CredentialUpdate, service: ModelDownloads = Depends(downloads)):
    return service.credentials.save(provider, body.token.get_secret_value())


@router.delete("/models/credentials/{provider}", response_model=CredentialState)
def clear_credential(provider: Provider, service: ModelDownloads = Depends(downloads)):
    return service.credentials.save(provider, "")


class CatalogDownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["huggingface"] = "huggingface"
    mirror: Literal["official", "hf-mirror"] = "official"


class ModelCatalogEntry(BaseModel):
    id: str
    role: Literal["tagger"]
    name: str
    repo_id: str
    revision: str
    files: list[str]
    path: str
    ready: bool
    providers: list[Provider]
    license: str
    size: int
    url: str


@router.get("/models/catalog", response_model=list[ModelCatalogEntry])
def model_catalog(service: ModelDownloads = Depends(downloads)):
    return service.catalog()


@router.post("/models/catalog/{catalog_id}/download", response_model=ModelDownload, status_code=202)
def download_catalog(
    catalog_id: str, body: CatalogDownloadRequest, service: ModelDownloads = Depends(downloads)
):
    raise NotFound("catalog entry not found", code="model.catalog")
