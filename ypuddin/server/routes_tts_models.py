"""Speech model packages share the application's download credentials and storage."""

from fastapi import APIRouter, Depends, Request

from ypuddin.tts.model_download_models import (
    TtsInstalledModel,
    TtsModelDownload,
    TtsModelDownloadRequest,
    TtsModelPackage,
)

from .models import ApiErrorResponse
from .tts_model_downloads import TtsModelDownloads

router = APIRouter(responses={code: {"model": ApiErrorResponse} for code in (400, 403, 404, 409, 422)})


def downloads(request: Request) -> TtsModelDownloads:
    return request.app.state.model_downloads.tts


@router.get("/tts/models/catalog", response_model=list[TtsModelPackage])
def model_catalog(service: TtsModelDownloads = Depends(downloads)):
    return service.catalog()


@router.get("/tts/models", response_model=list[TtsInstalledModel])
def installed_models(service: TtsModelDownloads = Depends(downloads)):
    return service.installed()


@router.get("/tts/models/downloads", response_model=list[TtsModelDownload])
def list_downloads(service: TtsModelDownloads = Depends(downloads)):
    return service.list()


@router.post("/tts/models/downloads", response_model=TtsModelDownload, status_code=202)
def start_download(body: TtsModelDownloadRequest, service: TtsModelDownloads = Depends(downloads)):
    return service.start(body.package_id, provider=body.provider)


@router.get("/tts/models/downloads/{download_id}", response_model=TtsModelDownload)
def get_download(download_id: str, service: TtsModelDownloads = Depends(downloads)):
    return service.get(download_id)


@router.post("/tts/models/downloads/{download_id}/cancel", response_model=TtsModelDownload)
def cancel_download(download_id: str, service: TtsModelDownloads = Depends(downloads)):
    return service.cancel(download_id)


@router.post("/tts/models/downloads/{download_id}/retry", response_model=TtsModelDownload, status_code=202)
def retry_download(download_id: str, service: TtsModelDownloads = Depends(downloads)):
    return service.retry(download_id)
