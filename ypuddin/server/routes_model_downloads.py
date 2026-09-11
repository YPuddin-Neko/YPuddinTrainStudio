"""Model downloads are service tasks; completed files join the same model registry as local paths."""

from fastapi import APIRouter, Depends, Request

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
