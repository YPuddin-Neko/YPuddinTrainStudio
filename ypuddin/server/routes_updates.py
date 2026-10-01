"""Trainer source version checks."""

from fastapi import APIRouter, Depends, Request

from .trainer_install import TrainerInstallRequest, TrainerInstallStatus
from .trainer_updates import TrainerUpdates, TrainerUpdateStatus

router = APIRouter()


def updates(request: Request) -> TrainerUpdates:
    return request.app.state.trainer_updates


@router.get("/updates", response_model=TrainerUpdateStatus)
def status(service: TrainerUpdates = Depends(updates)):
    return service.status()


@router.post("/updates/check", response_model=TrainerUpdateStatus)
def check(service: TrainerUpdates = Depends(updates)):
    return service.check()


@router.get("/updates/install", response_model=TrainerInstallStatus)
def install_status(request: Request):
    return request.app.state.trainer_installer.status()


@router.post("/updates/install", response_model=TrainerInstallStatus, status_code=202)
def install_update(body: TrainerInstallRequest, request: Request):
    return request.app.state.trainer_installer.start(body)
