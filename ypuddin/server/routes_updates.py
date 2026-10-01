"""Trainer source version checks."""

from fastapi import APIRouter, Depends, Request

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
