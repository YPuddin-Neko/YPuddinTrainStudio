"""Central access-key settings. Public responses contain configured state only."""

from fastapi import APIRouter, Depends, Request, Response

from .model_credentials import (
    AccessCredentialStates,
    AccessProvider,
    CredentialState,
    CredentialUpdate,
    DanbooruCredentialUpdate,
    GelbooruCredentialUpdate,
    ModelCredentials,
    Provider,
)

router = APIRouter()


def credentials(request: Request) -> ModelCredentials:
    return request.app.state.model_downloads.credentials


@router.get("/credentials", response_model=AccessCredentialStates)
def credential_states(response: Response, store: ModelCredentials = Depends(credentials)):
    response.headers["Cache-Control"] = "no-store"
    return store.access_state()


@router.put("/credentials/danbooru", response_model=CredentialState)
def save_danbooru(body: DanbooruCredentialUpdate, store: ModelCredentials = Depends(credentials)):
    return store.save_site("danbooru", body)


@router.put("/credentials/gelbooru", response_model=CredentialState)
def save_gelbooru(body: GelbooruCredentialUpdate, store: ModelCredentials = Depends(credentials)):
    return store.save_site("gelbooru", body)


@router.put("/credentials/{provider}", response_model=CredentialState)
def save_model_token(
    provider: Provider, body: CredentialUpdate, store: ModelCredentials = Depends(credentials)
):
    return store.save(provider, body.token.get_secret_value())


@router.delete("/credentials/{provider}", response_model=CredentialState)
def clear_credential(provider: AccessProvider, store: ModelCredentials = Depends(credentials)):
    return store.clear(provider)
