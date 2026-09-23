from pathlib import Path

from fastapi.testclient import TestClient

from ypuddin.server.app import create_app
from ypuddin.server.model_layout import model_folder


def test_category_first_layout_and_shared_vae(tmp_path):
    assert model_folder(tmp_path, 'anima', 'dit') == tmp_path/'diffusion_models/anima'
    assert model_folder(tmp_path, 'sdxl', 'text_encoder_2') == tmp_path/'text_encoders/sdxl'
    assert model_folder(tmp_path, 'anima', 'vae') == model_folder(tmp_path, 'krea2', 'vae') == tmp_path/'vae/shared'


def test_browser_uses_custom_storage_root_and_existing_category(tmp_path):
    import json
    root = tmp_path/'data'
    root.mkdir()
    models = root/'custom-models'
    for category in ('diffusion_models', 'vae', 'text_encoders'):
        (models/category/'shared').mkdir(parents=True)
    (root/'settings.json').write_text(json.dumps({'paths': {'models_dir': str(models)}}))
    with TestClient(create_app(data_root=root)) as client:
        for kind, category in [('dit','diffusion_models'),('vae','vae'),('text_encoder','text_encoders'),('text_encoder_2','text_encoders')]:
            response = client.get('/api/models/browse-root', params={'kind':kind})
            assert response.status_code == 200, response.text
            assert Path(response.json()['path']) == models/category
        assert client.get('/api/models/browse-root', params={'kind':'../../etc'}).status_code == 400
