"""Reviewed auxiliary assets. Revisions and content hashes are pinned together.

Taggers match the toolbox's list: WD v3 / MOAT v2 (SmilingWolf, Apache-2.0), WD EVA02 2026 Canary
(Misaka41Z, no licence declared), PixAI v1.0 (noaione, Apache-2.0) and v0.9 (deepghs, Apache-2.0),
CL Tagger v1.02 (cella110n, Apache-2.0) and v2.01a (cella110n, gated, cl-tagger-v2-model-license-v1.0,
downloaded with the user's own Hugging Face token). The head detector is deepghs/anime_head_detection
(MIT). Official metadata verified 2026-09-28. No executable repository code is loaded: only ONNX
graphs, their external weights and their label tables.

Each model is stored under ``<models_dir>/<folder>``: ``tagger/<family>/<version>`` for taggers and
``mask/<kind>/<version>`` for mask detectors.
"""

TAGGER_ID = "wd-swinv2-tagger-v3"
TAGGER_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
TAGGER_REVISION = "627aef95638667ddcaa3ac8ae625e88ea5b02f51"
TAGGER_FILES = {
    "model.onnx": {
        "size": 467460978,
        "sha256": "e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1",
    },
    "selected_tags.csv": {
        "size": 308468,
        "sha256": "298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
    },
}

# Every wd-v3 repository ships the same 308468-byte label table.
WD_LABELS = TAGGER_FILES["selected_tags.csv"]

# How each tagger reads an image (see vision_models.prepare_tagger_input).
WD = {
    "preprocess": "wd",
    "input_size": 448,
    "labels": "csv",
    "output": "probability",
    # Label categories the model's table has, in caption order.
    "categories": ("general", "character", "rating"),
}
THRESHOLDS = {"general": 0.35, "character": 0.85}


def _tagger(
    family: str,
    version: str,
    repo: str,
    revision: str,
    label: str,
    files: dict,
    *,
    license: str = "Apache-2.0",
    modelscope: str | None = None,
    thresholds: dict | None = None,
    **extra,
) -> dict:
    return {
        "role": "tagger",
        "family": family,
        "folder": f"tagger/{family}/{version}",
        "repo": repo,
        "revision": revision,
        "license": license,
        "label": label,
        # repository path -> local name, size, sha256
        "files": files,
        "thresholds": thresholds or THRESHOLDS,
        **({"modelscope": modelscope} if modelscope else {}),
        **WD,
        **extra,
    }


def _wd(version: str, repo: str, revision: str, label: str, size: int, sha256: str, **extra) -> dict:
    files = {
        "model.onnx": ("model.onnx", size, sha256),
        "selected_tags.csv": ("selected_tags.csv", WD_LABELS["size"], WD_LABELS["sha256"]),
    }
    return _tagger(
        "wd",
        version,
        repo,
        revision,
        label,
        files,
        modelscope="fireicewolf/" + repo.split("/", 1)[1],
        **extra,
    )


# Listed as the pickers show them: each series newest first.
VISION_MODELS: dict[str, dict] = {
    "wd-eva02-tagger-2026-canary": _tagger(
        "wd",
        "eva02-2026-canary",
        "Misaka41Z/wd-eva02-tagger-2026-canary-onnx-v2",
        "0a86acfa093b33b8818667820e52fc5eccf27ff8",
        "WD EVA02 2026 Canary",
        {
            "model.onnx": (
                "model.onnx",
                1309248037,
                "fd78fbdf9390cbd163e4dd28f754a5bbf83bc7a111c4d20270f22415a0f66c95",
            ),
            "selected_tags.csv": (
                "selected_tags.csv",
                467782,
                "3f78c28ee0d50779edb320733f76aeaf4184694cbd09c631deef6889865f9178",
            ),
        },
        license="unspecified",
        # A timm export: NCHW, RGB scaled to -1..1, output already passed through a sigmoid.
        preprocess="wd_nchw",
    ),
    "wd-eva02-large-tagger-v3": _wd(
        "eva02-large-v3",
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "b25b82a03f7282e41aa2f257a52c7583b710bd1c",
        "WD EVA02-Large v3",
        1260435999,
        "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc",
        recommended=True,
    ),
    "wd-vit-large-tagger-v3": _wd(
        "vit-large-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "ae469aa2e4706a3af08d3673cf73a11d1add314c",
        "WD ViT-Large v3",
        1260645673,
        "e4c8001b000a6c98f2db10794f7c406daa79873d071d6ca924330fa053fa1845",
    ),
    "wd-swinv2-tagger-v3": _wd(
        "swinv2-v3",
        TAGGER_REPO,
        TAGGER_REVISION,
        "WD SwinV2 v3",
        TAGGER_FILES["model.onnx"]["size"],
        TAGGER_FILES["model.onnx"]["sha256"],
    ),
    "wd-convnext-tagger-v3": _wd(
        "convnext-v3",
        "SmilingWolf/wd-convnext-tagger-v3",
        "d39e46de298d27340111b64965e20b8185c407e6",
        "WD ConvNeXt v3",
        394990732,
        "1b8a7abf13d9b8368267df47501d523789c4aeae66b2296ad98483239dfa32eb",
    ),
    "wd-vit-tagger-v3": _wd(
        "vit-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
        "WD ViT v3",
        378536310,
        "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6",
    ),
    "wd-v1-4-moat-tagger-v2": _tagger(
        "wd",
        "moat-v2",
        "SmilingWolf/wd-v1-4-moat-tagger-v2",
        "8452cddf280b952281b6e102411c50e981cb2908",
        "WD MOAT v2",
        {
            "model.onnx": (
                "model.onnx",
                326197340,
                "b8cef913be4c9e8d93f9f903e74271416502ce0b4b04df0ff1e2f00df488aa03",
            ),
            "selected_tags.csv": (
                "selected_tags.csv",
                253906,
                "8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
            ),
        },
        modelscope="fireicewolf/wd-v1-4-moat-tagger-v2",
    ),
    "pixai-tagger-v1.0": _tagger(
        "pixai",
        "v1.0",
        "noaione/pixai-tagger-v1.0-onnx",
        "68e8f4f02dd56a5f40c1b7474489fa0f599dec34",
        "PixAI v1.0",
        {
            "model.onnx": (
                "model.onnx",
                2633225,
                "563f4576c2668560c20f403b957f0ec4a7bd6a2275da2c2aa7f82f898ad34e5c",
            ),
            # External weights; the graph names this file, so it must sit next to model.onnx.
            "model.onnx.data": (
                "model.onnx.data",
                1955123200,
                "4de1c25a38d1f2a2172fbcb0d2485b5f02a058b6e67bedd7bbbcfdf4de9329dc",
            ),
            "tags.json": (
                "tags.json",
                816539,
                "0d34f2078016798808dc066dc206b18fb6ce7622f64241002ecd24172a4da068",
            ),
        },
        preprocess="pixai_v1",
        input_size=1008,
        labels="pixai_json",
        categories=("general", "character", "copyright", "artist", "meta", "rating"),
        output="logits",
        # PixAI's own thresholds; general and character follow the sliders, which start at these.
        thresholds={"general": 0.17, "character": 0.27},
        fixed_thresholds={"copyright": 0.24, "artist": 0.15, "meta": 0.17, "rating": 0.41},
    ),
    "pixai-tagger-v0.9": _tagger(
        "pixai",
        "v0.9",
        "deepghs/pixai-tagger-v0.9-onnx",
        "d8cf666911a2c3d10d586d7823259192313c7eb7",
        "PixAI v0.9",
        {
            "model.onnx": (
                "model.onnx",
                1271365854,
                "a8d479098b5e23f253543c93df42391736abbb77c21c2efd3a513b9cda7b3657",
            ),
            "selected_tags.csv": (
                "selected_tags.csv",
                596868,
                "76b5dd39354a7a4d9baefb94d63b44a09a4934ee15303b7eb86c38f2128eb68a",
            ),
        },
        modelscope="deepghs/pixai-tagger-v0.9-onnx",
        preprocess="pixai",
        output_name="prediction",
        categories=("general", "character"),
        thresholds={"general": 0.3, "character": 0.85},
    ),
    "cl-tagger-v2-01a": _tagger(
        "cl",
        "v2.01a",
        "cella110n/cl_tagger_v2",
        "b57909b8e9c63f71e208a26473e7aabdf45ed6b6",
        "CL Tagger v2.01a",
        {
            "v2_01a/model.onnx": (
                "model.onnx",
                791773,
                "12581711ccf803f914129b9e87932a4cf93c80c3382b7c63305e67afdcc2a02f",
            ),
            "v2_01a/model.onnx.data": (
                "model.onnx.data",
                2211645300,
                "d9f162b7c8127790879f17fb87bc643a1c803bc17bcce44ab021fca65b2dafff",
            ),
            "v2_01a/model_vocabulary.json": (
                "model_vocabulary.json",
                14594140,
                "4966d2779825a8a4c4e46644fa8e2741824622929bdb21dbe4f9c18df2ebcf95",
            ),
        },
        license="cl-tagger-v2-model-license-v1.0",
        # Gated: the author's terms must be accepted on Hugging Face with the saved token's account.
        token_required=True,
        # The model card recommends one threshold of 0.55 for practical tagging.
        thresholds={"general": 0.55, "character": 0.55},
        preprocess="siglip2",
        input_size=384,
        labels="cl_vocabulary",
        categories=("general", "character", "copyright", "meta", "quality", "rating"),
        output="logits",
    ),
    "cl-tagger-1-02": _tagger(
        "cl",
        "v1.02",
        "cella110n/cl_tagger",
        "0b6e9b4e145b1423bfd1715119074a24a301b471",
        "CL Tagger v1.02",
        {
            "cl_tagger_1_02/model.onnx": (
                "model.onnx",
                1425860192,
                "1459e946ffc083159015e0eb26239016349ddd36c3daecfeb70bfa4ccf38b944",
            ),
            "cl_tagger_1_02/tag_mapping.json": (
                "tag_mapping.json",
                4088861,
                "9611988482f1bf9a174622c0067d0baa88fda69c0a855cc500411426e77832ec",
            ),
        },
        # Same author as v2, whose card recommends 0.55; lower values flood CL captions with tags.
        thresholds={"general": 0.55, "character": 0.55},
        preprocess="cl",
        labels="cl_mapping",
        categories=("general", "character", "copyright", "artist", "meta", "model", "quality", "rating"),
        output="logits",
    ),
    "anime-head-detector-v2": {
        "role": "head_detector",
        "family": "anime-head",
        "folder": "mask/anime-head/v2.0-s",
        "repo": "deepghs/anime_head_detection",
        "revision": "06604feee81983792a57c21081e539c0ae229833",
        "license": "MIT",
        "label": "Anime Head Detector v2.0-s",
        "files": {
            "head_detect_v2.0_s/model.onnx": (
                "model.onnx",
                44585386,
                "6679f9b71192298bbf174d82e9e5581c3237b0c3dc67deace7cdbf686b070a00",
            ),
        },
        "modelscope": "deepghs/anime_head_detection",
        "recommended": True,
    },
}
DEFAULT_TAGGER = "wd-eva02-large-tagger-v3"
DEFAULT_HEAD_DETECTOR = "anime-head-detector-v2"
# The label table each tagger reads, by format.
LABEL_FILES = {
    "csv": "selected_tags.csv",
    "pixai_json": "tags.json",
    "cl_vocabulary": "model_vocabulary.json",
    "cl_mapping": "tag_mapping.json",
}
