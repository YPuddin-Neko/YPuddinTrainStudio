"""Reviewed auxiliary assets. Revisions and content hashes are pinned together.

Sources: https://huggingface.co/SmilingWolf/wd-*-tagger-v3 (Apache-2.0) and
https://huggingface.co/deepghs/anime_head_detection (MIT); official metadata verified 2026-09-28
(the swinv2 entry first on 2026-09-11). No executable repository code is loaded: only ONNX graphs
and their label tables.
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


def _wd(repo: str, revision: str, label: str, size: int, sha256: str, **extra) -> dict:
    return {
        "role": "tagger",
        "repo": repo,
        "revision": revision,
        "license": "Apache-2.0",
        "label": label,
        # repository path -> local name, size, sha256
        "files": {
            "model.onnx": ("model.onnx", size, sha256),
            "selected_tags.csv": ("selected_tags.csv", WD_LABELS["size"], WD_LABELS["sha256"]),
        },
        "modelscope": "fireicewolf/" + repo.split("/", 1)[1],
        **extra,
    }


VISION_MODELS: dict[str, dict] = {
    "wd-eva02-large-tagger-v3": _wd(
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "b25b82a03f7282e41aa2f257a52c7583b710bd1c",
        "WD EVA02-Large v3",
        1260435999,
        "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc",
        recommended=True,
    ),
    "wd-vit-large-tagger-v3": _wd(
        "SmilingWolf/wd-vit-large-tagger-v3",
        "ae469aa2e4706a3af08d3673cf73a11d1add314c",
        "WD ViT-Large v3",
        1260645673,
        "e4c8001b000a6c98f2db10794f7c406daa79873d071d6ca924330fa053fa1845",
    ),
    "wd-swinv2-tagger-v3": _wd(
        TAGGER_REPO,
        TAGGER_REVISION,
        "WD SwinV2 v3",
        TAGGER_FILES["model.onnx"]["size"],
        TAGGER_FILES["model.onnx"]["sha256"],
    ),
    "wd-convnext-tagger-v3": _wd(
        "SmilingWolf/wd-convnext-tagger-v3",
        "d39e46de298d27340111b64965e20b8185c407e6",
        "WD ConvNeXt v3",
        394990732,
        "1b8a7abf13d9b8368267df47501d523789c4aeae66b2296ad98483239dfa32eb",
    ),
    "wd-vit-tagger-v3": _wd(
        "SmilingWolf/wd-vit-tagger-v3",
        "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
        "WD ViT v3",
        378536310,
        "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6",
    ),
    "anime-head-detector-v2": {
        "role": "head_detector",
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
