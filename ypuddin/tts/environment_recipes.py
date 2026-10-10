"""Dependencies of the fixed training and sampling entry points."""

from __future__ import annotations

from .core import UPSTREAM_REVISION as VOX_REVISION
from .gpt_sovits.core import UPSTREAM_REVISION as GSV_REVISION

RECIPE_REVISION = "2026-10-10.2"
SOURCES = {
    "voxcpm1.5": {"url": "https://github.com/OpenBMB/VoxCPM.git", "revision": VOX_REVISION},
    "gpt-sovits-v5": {"url": "https://github.com/RVC-Boss/GPT-SoVITS.git", "revision": GSV_REVISION},
}
_REQUIREMENTS = {
    # Sampling disables denoising and text normalization; the official Gradio application is not imported.
    "voxcpm1.5": (
        "torch>=2.5", "torchaudio>=2.5", "torchcodec", "numpy>=1.26", "transformers>=4.36.2",
        "einops>=0.7", "pydantic>=2.6", "datasets>=3,<4", "soundfile>=0.12", "librosa>=0.10",
        "matplotlib>=3.8", "argbind", "tensorboardX>=2.6", "safetensors>=0.4", "sentencepiece>=0.2",
        "huggingface-hub>=0.25", "PyYAML>=6", "tqdm>=4.66", "packaging>=23",
    ),
    # tools.my_utils imports Gradio even when running dataset preparation without its WebUI.
    "gpt-sovits-v5": (
        "torch>=2.4", "torchaudio", "numpy>=1.26,<2", "scipy", "tensorboard>=2.16",
        # The GPT worker's Gloo launch failed on the tested Windows 2.8.0 build.
        "torch!=2.8.0; sys_platform == 'win32'",
        "librosa==0.10.2", "resampy>=0.4.2,<0.5", "numba", "pytorch-lightning>=2.4,<3",
        "torchmetrics<=1.5", "gradio>=4.44,<5", "ffmpeg-python", "soundfile>=0.12",
        "pandas>=2,<3", "matplotlib>=3.8", "einops>=0.8", "transformers>=4.51,<5",
        "peft<0.18", "pydantic>=2.6,<=2.10.6", "cn2an", "pypinyin", "pyopenjtalk>=0.4.1",
        "g2p_en", "sentencepiece>=0.2", "PyYAML>=6", "psutil", "jieba_fast", "jieba",
        "split-lang", "fast_langdetect>=0.3.1", "wordsegment", "rotary_embedding_torch",
        "ToJyutping", "g2pk2", "ko_pron", "opencc", "x_transformers", "safetensors>=0.4",
        "python_mecab_ko; sys_platform != 'win32'", "onnxruntime", "tqdm>=4.66", "packaging>=23",
        "imageio-ffmpeg>=0.5",
    ),
}

TEXT_RESOURCES = (
    {"name": "nltk_data.zip", "kind": "zip", "size": 9924448,
     "sha256": "eb3ec26ace3f9ccbb08a6d333e26f0941c47e230ece0717dc992bdb7e99808dd",
     "url": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/0c47645e02a7bc3688d7b263b0042c81e3cd82cd/nltk_data.zip"},
    {"name": "open_jtalk_dic_utf_8-1.11.tar.gz", "kind": "tar", "size": 23646843,
     "sha256": "fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a",
     "url": "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/0c47645e02a7bc3688d7b263b0042c81e3cd82cd/open_jtalk_dic_utf_8-1.11.tar.gz"},
)

# The fixed upstream's Windows packaging points to these two CPython 3.12 native wheels.
WINDOWS_GSV_WHEELS = (
    {"name": "pyopenjtalk-0.4.1-cp312-cp312-win_amd64.whl", "size": 1057544,
     "sha256": "54199f8427ad91528e3adb03d3910ad79af109420cac819548b16592ce3960f0",
     "url": "https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/f4121685cb6ba616d618607ddaf940d233e79953/wheels/cu128/pyopenjtalk-0.4.1-cp312-cp312-win_amd64.whl"},
    {"name": "jieba_fast-0.53-cp312-cp312-win_amd64.whl", "size": 7630585,
     "sha256": "d59bc49c0bf1be2f20d0066ffa53e18bccba39f310bde7e470564d0165dc250b",
     "url": "https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/f4121685cb6ba616d618607ddaf940d233e79953/wheels/cu128/jieba_fast-0.53-cp312-cp312-win_amd64.whl"},
)

# FFmpeg's download page links this Windows build provider. TorchCodec 0.10 supports FFmpeg 4–8.
WINDOWS_SHARED_FFMPEG = {
    "name": "ffmpeg-8.0.1-full_build-shared.zip", "kind": "zip", "size": 94371734,
    "sha256": "e4a40b46e3a3f3e5f2ed28352bd1fd6c733ae3808a8743682ef7f0d4e20c7d51",
    "url": "https://github.com/GyanD/codexffmpeg/releases/download/8.0.1/ffmpeg-8.0.1-full_build-shared.zip",
}

# Only the FFmpeg dylibs are used; no Python extension is installed from this wheel.
MACOS_ARM64_FFMPEG = {
    "name": "av-16.1.0-cp312-cp312-macosx_14_0_arm64.whl", "size": 21481147,
    "sha256": "ae3fb658eec00852ebd7412fdc141f17f3ddce8afee2d2e1cf366263ad2a3b35",
    "url": "https://files.pythonhosted.org/packages/b6/17/ffb940c9e490bf42e86db4db1ff426ee1559cd355a69609ec1efe4d3a9eb/av-16.1.0-cp312-cp312-macosx_14_0_arm64.whl",
}
MACOS_FFMPEG_ALIASES = {
    "libavutil.60.dylib": "libavutil.60.8.100.dylib",
    "libavcodec.62.dylib": "libavcodec.62.11.100.dylib",
    "libavformat.62.dylib": "libavformat.62.3.100.dylib",
    "libavdevice.62.dylib": "libavdevice.62.1.100.dylib",
    "libavfilter.11.dylib": "libavfilter.11.4.100.dylib",
    "libswscale.9.dylib": "libswscale.9.1.100.dylib",
    "libswresample.6.dylib": "libswresample.6.1.100.dylib",
}
MACOS_FFMPEG_LIBRARIES = frozenset({
    *MACOS_FFMPEG_ALIASES.values(), "libSvtAv1Enc.3.1.2.dylib", "libvorbisenc.2.dylib", "libdav1d.7.dylib",
    "libx264.165.dylib", "libmp3lame.0.dylib", "libwebp.7.1.10.dylib", "libopencore-amrnb.0.dylib",
    "libwebpmux.3.1.1.dylib", "libvpx.11.dylib", "libsharpyuv.0.1.1.dylib", "libvorbis.0.dylib",
    "libx265.215.dylib", "libopenh264.7.dylib", "libaom.3.13.1.dylib", "libspeex.1.dylib",
    "libopencore-amrwb.0.dylib", "libogg.0.dylib", "libopus.0.dylib",
})


def requirements(engine: str) -> list[str]:
    if engine not in _REQUIREMENTS:
        raise ValueError("不支持的语音训练引擎。")
    return list(_REQUIREMENTS[engine])


def source(engine: str) -> dict[str, str]:
    if engine not in SOURCES:
        raise ValueError("不支持的语音训练引擎。")
    return dict(SOURCES[engine])
