"""One-to-one image companions, matched by relative path without its extension."""

from pathlib import Path

from ypuddin.config import DatasetSourceConfig

from .index import ImageRecord, IndexDB, iter_images, probe_image, scan_sources


def paired_paths(source: DatasetSourceConfig, field: str) -> dict[str, str]:
    root = Path(source.path).expanduser()
    peer = Path(getattr(source, field) or "").expanduser()
    if not getattr(source, field) or not peer.is_dir():
        raise ValueError("请指定存在的配对图片目录。")
    excluded = {str(Path(value).with_suffix("")) for value in source.excluded_files}
    prefixes = tuple(value + "/" for value in source.excluded_dirs)

    def inventory(directory: Path) -> dict[str, Path]:
        result = {}
        resolved = directory.resolve()
        for path in iter_images(directory):
            relative = path.relative_to(directory).as_posix()
            key = Path(relative).with_suffix("").as_posix()
            if key in excluded or relative.startswith(prefixes):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(resolved):
                raise ValueError(f"配对图片必须位于所属目录内：{relative}")
            if key in result:
                raise ValueError(f"存在重复的配对文件名：{key}")
            result[key] = path
        return result

    targets, companions = inventory(root), inventory(peer)
    missing = targets.keys() - companions.keys()
    extra = companions.keys() - targets.keys()
    if missing:
        raise ValueError(f"缺少配对图片：{min(missing)}")
    if extra:
        raise ValueError(f"配对目录中存在未匹配图片：{min(extra)}")
    for key, target in targets.items():
        target_size, companion_size = probe_image(target)[:2], probe_image(companions[key])[:2]
        if field == "pair_path" and target_size != companion_size:
            raise ValueError(f"正负图片的原始尺寸必须相同：{key}")
    return {str(path): str(companions[key]) for key, path in targets.items()}


def paired_records(
    source: DatasetSourceConfig, field: str, *, index_db: IndexDB | None = None,
) -> dict[str, ImageRecord]:
    paths = paired_paths(source, field)
    peer = source.model_copy(update={
        "path": getattr(source, field), "pair_path": None, "reference_dir": None,
        "excluded_files": [], "excluded_dirs": [],
    })
    records = {record.path: record for record in scan_sources([peer], index_db=index_db)}
    missing = set(paths.values()) - records.keys()
    if missing:
        raise ValueError(f"无法读取配对图片：{Path(min(missing)).name}")
    return {target: records[companion] for target, companion in paths.items()}
