"""Values the trainer accepts but does not use exactly as typed.

Each item names the field, the grid the value is snapped to and a replacement that
would be used as written, so the interface can offer it as a one-click fix.
"""

from __future__ import annotations

from typing import Any

from ypuddin.config import TrainConfig


def _nearest(value: int, step: int) -> int:
    return max(step, (value + step // 2) // step * step)


def value_advice(cfg: TrainConfig, align: int) -> list[dict[str, Any]]:
    """Plan warnings for sizes that are rounded to the model's grid."""
    out: list[dict[str, Any]] = []
    ds = cfg.dataset
    step = ds.bucket_step or max(align, 64)
    # A bucket step off the model grid is already an error with its own message.
    if ds.resolution_mode == "bucket" and step % align == 0:
        lists = [("dataset.resolutions", ds.resolutions)] + [
            (f"dataset.sources.{index}.resolutions", source.resolutions)
            for index, source in enumerate(ds.sources)
            if source.resolutions
        ]
        for loc, resolutions in lists:
            off = [value for value in resolutions if value % step]
            if off:
                out.append(
                    {
                        "code": "dataset.resolution_step",
                        "loc": loc,
                        "msg": (
                            f"resolutions {', '.join(map(str, off))} are not multiples of the {step} px "
                            f"bucket step; buckets keep that area on a {step} px grid"
                        ),
                        "step": step,
                        "values": off,
                        "suggestion": sorted({_nearest(value, step) for value in resolutions}, reverse=True),
                    }
                )
    if ds.resolution_mode == "native" and ds.native_max_side % align:
        used = max(align, ds.native_max_side // align * align)
        out.append(
            {
                "code": "dataset.native_side_step",
                "loc": "dataset.native_max_side",
                "msg": f"longest side limit {ds.native_max_side} is not a multiple of {align}; sides stay within {used}",
                "step": align,
                "values": [ds.native_max_side],
                "suggestion": used,
            }
        )
    sampling = cfg.sampling
    if sampling.enabled:
        sizes = [(f"sampling.{key}", getattr(sampling, key)) for key in ("width", "height")]
        for index, prompt in enumerate(sampling.prompts):
            sizes += [
                (f"sampling.prompts.{index}.{key}", getattr(prompt, key))
                for key in ("width", "height")
                if getattr(prompt, key) is not None
            ]
        for loc, value in sizes:
            if value % align:
                used = max(align, value // align * align)
                out.append(
                    {
                        "code": "sampling.size_step",
                        "loc": loc,
                        "msg": f"{loc} {value} is not a multiple of {align}; previews use {used}",
                        "step": align,
                        "values": [value],
                        "suggestion": used,
                    }
                )
    return out
