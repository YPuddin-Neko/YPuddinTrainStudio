"""Execute official stages in an isolated task directory and publish paired exports."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import runpy
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

from ..events import Events
from .config import parse_config
from .core import (
    asset_paths,
    audio_info,
    file_digest,
    manifest_rows,
    require_text_assets,
    safe_checkpoint,
    text_asset_files,
    text_asset_roots,
)

log = logging.getLogger("ypuddin.tts")


class _SovitsLogFilter(logging.Filter):
    def filter(self, record):
        if type(record.msg).__name__ == "HParams" and type(record.msg).__module__ == "utils":
            record.levelno, record.levelname = logging.DEBUG, "DEBUG"
        elif (isinstance(record.msg, list) and len(record.msg) == 3
              and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in record.msg)):
            loss, step, rate = record.msg
            record.msg = "SoVITS step %d: loss=%.6f, lr=%.6e"
            record.args = (step, loss, rate)
        return True


def training_event(events, stage, step, metrics, epoch=None, *, offset=0):
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("GPT-SoVITS 训练指标出现非有限数值。")
    rate = metrics.get("lr", metrics.get("learning_rate"))
    loss = metrics.get("total_loss", metrics.get("loss/g/total"))
    fields = {"stage": stage, "step": offset + int(step), "epoch": epoch, "loss": loss,
              "lr": {stage: rate} if rate is not None else None, "metrics": {**metrics, "stage_step": int(step)}}
    events.emit("step", **fields)
    if rate is not None:
        log.info("%s step %d: lr=%.6e%s", stage, step, rate, f", loss={loss:.6f}" if loss is not None else "")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _CudaMask(str):
    """Keep the supervisor's device identity through the numeric-GPU CLI parser."""

    def __new__(cls, value, owner=None):
        instance = super().__new__(cls, value)
        instance.owner = owner
        return instance

    def replace(self, old, new, count=-1):
        if old == "-" and new == ",":
            if self.owner is not None:
                self.owner.gpu_numbers = str(self)
            return str(self)
        return super().replace(old, new, count)


def workspace(config, target):
    """Copy source assets; upstream caches and temporary files belong to this task."""
    target = Path(target)
    target.mkdir()
    upstream = Path(config.trainer_path)
    result = subprocess.run(["git", "-C", str(upstream), "ls-files", "-z"], capture_output=True, check=True)
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        source = upstream / relative
        if not source.is_file():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(upstream.resolve()):
            raise ValueError("上游源码不能包含目录外的文件重定向。")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    destinations = {"g2pw": "GPT_SoVITS/text/G2PWModel",
                    "fast_langdetect": "GPT_SoVITS/pretrained_models/fast_langdetect"}
    for key, root in text_asset_roots(config).items():
        destination = target / destinations[key]
        # G2PW downloads when its directory is absent; the language detector also needs its cache directory.
        destination.mkdir(parents=True, exist_ok=True)
        for name, source in text_asset_files(root).items():
            copied = destination / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
    vocoder = target / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"
    vocoder.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(asset_paths(config)["vocoder"], vocoder)
    return target


def stage_environment(config, work, prepared):
    from .core import worker_environment

    env = worker_environment(config)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[3]), str(work),
                                        str(work / "GPT_SoVITS"), str(work / "GPT_SoVITS/BigVGAN")])
    paths = asset_paths(config)
    env.update(version=config.variant, opt_dir=str(prepared), exp_name="studio", i_part="0", all_parts="1",
               is_half=str(config.sovits.precision == "fp16"), inp_text=str(prepared / "dataset.list"),
               inp_wav_dir="", bert_pretrained_dir=paths["bert"], cnhubert_base_dir=paths["hubert"],
               pretrained_s2G=paths["sovits"], s2config_path=str(work / "GPT_SoVITS/configs/s2.json"), hz="25hz")
    # CUDA_VISIBLE_DEVICES is already mapped by the supervisor; keep its physical identity.
    env["_CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    return env


def prepare_data(config, work, prepared, events):
    rows = manifest_rows(config.train_manifest)
    require_text_assets(config, {row["language"] for row in rows})
    prepared.mkdir()
    audio_dir = prepared / "audio"
    audio_dir.mkdir()
    lines = []
    names = set()
    for index, row in enumerate(rows):
        name = f"{index:08d}.wav"
        destination = audio_dir / name
        shutil.copyfile(row["audio"], destination)
        names.add(name)
        lines.append(f"{destination}|{row['speaker']}|{row['language']}|{row['text']}")
    (prepared / "dataset.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(prepared / "source-map.json", {f"{i:08d}.wav": row for i, row in enumerate(rows)})
    env = stage_environment(config, work, prepared)
    for stage, script in (("text", "1-get-text.py"), ("hubert", "2-get-hubert-wav32k.py"), ("semantic", "3-get-semantic.py")):
        events.emit("phase.changed", phase=f"prepare_{stage}")
        subprocess.run([config.python_path, "-s", str(work / "GPT_SoVITS/prepare_datasets" / script)],
                       env=env, cwd=work, check=True)
    text = (prepared / "2-name2text-0.txt").read_text(encoding="utf-8")
    semantic = (prepared / "6-name2semantic-0.tsv").read_text(encoding="utf-8")
    text_names = {line.split("\t")[0] for line in text.splitlines() if line.strip()}
    semantic_names = {line.split("\t")[0] for line in semantic.splitlines() if line.strip()}
    if text_names != names or semantic_names != names:
        raise ValueError("部分录音未完成音素或语义提取，请根据前面的逐行错误修正数据。")
    for name in names:
        for folder, suffix in (("4-cnhubert", ".pt"), ("5-wav32k", "")):
            if not (prepared / folder / (name + suffix)).is_file():
                raise ValueError(f"录音 {name} 缺少预处理产物 {folder}。")
    (prepared / "2-name2text.txt").write_text(text, encoding="utf-8")
    (prepared / "6-name2semantic.tsv").write_text("item_name\tsemantic_audio\n" + semantic.rstrip("\n") + "\n", encoding="utf-8")
    events.emit("tts.dataset.prepared", samples=len(rows))
    return env


def training_settings(config, work, prepared, output):
    import yaml

    paths = asset_paths(config)
    gpt = yaml.safe_load((work / "GPT_SoVITS/configs/s1longer-v2.yaml").read_text())
    g = config.gpt
    gpt["train"].update(epochs=g.epochs, batch_size=g.batch_size, precision=g.precision, seed=g.seed,
                        save_every_n_epoch=g.save_every_epoch, if_save_latest=g.save_latest,
                        if_save_every_weights=True, if_dpo=g.dpo, exp_name="studio",
                        half_weights_save_dir=str(output / "gpt_exports"))
    gpt["optimizer"].update(lr=g.learning_rate, lr_init=g.initial_learning_rate, lr_end=g.final_learning_rate,
                            warmup_steps=g.warmup_steps, decay_steps=g.decay_steps)
    gpt["data"].update(max_sec=g.max_seconds, num_workers=g.num_workers)
    gpt.update(pretrained_s1=paths["gpt"], output_dir=str(output / "gpt_state"),
               train_semantic_path=str(prepared / "6-name2semantic.tsv"), train_phoneme_path=str(prepared / "2-name2text.txt"))
    s2 = json.loads((work / "GPT_SoVITS/configs/s2.json").read_text())
    s = config.sovits
    s2["train"].update(epochs=s.epochs, batch_size=s.batch_size, fp16_run=s.precision == "fp16", seed=s.seed,
                       save_every_epoch=s.save_every_epoch, if_save_latest=s.save_latest, if_save_every_weights=True,
                       learning_rate=s.learning_rate, betas=[s.adam_beta1, s.adam_beta2], eps=s.adam_epsilon,
                       lr_decay=s.lr_decay, log_interval=s.log_interval, lora_rank=s.lora_rank,
                       grad_ckpt=s.gradient_checkpointing, pretrained_s2G=paths["sovits"], pretrained_s2D="",
                       gpu_numbers=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    s2["data"]["exp_dir"] = str(prepared)
    s2["model"]["version"] = config.variant
    s2.update(s2_ckpt_dir=str(output / "sovits_state"), save_weight_dir=str(output / "sovits_exports"),
              name="studio", version=config.variant)
    for path in (output / "gpt_exports", output / "sovits_exports"):
        path.mkdir()
    gpt_path, sovits_path = output / "gpt.yaml", output / "sovits.json"
    gpt_path.write_text(yaml.safe_dump(gpt), encoding="utf-8")
    write_json(sovits_path, s2)
    return {"gpt": str(gpt_path), "sovits": str(sovits_path)}


def train_stage(request, stage):
    config = parse_config(request["config"])
    events = Events(request["events_path"])
    output = Path(request["output"])
    work = Path(request["workspace"])
    outcome = {}
    latest_metrics = {}
    offset = request.get("step_offset", 0)
    if stage == "gpt":
        import pytorch_lightning as pl

        original_fit = pl.Trainer.fit

        class Progress(pl.Callback):
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                metrics = {}
                for key in ("total_loss", "lr", "top_3_acc"):
                    value = trainer.callback_metrics.get(key)
                    if value is not None:
                        metrics[key] = float(value.detach().cpu())
                latest_metrics.update(metrics)
                training_event(events, "gpt", int(trainer.global_step), metrics, trainer.current_epoch + 1, offset=offset)

        def fit(trainer, *args, **kwargs):
            trainer.callbacks.append(Progress())
            result = original_fit(trainer, *args, **kwargs)
            if trainer.current_epoch != config.gpt.epochs or trainer.global_step <= 0:
                raise RuntimeError("GPT 没有完成指定训练轮数或未执行优化步骤。")
            outcome.update(epoch=config.gpt.epochs, global_step=int(trainer.global_step),
                           path=str(output / "gpt_exports" / f"studio-e{config.gpt.epochs}.ckpt"))
            return result

        pl.Trainer.fit = fit
        script = work / "GPT_SoVITS/s1_train.py"
        sys.argv = [str(script), "--config_file", request["settings"]["gpt"]]
    else:
        import process_ckpt
        import utils

        get_hparams = utils.get_hparams

        def protected_hparams(*args, **kwargs):
            hps = get_hparams(*args, **kwargs)
            mask = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
            if hps.train.gpu_numbers != mask:
                raise ValueError("SoVITS 配置显卡与队列分配不一致。")
            hps.train.gpu_numbers = _CudaMask(mask, hps.train)
            return hps

        utils.get_hparams = protected_hparams
        original_logger = utils.get_logger

        def get_logger(*args, **kwargs):
            logger = original_logger(*args, **kwargs)
            logger.addFilter(_SovitsLogFilter())
            return logger

        utils.get_logger = get_logger
        original_summarize = utils.summarize

        def summarize(writer, global_step, scalars=None, *args, **kwargs):
            metrics = {key: float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
                       for key, value in (scalars or {}).items()}
            latest_metrics.update(metrics)
            # Upstream logs after the update, before incrementing its zero-based counter.
            training_event(events, "sovits", global_step + 1, metrics, offset=offset)
            return original_summarize(writer, global_step, scalars or {}, *args, **kwargs)

        utils.summarize = summarize

        original_save = process_ckpt.savee

        def save(ckpt, name, epoch, steps, hps, **kwargs):
            result = original_save(ckpt, name, epoch, steps, hps, **kwargs)
            if result != "Success.":
                raise RuntimeError(f"SoVITS 导出失败：{result}")
            outcome.update(epoch=int(epoch), global_step=int(steps), path=str(Path(hps.save_weight_dir) / f"{name}.pth"))
            return result

        process_ckpt.savee = save
        script = work / "GPT_SoVITS/s2_train_v3_lora.py"
        sys.argv = [str(script), "--config", request["settings"]["sovits"]]
    runpy.run_path(str(script), run_name="__main__")
    target_epochs = config.gpt.epochs if stage == "gpt" else config.sovits.epochs
    if outcome.get("epoch") != target_epochs or not Path(outcome.get("path", "")).is_file():
        raise RuntimeError(f"{stage} 训练没有产出最后一轮的有效权重。")
    if outcome.get("global_step", 0) <= 0:
        raise RuntimeError(f"{stage} 没有执行训练步骤，请检查预处理后的样本长度和批次设置。")
    training_event(events, stage, outcome["global_step"], latest_metrics, target_epochs, offset=offset)
    write_json(output / f"{stage}-completed.json", outcome)


def publish_bundle(config, output, outcomes):
    paths = asset_paths(config)
    bundle = output / "final"
    bundle.mkdir()
    value = {"engine": config.engine, "variant": config.variant, "stage": config.stage, "files": {}}
    for stage, filename in (("gpt", "gpt.ckpt"), ("sovits", "sovits.pth")):
        outcome = outcomes.get(stage)
        source = Path(outcome["path"] if outcome else paths[stage])
        target = bundle / filename
        shutil.copyfile(source, target)
        value[stage] = {"epoch": outcome["epoch"] if outcome else None,
                        "global_step": outcome["global_step"] if outcome else None}
        value["files"][filename] = {"sha256": file_digest(target), "size": target.stat().st_size}
    write_json(bundle / "checkpoint.json", value)
    safe_checkpoint(output.parent, bundle)
    return bundle


def train(request):
    config = parse_config(request["config"])
    work = workspace(config, Path(request["run_dir"]) / "upstream")
    output = Path(request["output"])
    output.mkdir()
    events = Events(request["events_path"])
    prepared = output / "prepared"
    env = prepare_data(config, work, prepared, events)
    settings = training_settings(config, work, prepared, output)
    request.update(workspace=str(work), settings=settings)
    stage_request = output / "stage-request.json"
    outcomes = {}
    for stage in ("sovits", "gpt"):
        if config.stage not in {"both", stage}:
            continue
        events.sync()
        events.emit("phase.changed", phase=f"train_{stage}")
        request["step_offset"] = sum(value["global_step"] for value in outcomes.values())
        write_json(stage_request, request)
        subprocess.run([config.python_path, "-m", "ypuddin.tts.gpt_sovits.bridge", "--request", str(stage_request),
                        "--stage", stage], env=env, cwd=work, check=True)
        outcomes[stage] = json.loads((output / f"{stage}-completed.json").read_text())
    bundle = publish_bundle(config, output, outcomes)
    events.sync()
    events.emit("checkpoint.saved", path=str(bundle), engine=config.engine, variant=config.variant)
    return {"mode": "train", "checkpoint": str(bundle), "stage": config.stage, "variant": config.variant,
            "stages": {stage: {key: value[key] for key in ("epoch", "global_step")} for stage, value in outcomes.items()}}


def offline_language_detection(work):
    infer = sys.modules.get("fast_langdetect.infer")
    if infer is None:
        return
    downloader = getattr(infer, "ModelDownloader", None)
    if downloader is None or not hasattr(downloader, "download") or not hasattr(infer, "_LOCAL_SMALL_MODEL_PATH"):
        raise ValueError("fast_langdetect 版本缺少本地模型加载接口。")

    def missing_local_model(url, save_path, proxy=None):
        raise FileNotFoundError(f"语言识别模型不存在，请在模型管理中重新准备：{save_path}")

    downloader.download = staticmethod(missing_local_model)
    small = Path(work) / "GPT_SoVITS/pretrained_models/fast_langdetect/lid.176.ftz"
    if small.is_file():
        infer._LOCAL_SMALL_MODEL_PATH = small


def sample(request):
    import numpy as np
    import soundfile as sf
    import torch

    config = parse_config(request["config"])
    options = request["sample"]
    require_text_assets(config, [options["gpt_sovits"][key] for key in ("text_language", "reference_language")],
                        inference=True)
    work = workspace(config, Path(request["run_dir"]) / "upstream")
    os.chdir(work)
    for folder in (work / "GPT_SoVITS", work / "GPT_SoVITS/BigVGAN", work):
        sys.path.insert(0, str(folder))
    os.environ["version"] = config.variant
    from TTS_infer_pack.TTS import TTS, TTS_Config

    offline_language_detection(work)
    bundle = safe_checkpoint(options["source_output_dir"], options["checkpoint"])
    metadata = json.loads((bundle / "checkpoint.json").read_text())
    if metadata["variant"] != config.variant:
        raise ValueError("试听检查点与冻结的模型变体不一致。")
    if metadata["stage"] == "gpt":
        from TTS_infer_pack import TTS as tts_module

        original_version = tts_module.get_sovits_version_from_path_fast

        def paired_version(path):
            if Path(path).resolve() == (bundle / "sovits.pth").resolve():
                return "v2", config.variant, False
            return original_version(path)

        # The paired full base was validated before launch; upstream's size heuristic predates v5.
        tts_module.get_sovits_version_from_path_fast = paired_version
    paths = asset_paths(config)
    custom = {"device": "cuda:0", "is_half": config.sovits.precision == "fp16", "version": config.variant,
              "t2s_weights_path": str(bundle / "gpt.ckpt"), "vits_weights_path": str(bundle / "sovits.pth"),
              "bert_base_path": paths["bert"], "cnhuhbert_base_path": paths["hubert"]}
    tts_config = TTS_Config({"custom": custom, config.variant: {**custom, "vits_weights_path": paths["sovits_base"]}})
    tts_config.configs_path = str(Path(request["run_dir"]) / "inference.yaml")
    if not torch.cuda.is_available():
        raise ValueError("GPT-SoVITS 试听需要已分配的 NVIDIA CUDA 显卡。")
    tts = TTS(tts_config)
    specific = options["gpt_sovits"]
    requested_seed = options.get("seed", -1)
    seed = secrets.randbelow(2**32) if requested_seed == -1 else requested_seed
    inputs = {"text": options["text"], "text_lang": specific["text_language"],
              "ref_audio_path": options["reference_audio"], "prompt_text": options["reference_text"],
              "prompt_lang": specific["reference_language"], "seed": seed,
              "top_k": specific.get("top_k", 15), "top_p": specific.get("top_p", 1.0),
              "temperature": specific.get("temperature", 1.0), "speed_factor": specific.get("speed", 1.0),
              "sample_steps": specific["sample_steps"], "cfg_rate": specific["cfg_scale"],
              "repetition_penalty": specific.get("repetition_penalty", 1.35),
              "fragment_interval": specific.get("fragment_interval", 0.3), "text_split_method": "cut1",
              "use_cuda_graph": False, "use_flash_attention": False, "parallel_infer": False,
              "streaming_mode": False, "return_fragment": False, "batch_size": 1}
    generated = list(tts.run(inputs))
    if not generated or any(rate != 48000 for rate, _ in generated):
        raise RuntimeError("GPT-SoVITS 未返回有效的 48 kHz 试听音频。")
    audio = np.concatenate([np.asarray(value) for _, value in generated])
    if not len(audio) or not np.isfinite(audio).all():
        raise RuntimeError("GPT-SoVITS 试听音频为空或包含非有限数值。")
    sf.write(request["sample_path"], audio, 48000, subtype="PCM_16")
    info = audio_info(request["sample_path"])
    return {"mode": "sample", "path": request["sample_path"], "seed": seed, "requested_seed": requested_seed,
            "duration_seconds": info["duration"], "sample_rate": 48000, "cfg_value": inputs["cfg_rate"],
            "inference_timesteps": inputs["sample_steps"], "gpt_sovits": specific, "engine": config.engine,
            "variant": config.variant}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--stage", choices=("gpt", "sovits"))
    args = parser.parse_args()
    from ypuddin.worker_log import configure

    configure()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.stage:
        train_stage(request, args.stage)
        return
    result = train(request) if request["mode"] == "train" else sample(request)
    write_json(request["receipt_path"], result)


if __name__ == "__main__":
    main()
