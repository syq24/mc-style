import argparse
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = PROJECT_ROOT / "model" / "DiffSynth-Studio"

sys.path.insert(0, str(DIFFSYNTH_ROOT))

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def get_ffmpeg_executable():
    try:
        import imageio_ffmpeg
    except ImportError:
        return shutil.which("ffmpeg")

    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def flatten_config(config):
    flattened = {}
    for key, value in (config or {}).items():
        if isinstance(value, dict):
            flattened.update(flatten_config(value))
        else:
            flattened[key] = value
    return flattened


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    return flatten_config(raw_config)


def build_parser():
    parser = argparse.ArgumentParser(description="Run stage-2 Wan style inference from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--pipeline", default="wan_video_mc", help="Pipeline module name under diffsynth.pipelines")
    parser.add_argument("--model_dir", default=str(PROJECT_ROOT / "model" / "checkpoints" / "wan" / "Wan2.1-T2V-1.3B"), help="Local Wan model directory")
    parser.add_argument("--style_repo_path", default=str(DIFFSYNTH_ROOT / "diffsynth" / "style-tokenizer"), help="Local style-tokenizer repo path")
    parser.add_argument("--style_ckpt", default=None, help="Stage-1 style encoder checkpoint")
    parser.add_argument("--style_train_ckpt", default=None, help="Stage-2 trainable style branch checkpoint")
    parser.add_argument("--style_num_tokens", type=int, default=8, help="Number of style tokens")
    parser.add_argument("--lora_ckpt", default=None, help="Optional DiT LoRA checkpoint to fuse for inference")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA fusion scale")
    parser.add_argument("--model_paths", default=None, help="Optional JSON list of model paths; defaults to local Wan 1.3B weights")
    parser.add_argument("--prompt", default="", help="Fallback prompt when dataset prompt is missing")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt")
    parser.add_argument("--reference_image", default=None, help="Optional single reference image path")
    parser.add_argument("--output", default=None, help="Single-sample output path")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "outputs" / "infer" / "stage2_eval"), help="Batch output directory")
    parser.add_argument("--dataset_metadata_path", default=None, help="Optional JSONL metadata path for small-batch validation")
    parser.add_argument("--prompt_column", default="prompt", help="Prompt column in JSONL")
    parser.add_argument("--image_column", default="image_path", help="Reference image column in JSONL")
    parser.add_argument("--video_column", default="video_path", help="Video path column in JSONL, only used for naming outputs")
    parser.add_argument("--sample_offset", type=int, default=0, help="Skip this many dataset rows before inference")
    parser.add_argument("--max_samples", type=int, default=4, help="Max number of samples to run from the dataset")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=832, help="Output width")
    parser.add_argument("--num_frames", type=int, default=33, help="Number of output frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="Global classifier-free guidance scale")
    parser.add_argument("--style_cfg_scale", type=float, default=1.0, help="Independent scale for the injected style context")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--reserve_vram_gb", type=float, default=6.0, help="Reserved VRAM in GiB for safety")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE processing")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", required=True, help="Path to YAML config file")
    bootstrap_args, remaining_argv = bootstrap_parser.parse_known_args()

    config_path = Path(bootstrap_args.config).expanduser().resolve()
    yaml_config = load_yaml_config(config_path)

    parser = build_parser()
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(yaml_config) - valid_keys)
    if unknown_keys:
        raise ValueError("Unknown keys found in YAML config: " + ", ".join(unknown_keys))

    parser.set_defaults(**yaml_config)
    return parser.parse_args(remaining_argv + ["--config", str(config_path)])


def validate_args(args):
    model_dir = Path(args.model_dir).expanduser().resolve()
    required_files = [
        model_dir / "diffusion_pytorch_model.safetensors",
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "google" / "umt5-xxl",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files or directories: " + ", ".join(missing))

    for path_str, label in [
        (args.style_repo_path, "style_repo_path"),
        (args.style_ckpt, "style_ckpt"),
        (args.style_train_ckpt, "style_train_ckpt"),
    ]:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if args.lora_ckpt is not None:
        lora_path = Path(args.lora_ckpt).expanduser().resolve()
        if not lora_path.exists():
            raise FileNotFoundError(f"lora_ckpt does not exist: {lora_path}")

    if args.reference_image is not None and not Path(args.reference_image).expanduser().exists():
        raise FileNotFoundError(f"reference_image does not exist: {args.reference_image}")

    if args.dataset_metadata_path is not None and not Path(args.dataset_metadata_path).expanduser().exists():
        raise FileNotFoundError(f"dataset_metadata_path does not exist: {args.dataset_metadata_path}")

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be multiples of 16")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must satisfy 4n+1")
    if args.max_samples <= 0:
        raise ValueError("max_samples must be > 0")


def build_model_configs(args):
    if args.model_paths:
        return [ModelConfig(path=path) for path in json.loads(args.model_paths)]

    if args.device == "cuda":
        device_index = torch.cuda.current_device()
        free_gb = torch.cuda.mem_get_info(device_index)[0] / (1024 ** 3)
        vram_limit = max(1.0, free_gb - args.reserve_vram_gb)
        preparing_device = "cuda"
    else:
        vram_limit = None
        preparing_device = args.device

    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": preparing_device,
        "computation_dtype": torch.bfloat16,
        "computation_device": args.device,
    }
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_configs = [
        ModelConfig(path=str(model_dir / "diffusion_pytorch_model.safetensors"), **vram_config),
        ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
        ModelConfig(path=str(model_dir / "Wan2.1_VAE.pth"), **vram_config),
    ]
    return model_configs, vram_limit


def build_pipeline(args):
    pipeline_module = importlib.import_module(f"diffsynth.pipelines.{args.pipeline}")
    wan_pipeline_class = getattr(pipeline_module, "WanVideoPipeline")
    model_configs, vram_limit = build_model_configs(args)
    model_dir = Path(args.model_dir).expanduser().resolve()
    pipe = wan_pipeline_class.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(model_dir / "google" / "umt5-xxl")),
        vram_limit=vram_limit,
    )
    if not hasattr(pipe, "initialize_style_modules"):
        raise ValueError(f"Pipeline {args.pipeline} does not expose initialize_style_modules.")
    pipe.initialize_style_modules(
        style_repo_path=str(Path(args.style_repo_path).expanduser().resolve()),
        style_ckpt=str(Path(args.style_ckpt).expanduser().resolve()),
        train_ckpt=str(Path(args.style_train_ckpt).expanduser().resolve()),
        style_num_tokens=args.style_num_tokens,
    )
    if args.lora_ckpt is not None:
        pipe.load_lora(
            pipe.dit,
            str(Path(args.lora_ckpt).expanduser().resolve()),
            alpha=args.lora_alpha,
        )
    return pipe


def iter_dataset_samples(args):
    if args.dataset_metadata_path is None:
        yield {
            "prompt": args.prompt,
            "reference_image": args.reference_image,
            "source_name": "single_sample",
            "dataset_index": 0,
        }
        return

    metadata_path = Path(args.dataset_metadata_path).expanduser().resolve()
    selected = 0
    with metadata_path.open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < args.sample_offset:
                continue
            if selected >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompt = data.get(args.prompt_column, args.prompt)
            reference_image = data.get(args.image_column)
            video_name = Path(data.get(args.video_column, f"sample_{index}")).stem
            style_name = Path(reference_image).stem if reference_image else "no_style"
            yield {
                "prompt": prompt,
                "reference_image": reference_image,
                "source_video": data.get(args.video_column),
                "source_name": f"{index:05d}_{video_name}_{style_name}",
                "dataset_index": index,
                "raw": data,
            }
            selected += 1


def load_reference_image(path):
    if path is None:
        return None
    return Image.open(Path(path).expanduser()).convert("RGB")


def save_manifest(output_dir, record):
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_reference_image(output_dir, sample):
    reference_image = sample.get("reference_image")
    if reference_image is None:
        return None

    source_path = Path(reference_image).expanduser().resolve()
    if not source_path.exists():
        print(f"[warn] reference image not found: {source_path}", flush=True)
        return None

    target_path = output_dir / f"{sample['source_name']}__style{source_path.suffix}"
    if not target_path.exists():
        shutil.copy2(source_path, target_path)
    return str(target_path)


def copy_source_video(output_dir, sample):
    output_dir = Path(output_dir)
    source_video = sample.get("source_video")
    if source_video is None:
        return None

    source_path = Path(source_video).expanduser().resolve()
    if not source_path.exists():
        print(f"[warn] source video not found: {source_path}", flush=True)
        return None

    target_path = output_dir / f"{sample['source_name']}__source{source_path.suffix}"
    # if not target_path.exists():
    #     shutil.copy2(source_path, target_path)
    if target_path.exists():
        return str(target_path)

    ffmpeg_executable = get_ffmpeg_executable()
    if ffmpeg_executable is None:
        print("[warn] ffmpeg not available, skip source preview transcode", flush=True)
        return None

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as error:
        print(f"[warn] failed to transcode source video: {error.stderr.strip()}", flush=True)
        return None

    return str(target_path)


def parse_num_frames_from_path(path):
    stem = Path(path).stem
    match = re.search(r"(\d+)to(\d+)$", stem)
    if match is None:
        return None
    start_frame = int(match.group(1))
    end_frame = int(match.group(2))
    if end_frame < start_frame:
        return None
    return end_frame - start_frame + 1


def get_source_video_num_frames(sample):
    raw = sample.get("raw") or {}
    video_meta = raw.get("video_meta") or {}
    for key in ("num_frames", "frame_count", "frames"):
        value = video_meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    source_video = sample.get("source_video")
    if source_video is None:
        return None
    return parse_num_frames_from_path(source_video)


def run_batch_inference(args, pipe):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id, sample in enumerate(iter_dataset_samples(args), start=1):
        output_path = output_dir / f"{sample['source_name']}.mp4"
        reference_image_copy = copy_reference_image(output_dir, sample)
        source_video_copy = copy_source_video(output_dir, sample)
        source_video_num_frames = get_source_video_num_frames(sample)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} already exists")
            continue

        reference_image = load_reference_image(sample["reference_image"])
        print(
            f"[infer] sample={sample_id} dataset_index={sample['dataset_index']} output={output_path.name}",
            flush=True,
        )
        video = pipe(
            prompt=sample["prompt"],
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            style_cfg_scale=args.style_cfg_scale,
            tiled=args.tiled,
            reference_image=reference_image,
        )
        save_video(video, str(output_path), fps=args.fps, quality=5)
        save_manifest(
            output_dir,
            {
                "dataset_index": sample["dataset_index"],
                "output": str(output_path),
                "prompt": sample["prompt"],
                "reference_image": sample["reference_image"],
                "reference_image_copy": reference_image_copy,
                "source_video": sample.get("source_video"),
                "source_video_copy": source_video_copy,
                "source_video_num_frames": source_video_num_frames,
                "cfg_scale": args.cfg_scale,
                "style_cfg_scale": args.style_cfg_scale,
            },
        )


def main():
    args = parse_args()
    validate_args(args)
    pipe = build_pipeline(args)
    run_batch_inference(args, pipe)


if __name__ == "__main__":
    main()