import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = PROJECT_ROOT / "model" / "DiffSynth-Studio"

sys.path.insert(0, str(DIFFSYNTH_ROOT))

# Avoid remote model lookups and downloads when loading local checkpoints.
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "True")

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


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
    parser = argparse.ArgumentParser(description="Run local Wan2.1-T2V-1.3B inference from YAML config")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model_dir", default=str(PROJECT_ROOT / "model" / "checkpoints" / "wan" / "Wan2.1-T2V-1.3B"), help="Local model directory")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for video generation")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "t2v_yaml.mp4"), help="Output video path")
    parser.add_argument("--height", type=int, default=832, help="Output height")
    parser.add_argument("--width", type=int, default=480, help="Output width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of output frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS")
    parser.add_argument("--device", default="cuda", help="Computation device")
    parser.add_argument("--reserve_vram_gb", type=float, default=2.0, help="Reserved VRAM in GiB for safety")
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE processing")
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
    model_dir = Path(args.model_dir)
    required_files = [
        model_dir / "diffusion_pytorch_model.safetensors",
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "google" / "umt5-xxl",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model files or directories: " + ", ".join(missing))

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be multiples of 16")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must satisfy 4n+1")



def build_pipeline(args):
    device_index = torch.cuda.current_device()
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": args.device,
        "computation_dtype": torch.bfloat16,
        "computation_device": args.device,
    }
    model_dir = Path(args.model_dir)
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=str(model_dir / "diffusion_pytorch_model.safetensors"), **vram_config),
            ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
            ModelConfig(path=str(model_dir / "Wan2.1_VAE.pth"), **vram_config),
        ],
        tokenizer_config=ModelConfig(path=str(model_dir / "google" / "umt5-xxl")),
        vram_limit=torch.cuda.mem_get_info(device_index)[1] / (1024 ** 3) - args.reserve_vram_gb,
    )



def main():
    args = parse_args()
    validate_args(args)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args)
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=args.tiled,
    )
    save_video(video, str(output_path), fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
