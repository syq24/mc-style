import os
import argparse

import torch

# Avoid remote model lookups and downloads when loading local checkpoints.
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


MODEL_DIR = "/root/autodl-tmp/mc-style/model/checkpoints/wan/Wan2.1-T2V-1.3B"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/mc-style/outputs/t2v.mp4"
DEFAULT_PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

HEIGHT = 832
WIDTH = 480
NUM_FRAMES = 49


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Wan2.1-T2V-1.3B inference")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt for video generation")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for video generation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output video path")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Output height")
    parser.add_argument("--width", type=int, default=WIDTH, help="Output width")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES, help="Number of output frames")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--fps", type=int, default=15, help="Output FPS")
    return parser.parse_args()


def validate_args(args):
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be multiples of 16")

    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must satisfy 4n+1")



def build_pipeline():
    device_index = torch.cuda.current_device()
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=f"{MODEL_DIR}/diffusion_pytorch_model.safetensors", **vram_config),
            ModelConfig(path=f"{MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth", **vram_config),
            ModelConfig(path=f"{MODEL_DIR}/Wan2.1_VAE.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(path=f"{MODEL_DIR}/google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info(device_index)[1] / (1024 ** 3) - 2,
    )



def main():
    args = parse_args()
    validate_args(args)

    pipe = build_pipeline()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, args.output, fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
