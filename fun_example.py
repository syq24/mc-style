import os
import argparse

import torch
from PIL import Image

# Avoid remote model lookups and downloads when loading local checkpoints.
os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "True"

from diffsynth.utils.data import VideoData, save_video
from diffsynth.utils.data import crop_and_resize
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline


MODEL_DIR = "/root/autodl-tmp/mc-style/model/checkpoints/wan/Wan2.1-Fun-V1.1-1.3B-Control"
DEFAULT_CONTROL_VIDEO_PATH = "/root/autodl-tmp/mc-style/outputs/v1.mp4"
DEFAULT_REFERENCE_IMAGE_PATH = "/root/autodl-tmp/mc-style/data/style_refs/Starry_Night.webp"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/mc-style/outputs/v2.mp4"
DEFAULT_PROMPT = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
DEFAULT_NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

HEIGHT = 832
WIDTH = 480
NUM_FRAMES = 49


def parse_args():
    parser = argparse.ArgumentParser(description="Run local Wan2.1-Fun-V1.1-1.3B-Control inference")
    parser.add_argument("--control_video", default=DEFAULT_CONTROL_VIDEO_PATH, help="Path to the content reference video")
    parser.add_argument("--reference_image", default=DEFAULT_REFERENCE_IMAGE_PATH, help="Optional path to the style reference image")
    parser.add_argument(
        "--reference_strength",
        type=float,
        default=1.0,
        help="Scale applied to reference-image features. 0 disables the reference branch, 1 keeps the default behavior, values above 1 amplify it.",
    )
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
    if args.reference_strength < 0:
        raise ValueError("reference_strength must be greater than or equal to 0")

    if args.control_video is not None and not os.path.isfile(args.control_video):
        raise FileNotFoundError(
            f"Missing control video: {args.control_video}\n"
            "Pass a valid path with --control_video before running this script."
        )

    if args.reference_image is not None and not os.path.isfile(args.reference_image):
        raise FileNotFoundError(
            f"Missing reference image: {args.reference_image}\n"
            "Pass a valid path with --reference_image or omit this argument."
        )

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
            ModelConfig(path=f"{MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", **vram_config),
        ],
        tokenizer_config=ModelConfig(path=f"{MODEL_DIR}/google/umt5-xxl/"),
        vram_limit=torch.cuda.mem_get_info(device_index)[1] / (1024 ** 3) - 2,
    )


def apply_reference_strength(pipe, reference_strength):
    if reference_strength == 1.0:
        return

    for unit in pipe.units:
        if unit.__class__.__name__ != "WanVideoUnit_FunReference":
            continue

        original_process = unit.process

        def scaled_process(pipe_obj, reference_image, height, width, _original_process=original_process):
            outputs = _original_process(pipe_obj, reference_image, height, width)
            if not outputs:
                return outputs
            if "reference_latents" in outputs:
                outputs["reference_latents"] = outputs["reference_latents"] * reference_strength
            if "clip_feature" in outputs:
                outputs["clip_feature"] = outputs["clip_feature"] * reference_strength
            return outputs

        unit.process = scaled_process
        return

    raise RuntimeError("WanVideoUnit_FunReference was not found in the pipeline units")


def build_control_video(args):
    if args.control_video is not None:
        video = VideoData(args.control_video)
        src_height, src_width = video.shape()
        if (src_height, src_width) != (args.height, args.width):
            print(
                f"Resizing control video from {src_width}x{src_height} to {args.width}x{args.height}."
            )
        return VideoData(args.control_video, height=args.height, width=args.width)

    # Wan2.1-Fun-V1.1-1.3B-Control expects control latents. When no external
    # control video is provided, use blank frames as a neutral control input.
    return [Image.new("RGB", (args.width, args.height), color="black") for _ in range(args.num_frames)]


def build_reference_image(args):
    if args.reference_image is None or args.reference_strength == 0:
        return None

    reference_image = Image.open(args.reference_image).convert("RGB")
    if reference_image.size != (args.width, args.height):
        print(
            f"Resizing reference image from {reference_image.width}x{reference_image.height} to {args.width}x{args.height}."
        )
        reference_image = crop_and_resize(reference_image, args.height, args.width)
    return reference_image


def main():
    args = parse_args()
    validate_args(args)

    pipe = build_pipeline()
    apply_reference_strength(pipe, args.reference_strength)
    control_video = build_control_video(args)
    reference_image = build_reference_image(args)

    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        control_video=control_video,
        reference_image=reference_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, args.output, fps=args.fps, quality=5)


if __name__ == "__main__":
    main()