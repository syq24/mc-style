import argparse
import os
import sys
from pathlib import Path

import accelerate
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = PROJECT_ROOT / "model" / "DiffSynth-Studio"
WAN_TRAINING_ROOT = DIFFSYNTH_ROOT / "examples" / "wanvideo" / "model_training"

sys.path.insert(0, str(DIFFSYNTH_ROOT))
sys.path.insert(0, str(WAN_TRAINING_ROOT))

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadAudio, LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import ModelLogger, launch_data_process_task, launch_training_task
from train import WanTrainingModule, wan_parser


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


def validate_yaml_keys(parser, yaml_config):
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(yaml_config) - valid_keys)
    if unknown_keys:
        raise ValueError("Unknown keys found in YAML config: " + ", ".join(unknown_keys))


def parse_args():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--config", required=True, help="Path to YAML config file")
    bootstrap_args, remaining_argv = bootstrap_parser.parse_known_args()

    config_path = Path(bootstrap_args.config).expanduser().resolve()
    yaml_config = load_yaml_config(config_path)

    parser = wan_parser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    validate_yaml_keys(parser, yaml_config)
    parser.set_defaults(**yaml_config)
    args = parser.parse_args(remaining_argv + ["--config", str(config_path)])
    return args


def build_dataset(args):
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
                args.num_frames,
                4,
                1,
                frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
            ),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )


def build_model(args, accelerator):
    return WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)
        ],
    )
    dataset = build_dataset(args)
    model = build_model(args, accelerator)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    if args.task not in launcher_map:
        raise ValueError(f"Unsupported task: {args.task}")

    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
