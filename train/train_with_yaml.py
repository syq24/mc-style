import argparse
import csv
import json
import math
import importlib.util
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import accelerate
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSYNTH_ROOT = PROJECT_ROOT / "model" / "DiffSynth-Studio"
WAN_TRAINING_ROOT = DIFFSYNTH_ROOT / "examples" / "wanvideo" / "model_training"

sys.path.insert(0, str(DIFFSYNTH_ROOT))
sys.path.insert(0, str(WAN_TRAINING_ROOT))

from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import LoadAudio, LoadVideo, ImageCropAndResize, ToAbsolutePath
from diffsynth.diffusion import DiffusionTrainingModule, ModelLogger, launch_data_process_task, launch_training_task


def load_wan_training_entrypoints():
    module_path = WAN_TRAINING_ROOT / "train.py"
    spec = importlib.util.spec_from_file_location("wanvideo_model_training_train", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Wan training module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WanTrainingModule, module.wan_parser


WanTrainingModule, wan_parser = load_wan_training_entrypoints()


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


def infer_metadata_columns(metadata_path):
    metadata_path = Path(metadata_path).expanduser().resolve()
    suffix = metadata_path.suffix.lower()

    if suffix == ".csv":
        with open(metadata_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                return set()
            return set(reader.fieldnames)

    if suffix == ".jsonl":
        with open(metadata_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if isinstance(data, dict):
                    return set(data.keys())
                break
        return set()

    if suffix == ".json":
        with open(metadata_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return set(data[0].keys())
        if isinstance(data, dict):
            return set(data.keys())

    return set()


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
    parser.add_argument("--training_stage", type=str, default="wan_style", choices=["wan_style", "style_encoder"], help="Training stage. 'wan_style' trains the Wan pipeline with style tokens; 'style_encoder' trains only the style encoder with contrastive loss.")
    parser.add_argument("--video_column", type=str, default="video_path", help="Source column name for video path in JSONL/CSV metadata.")
    parser.add_argument("--image_column", type=str, default="image_path", help="Source column name for reference/style image path in JSONL/CSV metadata.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Source column name for prompt text in JSONL/CSV metadata.")
    parser.add_argument("--style_label_column", type=str, default=None, help="Optional column name for explicit style labels. If empty, labels are derived from image file names.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size. style_encoder stage should use batch_size > 1.")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07, help="Temperature for supervised contrastive loss in style_encoder stage.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    validate_yaml_keys(parser, yaml_config)
    parser.set_defaults(**yaml_config)
    args = parser.parse_args(remaining_argv + ["--config", str(config_path)])
    return args


class ColumnMappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.mapping = mapping
        self.load_from_cache = getattr(dataset, "load_from_cache", False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        mapped = {}
        for target_key, source_key in self.mapping.items():
            if source_key is not None and source_key in data:
                mapped[target_key] = data[source_key]
        return mapped


class StyleEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, image_column, style_label_column=None):
        self.dataset = dataset
        self.image_column = image_column
        self.style_label_column = style_label_column
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.base_length = len(getattr(self.dataset, "data", []))
        self.repeat = getattr(self.dataset, "repeat", 1)
        self.base_style_codes = []
        self.style_to_base_indices = defaultdict(list)
        for base_index, raw in enumerate(getattr(self.dataset, "data", [])):
            image_path = raw[self.image_column]
            if self.style_label_column is not None and self.style_label_column in raw:
                style_code = raw[self.style_label_column]
            else:
                style_code = os.path.basename(str(image_path)).split("____")[0]
            self.base_style_codes.append(style_code)
            self.style_to_base_indices[style_code].append(base_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        processed = self.dataset[index]
        raw = self.dataset.data[index % len(self.dataset.data)]
        image_path = raw[self.image_column]
        if self.style_label_column is not None and self.style_label_column in raw:
            style_code = raw[self.style_label_column]
        else:
            style_code = os.path.basename(str(image_path)).split("____")[0]
        return {
            "image": processed[self.image_column],
            "style_code": style_code,
            "image_path": image_path,
        }


class StyleCodeBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.base_length = max(1, dataset.base_length)
        self.repeat = max(1, getattr(dataset, "repeat", 1))
        self.style_codes = list(dataset.style_to_base_indices.keys())
        if len(self.style_codes) == 0:
            raise ValueError("StyleCodeBatchSampler requires at least one style code.")

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def _sample_virtual_index(self, base_index):
        repeat_offset = random.randrange(self.repeat)
        return base_index + repeat_offset * self.base_length

    def __iter__(self):
        pair_count = max(1, self.batch_size // 2)
        remainder = self.batch_size - pair_count * 2
        for _ in range(len(self)):
            batch = []
            chosen_styles = random.choices(self.style_codes, k=pair_count)
            for style_code in chosen_styles:
                base_indices = self.dataset.style_to_base_indices[style_code]
                sampled = random.choices(base_indices, k=2) if len(base_indices) == 1 else random.sample(base_indices, 2)
                batch.extend(self._sample_virtual_index(index) for index in sampled)

            if remainder > 0:
                extra_styles = random.choices(self.style_codes, k=remainder)
                for style_code in extra_styles:
                    base_index = random.choice(self.dataset.style_to_base_indices[style_code])
                    batch.append(self._sample_virtual_index(base_index))

            random.shuffle(batch)
            yield batch


def load_style_encoder_only(style_repo_path, style_ckpt, device):
    repo = Path(style_repo_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"style-tokenizer repo not found at {repo}")

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import infer as style_infer

    if style_ckpt is not None and Path(style_ckpt).expanduser().exists():
        ckpt_path = Path(style_ckpt).expanduser().resolve()
        if ckpt_path.suffix == ".safetensors":
            style_encoder = style_infer.StyleClip(feat_dim=768, model_path=str(repo)).to(device)
            preprocess = style_encoder.get_transform()
            state_dict = load_state_dict(str(ckpt_path), device="cpu")
            style_encoder.load_state_dict(state_dict, strict=False)
        else:
            style_encoder, preprocess = style_infer.load_style_encoder(
                device=device,
                style_ckpt=str(ckpt_path),
                model_path=str(repo),
            )
    else:
        style_encoder = style_infer.StyleClip(feat_dim=768, model_path=str(repo)).to(device)
        preprocess = style_encoder.get_transform()

    style_encoder = style_encoder.float().to(device)
    if hasattr(style_encoder, "freeze_encoder"):
        style_encoder.freeze_encoder(False)
    style_encoder.train()
    return style_encoder, preprocess


def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    labels = list(labels)
    similarity = torch.matmul(features, features.T) / temperature
    logits_mask = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=device)
    similarity = similarity.masked_fill(~logits_mask, float("-inf"))

    label_mask = torch.tensor(
        [[labels[i] == labels[j] for j in range(len(labels))] for i in range(len(labels))],
        device=device,
        dtype=torch.bool,
    )
    positive_mask = label_mask & logits_mask
    positive_count = positive_mask.sum(dim=1)
    valid_rows = positive_count > 0
    if not torch.any(valid_rows):
        return features.sum() * 0.0

    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    log_prob_pos = log_prob.masked_fill(~positive_mask, 0.0)
    mean_log_prob_pos = log_prob_pos.sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid_rows].mean()


class StyleEncoderTrainingModule(DiffusionTrainingModule):
    def __init__(self, style_repo_path, style_ckpt=None, device="cpu", contrastive_temperature=0.07):
        super().__init__()
        self.device = device
        self.contrastive_temperature = contrastive_temperature
        self.style_encoder, self.style_preprocess = load_style_encoder_only(style_repo_path, style_ckpt, device)

    def forward(self, batch):
        images = []
        labels = []
        for sample in batch:
            image = sample["image"]
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                image = image.to(device=self.device, dtype=torch.float32)
            else:
                image = self.style_preprocess(image).unsqueeze(0).to(self.device)
            images.append(image)
            labels.append(sample["style_code"])
        image_batch = torch.cat(images, dim=0)
        features = self.style_encoder(image_batch)
        features = F.normalize(features.float(), dim=-1)
        return supervised_contrastive_loss(features, labels, temperature=self.contrastive_temperature)


def build_dataset(args):
    if args.training_stage == "style_encoder":
        base_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=[args.image_column],
            main_data_operator=UnifiedDataset.default_image_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
            ),
        )
        return StyleEncoderDataset(base_dataset, image_column=args.image_column, style_label_column=args.style_label_column)

    metadata_columns = infer_metadata_columns(args.dataset_metadata_path)
    special_operator_map = {
        args.image_column: UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    }
    if "animate_face_video" in metadata_columns:
        special_operator_map["animate_face_video"] = ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
            args.num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
        )
    if "input_audio" in metadata_columns:
        special_operator_map["input_audio"] = ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000)
    if "wantodance_music_path" in metadata_columns:
        special_operator_map["wantodance_music_path"] = ToAbsolutePath(args.dataset_base_path)

    base_dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=[args.video_column, args.image_column],
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
        special_operator_map=special_operator_map,
    )
    return ColumnMappedDataset(
        base_dataset,
        mapping={
            "video": args.video_column,
            "reference_image": args.image_column,
            "prompt": args.prompt_column,
        },
    )


def build_model(args, accelerator):
    if args.training_stage == "style_encoder":
        return StyleEncoderTrainingModule(
            style_repo_path=args.style_repo_path,
            style_ckpt=args.style_ckpt,
            device="cpu" if args.initialize_model_on_cpu else accelerator.device,
            contrastive_temperature=args.contrastive_temperature,
        )
    return WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        style_repo_path=args.style_repo_path,
        style_ckpt=args.style_ckpt,
        style_fusion_ckpt=args.style_fusion_ckpt,
        style_train_ckpt=args.style_train_ckpt,
        style_num_tokens=args.style_num_tokens,
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
        pipeline=args.pipeline,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )


def launch_style_encoder_training_task(accelerator, dataset, model, model_logger, args=None):
    learning_rate = args.learning_rate if args is not None else 1e-4
    weight_decay = args.weight_decay if args is not None else 1e-2
    num_workers = args.dataset_num_workers if args is not None else 0
    save_steps = args.save_steps if args is not None else None
    num_epochs = args.num_epochs if args is not None else 1
    batch_size = args.batch_size if args is not None else 1

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    batch_sampler = StyleCodeBatchSampler(dataset, batch_size=batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x, num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if accelerator.is_main_process:
        print(
            f"[style_encoder] dataset_size={len(dataset)} batch_size={batch_size} "
            f"steps_per_epoch={len(dataloader)} num_epochs={num_epochs} save_steps={save_steps} sampler=style_code_pairs",
            flush=True,
        )
        print(f"[style_encoder] output_dir={model_logger.output_path}", flush=True)
    for epoch_id in range(num_epochs):
        if accelerator.is_main_process:
            print(f"[style_encoder] epoch {epoch_id + 1}/{num_epochs} start", flush=True)
        for step_id, batch in enumerate(dataloader, start=1):
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
            if accelerator.is_main_process and (step_id == 1 or step_id % 50 == 0):
                print(
                    f"[style_encoder] epoch={epoch_id + 1} step={step_id}/{len(dataloader)} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            if accelerator.is_main_process:
                print(f"[style_encoder] epoch {epoch_id + 1} checkpoint saved", flush=True)
    model_logger.on_training_end(accelerator, model, save_steps)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.training_stage == "style_encoder" and getattr(args, "remove_prefix_in_ckpt", None) in (None, "pipe.", "pipe"):
        args.remove_prefix_in_ckpt = "style_encoder."

    # If trainable_models not explicitly provided, derive it from freeze flags in YAML
    if args.training_stage != "style_encoder" and getattr(args, "trainable_models", None) in (None, ""):
        models = []
        # common backbone
        if not getattr(args, "freeze_dit", False):
            models.append("dit")
        if not getattr(args, "freeze_text_encoder", False):
            models.append("text_encoder")
        if not getattr(args, "freeze_vae", False):
            models.append("vae")
        # style-specific
        if not getattr(args, "freeze_style_encoder", False):
            models.append("style_encoder")
        if not getattr(args, "freeze_style_tokenizer", False):
            models.append("style_tokenizer")
        if not getattr(args, "freeze_style_proj", False):
            models.append("style_proj")
        if getattr(args, "pipeline", "") == "wan_video_mc_ipa_dit" and not getattr(args, "freeze_style_ip_adapter", False):
            models.append("style_ip_adapter")
        args.trainable_models = ",".join(models) if len(models) > 0 else None

    # ensure pipeline default
    args.pipeline = getattr(args, "pipeline", "wan_video_mc")

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
        "style_encoder": launch_style_encoder_training_task,
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_key = args.training_stage if args.training_stage == "style_encoder" else args.task
    if launcher_key not in launcher_map:
        raise ValueError(f"Unsupported task/stage: {launcher_key}")

    launcher_map[launcher_key](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
