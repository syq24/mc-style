# Stage-2 Experiment Matrix

## Validation splits

All split files are under:

`/root/autodl-tmp/mc-style/data/datasets/validation_splits`

### 1. Style30K in-style validation
- Train: `openvid_style30k_instyle_train.jsonl`
- Val: `openvid_style30k_instyle_val.jsonl`
- Goal: evaluate whether the model can generalize to unseen image instances of seen style codes.
- Leakage control:
  - `style_code` is shared across train/val by design.
  - `image_path` is disjoint.
  - `video_path` is disjoint.
- Size:
  - train rows: 24340
  - val rows: 3237
  - shared style codes: 843

### 2. Style30K out-style validation
- Train: `openvid_style30k_outstyle_train.jsonl`
- Val: `openvid_style30k_outstyle_val.jsonl`
- Goal: evaluate style generalization to unseen style codes.
- Leakage control:
  - `style_code` is disjoint.
  - `image_path` is disjoint.
  - `video_path` is disjoint.
- Size:
  - train rows: 24819
  - val rows: 2758
  - held-out style codes: 190

### 3. WikiArt validation
- Train: `openvid_wikiart_train.jsonl`
- Val: `openvid_wikiart_val.jsonl`
- Goal: evaluate stage-2 transfer under a broader style-image source where stage-1 pretraining is not used.
- Split rule:
  - connected-component split over `video_path` and `image_path`
  - stratified by `video_meta.bucket`
- Leakage control:
  - `video_path` is disjoint.
  - `image_path` is disjoint.
- Size:
  - train rows: 74457
  - val rows: 8274
  - train buckets: `generic_scene=28752`, `people=45705`
  - val buckets: `generic_scene=3165`, `people=5109`

## Stage-2 purpose

Stage-2 trains the style injection interface, not the style representation itself.

Input:
- content video
- prompt
- reference style image

Trainable modules:
- `style_tokenizer`
- `style_proj`

Frozen modules in the current baseline:
- `style_encoder`
- `text_encoder`
- `dit`
- `vae`

The goal is to learn a mapping from style-image features to a condition representation that Wan can consume stably.

## Current baseline family

### B0. No-style baseline
- Remove the style branch entirely.
- Purpose: quantify the gain from any style injection.

### B1. Current context-fusion baseline
- Use the current pipeline in `wan_video_mc.py`.
- Path: reference image -> style encoder -> style tokenizer -> style proj -> append to prompt context.
- Purpose: current engineering baseline.

### B2. Global-style-vector baseline
- Replace style tokenizer with a single global style vector + projector.
- Purpose: test whether tokenization is necessary.

## Injection variants

### I1. Context fusion
- Current implementation.
- Style tokens are concatenated into text context before the DiT consumes the condition sequence.

### I2. Decoupled style attention
- IP-Adapter-style variant adapted to DiT.
- Keep text conditioning on the original attention path.
- Add a separate style-conditioning attention or modulation path inside selected DiT blocks.
- Only the newly added style-conditioning modules are trainable.

### I3. Hybrid injection
- Keep a weak context fusion branch plus a shallow decoupled style branch.
- Purpose: test whether coarse global style and local style guidance are complementary.

## Leakage-control variants

### L0. No explicit leakage suppression
- Only use the frozen-backbone setup.
- Purpose: baseline.

### L1. Strong style-image augmentation
- Random crop, resize jitter, color jitter, grayscale probability, blur.
- Purpose: suppress direct content copying from reference images.

### L2. Content-decoupling regularization
- Penalize similarity between style tokens and content/video features.
- Example directions:
  - feature decorrelation
  - adversarial content prediction head
  - information bottleneck on style tokens

### L3. Reference masking / low-frequency style input
- Remove obvious object structure from the style image before encoding.
- Purpose: force the branch to encode style statistics rather than composition.

## Hierarchical-style variants

### H0. Single-level style feature
- Use one final image feature only.

### H1. Multi-layer style fusion
- Fuse multiple visual layers from the image encoder before tokenization.
- Purpose: separate coarse palette/texture from fine brushstroke statistics.

### H2. Layer-wise DiT injection
- Inject coarse style at early blocks and fine style at later blocks.
- Purpose: test whether style information should be distributed across denoising depth.

## Loss design

### Current stage-2 training loss
- Training task: `sft`
- Loss family: `FlowMatchSFTLoss`
- Interpretation:
  - the main diffusion objective is unchanged
  - only the style-conditioning interface is trainable
  - gradients force the style branch to produce useful conditional features for the frozen Wan backbone

### Optional auxiliary losses for ablation
- `A1`: style consistency loss
  - same reference style image should produce consistent style behavior across multiple content videos.
- `A2`: content preservation loss
  - preserve content semantics, motion, and structure while adding style.
- `A3`: leakage suppression loss
  - reduce direct content transfer from the reference image.

## Recommended experiment matrix

### Core matrix
1. `B1 + I1 + L0 + H0`
- Current baseline.

2. `B2 + I1 + L0 + H0`
- Test whether tokenization helps beyond a global style vector.

3. `B1 + I2 + L0 + H0`
- Main architectural comparison against an IP-Adapter-like decoupled path.

4. `B1 + I1 + L1 + H0`
- Test whether simple augmentation reduces content leakage.

5. `B1 + I2 + L1 + H0`
- Check whether decoupled attention plus augmentation is more robust.

### Extended matrix
6. `B1 + I1 + L2 + H0`
- Explicit leakage-control regularization.

7. `B1 + I2 + L2 + H0`
- Best candidate for a stronger style-only interface.

8. `B1 + I1 + L1 + H1`
- Multi-layer style extraction with current fusion.

9. `B1 + I2 + L1 + H1`
- Multi-layer extraction with decoupled style attention.

## Evaluation metrics

### Human-facing qualitative evaluation
- Style fidelity: does the output visually match the reference style?
- Content preservation: are objects, actions, and composition still correct?
- Temporal stability: is style consistent across frames?
- Leakage level: does the output copy subject/layout from the reference image?

### Quantitative evaluation
- CLIP text alignment
  - prompt vs generated frames.
- Style-image similarity
  - generated frames vs reference style image using image encoder features.
- Content-video preservation
  - generated frames vs content video frames using content encoder features.
- Temporal consistency
  - frame-to-frame feature variance or optical-flow-based consistency.
- Leakage score
  - similarity to reference image content beyond style statistics.

## Suggested execution order

1. Start with `B1 + I1 + L0 + H0` on Style30K in-style validation.
2. Add Style30K out-style validation to test style generalization.
3. Add WikiArt validation to test distribution transfer.
4. Compare `I1` vs `I2` before adding complicated auxiliary losses.
5. After deciding the better injection path, add leakage-control variants.
6. Only then test hierarchical-style extraction.

## Notes
- Use the same prompt/reference/content triplets across checkpoints when comparing stage-2 runs.
- Do not mix validation definitions between experiments; keep the three validation sets fixed.
- When comparing `I1` and `I2`, keep trainable parameter count and training budget as close as possible.
