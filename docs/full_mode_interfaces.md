# Full-Mode Interfaces (Contracts)

This document defines interface contracts for components not fully implemented as paper-faithful deep-learning pipelines yet.

## Explainability interfaces

Expected real outputs under:

```text
outputs/full_real/explainability/
  shap/
    text_distilbert_agnews_seed0_shap.json
  captum_ig/
    text_distilbert_agnews_seed0_captum_ig.json
  gradcam/
    image_mobilenetv2_cifar10_seed0_gradcam_summary.json
  scorecam/
    image_mobilenetv2_cifar10_seed0_scorecam_summary.json
  attention_rollout/
    multimodal_vilt_flickr30k_seed0_attention_summary.json
    multimodal_clip_flickr30k_seed0_attention_summary.json
  latest_summary.json
```

Policy:
- Explainability is supportive audit evidence only.
- Do not claim causal proof.
- Do not claim legal/forensic proof.

## Attack interfaces

Expected real outputs under:

```text
outputs/full_real/attacks/
  pruning/<run_name>/
    attack_metrics.json
    attack_metrics.csv
    attack_log.txt
    config_snapshot.yaml
  finetuning/<run_name>/
    attack_metrics.json
    attack_metrics.csv
    attack_log.txt
    config_snapshot.yaml
  distillation/<run_name>/
    attack_metrics.json
    attack_metrics.csv
    attack_log.txt
    config_snapshot.yaml
```

Each metrics file should include:
- model, dataset, seed
- clean_accuracy
- post_attack_accuracy
- robustness_drop
- retained_robustness
- config reference
- log path

## Deployment interfaces

Expected real outputs under:

```text
outputs/full_real/deployment/
  onnx/<run_name>/
    model.onnx
    onnx_export_summary.json
    export_log.txt
    input_example.json (optional)
    onnx_validation.json (optional)
  tensorrt/<run_name>/
    tensorrt_export_summary.json
    tensorrt_export_log.txt
    model.engine (on success)
  latency/
    <run_name>_<backend>_latency.json
    <run_name>_<backend>_latency.csv
    latency_log.txt
```

Notes:
- CI does not benchmark edge hardware.
- TensorRT export is environment-dependent and must fail clearly if unavailable.
- Do not generate fake latency values.


## Implemented in this stage: real_full text mode

Implemented local path:
- DistilBERT + AG News text classification and watermark verification
- DistilBERT + SQuAD v2.0 verification-focused text watermark path (question/context embedding verification only)
- MobileNetV2 + CIFAR-10 image classification and watermark verification
- CLIP + Flickr30K multimodal **verification-only** pipeline (embedding extraction + watermark verification)
- ViLT + Flickr30K multimodal **verification-only** pipeline (embedding extraction + watermark verification)
- Output roots: `outputs/full_real/text_distilbert_agnews_seed<seed>/`, `outputs/full_real/text_distilbert_squad_seed<seed>/`, `outputs/full_real/image_mobilenetv2_cifar10_seed<seed>/`, `outputs/full_real/multimodal_clip_flickr30k_seed<seed>/`, `outputs/full_real/multimodal_vilt_flickr30k_seed<seed>/`

Required artifacts now produced by this path:
- `config_snapshot.yaml`
- `metrics.json`, `metrics.csv`
- `signature.json`, `threshold.json`
- `training_log.txt`, `eval_log.txt`, `run_metadata.json`

All other interfaces in this document remain contracts for future stages.



## Implemented explainability methods in this stage

- Text (`text_distilbert_agnews_seed*`): token attribution via Captum Integrated Gradients (`--method shap` or `captum_ig`).
- Image (`image_mobilenetv2_cifar10_seed*`): Grad-CAM and Score-CAM style overlay summaries (`--method gradcam` / `scorecam`).
- Multimodal ViLT (`multimodal_vilt_flickr30k_seed*`): attention-based summary from model attentions (`--method attention_rollout`).
- Multimodal CLIP (`multimodal_clip_flickr30k_seed*`): documented alternative summary using token-ablation similarity-drop under `--method attention_rollout` because stable attention maps are not exposed in the selected wrapper backend. This is explicitly partial support and not true attention rollout.

Policy reminder: outputs are supportive audit evidence only, not causal proof and not legal/forensic proof.



Attack support in this stage:
- DistilBERT + AG News: pruning, finetuning, distillation implemented.
- MobileNetV2 + CIFAR-10: pruning, finetuning, distillation implemented.
- CLIP + Flickr30K (verification-only): model-mutation attacks unsupported (fails clearly).
- ViLT + Flickr30K (verification-only): model-mutation attacks unsupported (fails clearly).


ONNX support in this stage:
- DistilBERT + AG News: implemented.
- MobileNetV2 + CIFAR-10: implemented.
- CLIP + Flickr30K (verification-only): unsupported in this stage (fails clearly).
- ViLT + Flickr30K (verification-only): unsupported in this stage (fails clearly).


TensorRT support in this stage:
- DistilBERT + AG News: implemented interface (local TensorRT environment required).
- MobileNetV2 + CIFAR-10: implemented interface (local TensorRT environment required).
- CLIP + Flickr30K (verification-only): unsupported in this stage (fails clearly).
- ViLT + Flickr30K (verification-only): unsupported in this stage (fails clearly).


Latency support in this stage:
- DistilBERT + AG News: pytorch implemented; onnxruntime/tensorrt environment-dependent.
- MobileNetV2 + CIFAR-10: pytorch implemented; onnxruntime/tensorrt environment-dependent.
- CLIP + Flickr30K and ViLT + Flickr30K: unsupported in this stage for latency benchmarking.
- Hardware labels are user-provided context only; CI does not benchmark Jetson/Raspberry Pi hardware.


## Artifact aggregation interfaces (real_full)

Expected generated artifacts under:

```text
outputs/full_real/artifacts/
  tables/
    detection_accuracy_table.csv
    baseline_comparison_table.csv
    attack_robustness_table.csv
    explainability_alignment_table.csv
    threshold_sensitivity_table.csv
    trigger_size_ablation_table.csv
    latency_table.csv
    deployment_export_summary_table.csv
  figures/
    baseline_comparison_figure.txt
    robustness_figure.txt
    threshold_sensitivity_figure.txt
    trigger_size_ablation_figure.txt
    latency_figure.txt
  artifact_generation_log.txt
  artifact_sources.json
```

`artifact_sources.json` must map each generated artifact to source files, source mode, and timestamp.
