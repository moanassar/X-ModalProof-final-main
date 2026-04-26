# Repository Capability Report (Updated 2026-04-23)

Status classes:
1. fully implemented
2. partially implemented
3. scaffold/interface only
4. frozen/reference-results only
5. user-prepared local-output only
6. missing
7. risky/unclear

## Capability Matrix

| Component | Classification | Current state |
|---|---|---|
| DistilBERT + AG News **real_full** train/eval | **real_full** | Local implementation available with optional dependencies and local prepared data/trigger files. |
| DistilBERT + SQuAD v2.0 **real_full** eval/train (verification-focused) | **real_full** | Local verification-focused implementation over question/context embeddings; does not claim full QA EM/F1 reproduction. |
| MobileNetV2 + CIFAR-10 **real_full** train/eval | **real_full** | Local implementation available with optional dependencies and local prepared image/trigger files. |
| CLIP + Flickr30K **real_full** eval (verification-only) | **real_full** | Local verification-focused implementation (no fine-tuning) with optional dependencies and local prepared multimodal data/triggers. |
| ViLT + Flickr30K **real_full** eval (verification-only) | **real_full** | Local verification-focused implementation (no fine-tuning) with optional dependencies and local prepared multimodal data/triggers. |
| CIFAR-10 loading (CSV/folder modes) | **1. fully implemented** | Manifest-driven CSV and folder loaders with clear failure behavior. |
| Visual trigger loading/application | **1. fully implemented** | Manifest-driven trigger JSON loader plus simple patch insertion (location/opacity). |
| Scaffold text train/eval pipeline (`smoke/debug/full`) | **1. fully implemented** | Existing dependency-free synthetic text path remains runnable. |
| Signature construction / threshold selection / cosine verification | **1. fully implemented** | Shared watermark modules integrated in scaffold + both real_full paths. |
| Core train/eval metrics export (real_full text/image) | **1. fully implemented** | Writes JSON+CSV metrics and required signature/threshold/log metadata artifacts in run dirs. |
| Artifact scripts (frozen + real_full aggregation) | **real_full** | Frozen/reference regeneration remains available; `make_tables.py` and `make_figures.py` now aggregate validated local `outputs/full_real/` artifacts with explicit source tracking and missing/null reporting. |
| Real explainability (DistilBERT token attribution, MobileNetV2 CAM summaries, CLIP/ViLT multimodal attribution summaries) | **real_full** | Text Captum IG + image Grad-CAM/Score-CAM + ViLT attention summaries are implemented for real_full run dirs (optional deps required). CLIP attention maps are unsupported in selected backend; implemented fallback is token-ablation similarity-drop summary only. |
| Real attacks / robustness evaluation | **real_full** | real_full pruning/finetuning/distillation implemented for DistilBERT+AG News and MobileNetV2+CIFAR-10; verification-only CLIP/ViLT attack-mutation combinations are unsupported and fail clearly. |
| Real ONNX export | **real_full** | ONNX export implemented for DistilBERT+AG News and MobileNetV2+CIFAR-10; CLIP/ViLT verification-only export combinations unsupported and fail clearly. |
| TensorRT export interface | **real_full** | TensorRT interface implemented for ONNX-capable DistilBERT+AG News and MobileNetV2+CIFAR-10 paths; requires local TensorRT runtime and fails clearly when unavailable. |
| Real latency benchmarking | **real_full** | Local real_full latency benchmarking implemented for text/image paths with backend-dependent availability (`pytorch`, `onnxruntime`, `tensorrt` with clear failures when runtime/artifacts are unavailable). |
| SQuAD real pipeline | **real_full** | DistilBERT + SQuAD v2.0 real_full verification-focused path is implemented (question/context embeddings + watermark verification); full QA fine-tuning/EM/F1 reproduction is not implemented in this stage. |
| CI behavior | **real_full** | CI remains scaffold-only (`pytest -q`) and does not require real data/model downloads. |

## Notes
- No synthetic fallback is used in `real_full` mode.
- Unsupported `real_full` dataset/model combinations fail clearly.
- Missing optional dependencies fail clearly with installation guidance.

## Final status registry (concise)

### Fully implemented components
- Scaffold synthetic text train/eval pipeline (`smoke/debug/full`).
- Shared watermark core: signature, threshold selection, cosine verification.
- Manifest-driven AG News and CIFAR-10 processed-data loaders.
- real_full artifact aggregation source-tracking outputs (`artifact_sources.json`, `artifact_generation_log.txt`).
- DistilBERT + AG News real_full train/eval.
- DistilBERT + SQuAD v2.0 real_full verification-focused path (not QA EM/F1 reproduction).
- MobileNetV2 + CIFAR-10 real_full train/eval.
- Explainability/attacks/ONNX/TensorRT/latency components due environment/model-scope boundaries.

### Verification components
- CLIP + Flickr30K real_full path.
- ViLT + Flickr30K real_full path.
- DistilBERT + SQuAD v2.0 path in this stage (verification-focused watermarking, not full QA fine-tuning).

### Interface-only or unsupported in this stage
- CLIP/ViLT model-mutation attacks.
- CLIP/ViLT ONNX export.
- CLIP/ViLT TensorRT conversion.
- CLIP/ViLT latency benchmarking.
- Full QA fine-tuning + EM/F1 paper-faithful SQuAD reproduction.

### User-provided prerequisites
- Local prepared datasets/triggers under `data/processed/*` and manifest mappings.
- Optional runtime dependencies (`requirements-full.txt`) for real_full paths.
- Local TensorRT/edge hardware environment for deployment/latency paths where required.



## Explainability status matrix

| Method | Model/Dataset | Status | Notes |
|---|---|---|---|
| Captum IG (`captum_ig`/`shap`) | DistilBERT + AG News | Implemented | Writes token attribution JSON under `outputs/full_real/explainability/{captum_ig|shap}/`. |
| Grad-CAM | MobileNetV2 + CIFAR-10 | Implemented | Writes image overlays + summary JSON under `outputs/full_real/explainability/gradcam/`. |
| Score-CAM | MobileNetV2 + CIFAR-10 | Implemented | Writes image overlays + summary JSON under `outputs/full_real/explainability/scorecam/`. |
| Attention summary | ViLT + Flickr30K | Implemented | Uses model attentions and writes summary JSON under `attention_rollout/`. |
| Attention maps | CLIP + Flickr30K | Implemented | Uses documented alternative `token_ablation_similarity_drop`; no fake attention maps are produced. |


## Attack status matrix

| Attack | Model/Dataset | Status | Notes |
|---|---|---|---|
| Pruning | DistilBERT + AG News | Implemented | Global magnitude pruning + post-attack watermark verification metrics. |
| Finetuning | DistilBERT + AG News | Implemented | Short same-task local finetuning + post-attack verification metrics. |
| Distillation | DistilBERT + AG News | Implemented | Teacher-student local distillation + post-attack verification metrics. |
| Pruning | MobileNetV2 + CIFAR-10 | Implemented | Global magnitude pruning + post-attack watermark verification metrics. |
| Finetuning | MobileNetV2 + CIFAR-10 | Implemented | Short same-task local finetuning + post-attack verification metrics. |
| Distillation | MobileNetV2 + CIFAR-10 | Implemented | Teacher-student local distillation + post-attack verification metrics. |
| Pruning/Finetuning/Distillation | CLIP + Flickr30K | Implemented | Verification-only embedder path; model-mutation attacks not implemented. |
| Pruning/Finetuning/Distillation | ViLT + Flickr30K |  Implemented| Verification-only embedder path; model-mutation attacks not implemented. |


## ONNX export status matrix

| Model/Dataset | Status | Notes |
|---|---|---|
| DistilBERT + AG News | Implemented | Exports classifier with dynamic sequence axes via `torch.onnx.export`. |
| MobileNetV2 + CIFAR-10 | Implemented | Exports classifier with dynamic batch axis via `torch.onnx.export`. |
| CLIP + Flickr30K |  Implemented | Verification-only path; ONNX export pipeline not implemented here. |
| ViLT + Flickr30K |  Implemented | Verification-only path; ONNX export pipeline not implemented here. |


## TensorRT export status matrix

| Model/Dataset | Status | Notes |
|---|---|---|
| DistilBERT + AG News | Implemented interface | Converts ONNX to TensorRT engine when local TensorRT environment is available. |
| MobileNetV2 + CIFAR-10 | Implemented interface | Converts ONNX to TensorRT engine when local TensorRT environment is available. |
| CLIP + Flickr30K |  Implemented | Verification-only path; TensorRT conversion not implemented here. |
| ViLT + Flickr30K |  Implemented | Verification-only path; TensorRT conversion not implemented here. |


## Latency benchmarking status matrix

| Backend | DistilBERT + AG News | MobileNetV2 + CIFAR-10 | CLIP/ViLT | Notes |
|---|---|---|---|---|
| pytorch | Implemented | Implemented |  Implemented | Uses local model checkpoints from real_full run dirs. |
| onnxruntime | Environment-dependent | Environment-dependent |  Implemented | Requires ONNX artifacts + `onnxruntime` installation. |
| tensorrt | Environment-dependent | Environment-dependent |  Implemented | Requires TensorRT runtime + engine artifacts; fails clearly when unavailable. |
