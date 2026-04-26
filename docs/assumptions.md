# Assumptions Log

## A-001: Initial validated path modality
- **Type**: safe default
- **Choice**: Start with text modality only for first end-to-end validated path.
- **Reasoning**: CODEX_HANDOFF prioritizes incremental validated progress and identifies DistilBERT/AG News as first critical target.
- **Configurable**: Yes (`experiment.modality`, model/data modules).

## A-002: Dataset availability fallback
- **Type**: safe default
- **Choice**: Use deterministic synthetic text classification dataset for smoke/debug/full bootstrap when external AG News data is not present.
- **Reasoning**: Enables non-network, deterministic CI/smoke validation while preserving pipeline behavior.
- **Configurable**: Yes (`dataset.name`, `dataset.synthetic.*`).

## A-003: Signature method
- **Type**: explicit from paper / safe default
- **Choice**: Use centroid of normalized trigger embeddings as the watermark signature.
- **Reasoning**: Aligned with handoff guidance and cosine verification.
- **Configurable**: Yes (`watermark.signature_method`).

## A-004: Threshold selection
- **Type**: explicit from paper
- **Choice**: Select threshold on validation using max F1 via sweep.
- **Reasoning**: Required in handoff.
- **Configurable**: Yes (`watermark.threshold_search_step`, `watermark.threshold_selection`).

## A-005: Loss defaults
- **Type**: safe default
- **Choice**: `lambda_align=1.0`, `lambda_sep=0.1`, `margin_gamma=0.2`.
- **Reasoning**: Paper omits exact values; these match handoff-recommended defaults.
- **Configurable**: Yes (`watermark.*`).


## A-006: Dependency-free bootstrap path
- **Type**: safe default
- **Choice**: For Codex reproducibility, the initial smoke/debug/full pipeline uses only Python standard library components.
- **Reasoning**: Current Codex environment can block package installation; dependency-free path guarantees runnable validation commands.
- **Configurable**: Yes (future phases can switch to torch/transformers by replacing model/data modules).

## A-007: JSON-formatted `.yaml` configs
- **Type**: safe default
- **Choice**: Keep required `.yaml` filenames but store JSON content parseable via stdlib `json`.
- **Reasoning**: Preserves command contract while removing PyYAML dependency.
- **Configurable**: Yes (can revert to standard YAML parser when environment guarantees dependencies).


## A-008: Frozen paper-reported results registry
- **Type**: inferred from handoff reproducibility requirements
- **Choice**: Maintain immutable/reference values in `results/paper_results.json` and regenerate artifacts from it.
- **Reasoning**: Supports deterministic artifact regeneration without claiming full reruns.
- **Configurable**: Yes (`--paper-results` argument in artifact scripts).


## A-009: Full-mode local-scaffold policy
- **Type**: safe default
- **Choice**: Artifact scripts accept `--mode full` and run local scaffold computations from generated run artifacts (still not full paper-faithful deep-learning reruns).
- **Reasoning**: Prevents overclaiming capabilities while preserving forward-compatible CLI surface.
- **Configurable**: Yes (`--mode` flag in artifact scripts).

## A-010: Explainability routing source of truth
- **Type**: safe default
- **Choice**: In `--mode real_full`, explainability routing uses `config_snapshot.yaml` (`dataset.name`, `model.name`) first, with run-directory naming only as fallback.
- **Reasoning**: Avoids brittle behavior when local run directory names vary while preserving backward compatibility.
- **Configurable**: Partially (via config snapshot fields and future router extension).

## A-011: CLIP explainability fallback mode
- **Type**: inferred from implementation constraints
- **Choice**: In current `real_full` CLIP wrapper/backend, true cross-modal attention maps are treated as unsupported; explainability uses token-ablation similarity-drop summaries instead.
- **Reasoning**: Prevents fabricated attention outputs while still providing meaningful supportive attribution signals.
- **Configurable**: Yes (future backend with stable attention exposure can replace fallback).

## A-012: Attack support boundary for verification-only multimodal paths
- **Type**: safe default
- **Choice**: For CLIP/ViLT verification-only pipelines in this stage, model-mutation attacks (pruning/finetuning/distillation) are treated as unsupported and fail clearly.
- **Reasoning**: Current CLIP/ViLT implementations are verification-focused embedders without a stable local fine-tuning/distillation training stack in this repository stage.
- **Configurable**: Yes (future stage can add CLIP/ViLT attack training stack and remove this boundary).

## A-013: ONNX export support boundary for verification-only multimodal paths
- **Type**: safe default
- **Choice**: ONNX export in this stage is limited to DistilBERT+AG News and MobileNetV2+CIFAR-10; CLIP/ViLT verification-only paths fail clearly as unsupported.
- **Reasoning**: Current CLIP/ViLT implementations focus on local verification embeddings without a stable ONNX-export contract in this repository stage.
- **Configurable**: Yes (future stage can add explicit CLIP/ViLT ONNX export support).

## A-014: TensorRT environment dependency
- **Type**: safe default
- **Choice**: TensorRT export is implemented as a local interface requiring external NVIDIA TensorRT Python bindings/system support; CI does not install/provide it.
- **Reasoning**: TensorRT is platform-specific and not reliably pip-installable across generic CI/container environments.
- **Configurable**: Partially (local environment setup can enable real conversion).

## A-015: Latency backend environment dependency
- **Type**: safe default
- **Choice**: Real-full latency benchmarking supports backend selection, but onnxruntime/tensorrt measurements are environment-dependent and fail clearly when runtime/deployment artifacts are unavailable.
- **Reasoning**: Deployment backends are platform-specific and cannot be guaranteed in generic CI/container environments.
- **Configurable**: Yes (local environment can enable additional backend coverage).

## A-016: SQuAD scope boundary in this stage
- **Type**: safe default
- **Choice**: DistilBERT + SQuAD v2.0 path is implemented as verification-focused watermarking over question/context embeddings, not full QA EM/F1 fine-tuning/evaluation.
- **Reasoning**: Preserves honest capability claims while enabling local real_full watermark verification on SQuAD-shaped data without fabricating QA metrics.
- **Configurable**: Yes (future stage can add a full QA training/evaluation stack and explicit EM/F1 metrics).
