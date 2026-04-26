# Reproduction Log

## 2026-04-18 - Stage 1 scaffold + first validated path

### Completed
- Created repository scaffold for configs/src/scripts/tests/docs/outputs/data/.github/workflows.
- Implemented modular configuration loader and run directory setup.
- Implemented deterministic seed utility and run metadata capture.
- Implemented text pipeline baseline with:
  - synthetic text dataset module
  - trigger injection and splitting
  - text embedding model wrapper
  - watermark losses (alignment + separation)
  - signature computation
  - threshold selection by validation F1
  - verification engine
- Added train/eval scripts satisfying required command surface.
- Added smoke/debug/full configs and baseline B1-B4 config examples.
- Added unit tests for signature, losses, thresholding, verification, trigger split, and embedding shapes.

### Deviations / blockers
- AG News/HuggingFace data ingestion is not yet wired in this initial commit; synthetic deterministic dataset is used as bootstrap for validated plumbing.
- DistilBERT, MobileNetV2, CLIP, ViLT wrappers are scaffolded conceptually; initial runnable implementation uses a lightweight text encoder for deterministic local runs.

### Impact
- End-to-end watermark logic is testable and reproducible now.
- Results are not paper-comparable yet; this is an implementation-readiness baseline.

### Next steps
1. Add real AG News data module and DistilBERT wrapper.
2. Add B1-B4 runner script and metric table generation.
3. Expand to CIFAR-10 + MobileNetV2 with visual triggers.
4. Add explainability, attacks, and ONNX/int8 modules.


## 2026-04-20 - Codex reproducibility hardening

### Completed
- Removed hard runtime dependency on external ML packages for smoke/debug/full path.
- Reworked the first validated path to use standard-library-only components so commands run in clean/offline Codex containers.
- Converted config loader to JSON-backed `.yaml` files to avoid PyYAML dependency while preserving required command filenames.
- Updated tests to dependency-free assertions and ensured `pytest -q` passes in current environment.
- Added `requirements-dev.txt`, `pytest.ini`, and updated CI install step.
- Updated `AGENTS.md` and `README.md` with reproducible setup and validation commands.

### Impact
- `pytest -q`, `python scripts/train.py --config configs/{smoke,debug,full}.yaml`, and `python scripts/eval.py --config configs/debug.yaml` now execute successfully in this Codex environment without network package installation.
- The current implementation remains a validated scaffold path and not a full paper-faithful DistilBERT/AGNews implementation yet.


## 2026-04-20 - Frozen results + artifact regeneration implementation

### Completed
- Added `results/paper_results.json` as frozen paper-results registry.
- Added `docs/results_reference.md` describing non-fabrication and regeneration policy.
- Replaced placeholder scripts with runnable artifact-regeneration scripts:
  - `make_tables.py`, `make_figures.py`
  - `run_baselines.py`, `run_attacks.py`, `run_explainability.py`
  - `run_threshold_sensitivity.py`, `run_trigger_size_ablation.py`
  - `benchmark_latency.py`, `export_onnx.py`
- Added `src/results/reference.py` helper module for deterministic CSV/text/log generation.

### Notes
- These scripts regenerate artifacts from frozen references and **do not rerun full paper experiments**.
- Empty tables indicate unavailable or not-yet-curated paper-reported numeric values.


## 2026-04-20 - PR sync conflict-resolution pass

### Completed
- Re-synced conflict-prone files (`README.md`, docs, artifact scripts, wrapper scripts) while preserving frozen-results defaults.
- Preserved and validated `--mode full` graceful dry-run behavior for artifact scripts.
- Kept non-fabrication and non-rerun claims explicit in docs and script logs.


## 2026-04-22 - Local full-mode scaffold execution enabled

### Completed
- Upgraded artifact scripts `--mode full` from dry-run to executable local scaffold computations.
- Added `src/results/full_pipeline.py` for baseline aggregation, attack simulation, and latency benchmarking over local outputs.
- Added pseudo-ONNX JSON export artifact path in full mode.

### Scope clarification
- Full mode now executes end-to-end for the current dependency-free scaffold, but remains distinct from full paper-faithful reproduction with real model stacks (DistilBERT/MobileNetV2/CLIP/ViLT).

## 2026-04-22 - Validation rerun in clean container

### Completed
- Reinstalled development dependencies from `requirements-dev.txt` in the current container.
- Re-ran the full test suite (`pytest -q`) and confirmed all tests pass.
- Re-ran required command contract end-to-end:
  - `python scripts/train.py --config configs/smoke.yaml`
  - `python scripts/train.py --config configs/debug.yaml`
  - `python scripts/train.py --config configs/full.yaml`
  - `python scripts/eval.py --config configs/debug.yaml`

### Impact
- Confirms the repository remains reproducible and executable in a fresh environment with the currently documented command surface.

## 2026-04-22 - Capability audit refresh and real_full failure clarity

### Completed
- Updated `docs/repository_capability_report.md` with a detailed capability matrix using explicit status classes (fully implemented, partial, scaffold/interface, frozen-only, user-prepared-only, missing, risky/unclear).
- Updated `README.md` to add a capability snapshot and to clarify that core train/eval metrics are JSON (CSV generation currently comes from reporting scripts).
- Improved `scripts/validate_full_inputs.py` so missing manifest files are reported as a clean validation error (`missing manifest: ...`) instead of a traceback.

### Impact
- Repository status is now clearer for planning the next stage.
- `execution_mode=real_full` failure behavior is now explicit and user-facing when local prerequisites are absent.

## 2026-04-22 - Stage: real_full DistilBERT + AG News text-mode pipeline

### Completed
- Added manifest-driven real AG News CSV loading (`train.csv`, `validation.csv`, `test.csv`) with required-column checks.
- Added manifest-driven real trigger CSV loading (`ag_news_triggers.csv`) with required-column checks.
- Implemented optional-dependency DistilBERT training/evaluation pipeline for `execution_mode=real_full` text mode.
- Integrated real_full watermark workflow: trigger injection, embedding extraction, signature construction, threshold selection, cosine verification, and watermark error rates.
- Added real_full output artifacts: `metrics.json`, `metrics.csv`, `signature.json`, `threshold.json`, `config_snapshot.yaml`, logs, and run metadata.
- Updated train/eval CLI routing so scaffold modes still run unchanged and unsupported real_full configs fail clearly.
- Added lightweight tests for manifest/data/trigger loading contracts, unsupported/missing real_full failures, and output schema helpers.

### Scope boundary
- This stage implements only DistilBERT + AG News real_full text mode.
- SQuAD/CIFAR-10/CLIP/ViLT/explainability/attacks/deployment/edge latency remain out of scope.

## 2026-04-22 - Stage: real_full MobileNetV2 + CIFAR-10 image-mode pipeline

### Completed
- Added manifest-driven CIFAR-10 data loading with CSV mode (`image_path`,`label`) and folder mode (`<split>/<class>/*`) support.
- Added manifest-driven visual trigger JSON loading and simple patch-based trigger insertion (location + opacity/intensity).
- Implemented optional-dependency MobileNetV2 real_full pipeline for local training/evaluation, embedding extraction, signature creation, threshold selection, and cosine verification.
- Added real_full image routing in train/eval CLIs while keeping scaffold mode unchanged and preserving AG News routing.
- Added required real_full image artifacts: `metrics.json`, `metrics.csv`, `signature.json`, `threshold.json`, config snapshot, logs, and run metadata.
- Added lightweight tests for CIFAR-10 loaders, image trigger loader/application, real_full failure behavior, and routing checks.

### Scope boundary
- This stage implements only MobileNetV2 + CIFAR-10 real_full image mode in addition to existing text real_full.
- SQuAD, CLIP/ViLT, real explainability, attacks, ONNX/TensorRT, and edge latency remain out of scope.

## 2026-04-23 - Stage: real_full CLIP + Flickr30K multimodal verification pipeline

### Completed
- Added manifest-driven Flickr30K CSV loading for `train.csv`, `validation.csv`, and `test.csv` with required schema checks and image existence validation.
- Added multimodal trigger loading for Flickr30K CLIP triggers from CSV or JSON manifest entries.
- Implemented optional-dependency CLIP embedding pipeline (Transformers backend) for multimodal verification-only execution (no CLIP fine-tuning in this stage).
- Implemented real_full multimodal watermark verification flow: fused image/text embeddings, signature construction, threshold selection, cosine scoring, and watermark metrics export.
- Added CLI routing so `eval.py` supports `dataset=flickr30k` + `model=clip`, while `train.py` clearly reports that this stage is verification-only and points to `eval.py`.
- Added lightweight tests for Flickr30K loader, multimodal trigger loader, mocked CLIP embedding interface, and clear real_full failure behavior.

### Scope boundary
- CLIP path in this stage is verification-only.
- SQuAD, ViLT, explainability, attacks, ONNX/TensorRT, and edge latency remain out of scope.

## 2026-04-23 - Stage: real_full ViLT + Flickr30K multimodal verification pipeline

### Completed
- Added ViLT real_full config and eval routing for `dataset=flickr30k` + `model=vilt`.
- Implemented optional-dependency ViLT embedding pipeline (Transformers backend) using pooled multimodal embedding outputs for watermark verification.
- Reused Flickr30K processed CSV loader and extended multimodal trigger loader to support ViLT-specific trigger manifest keys.
- Implemented real_full ViLT verification flow: embedding extraction, signature construction, threshold selection, cosine verification, and metrics export.
- Added train-path guard to clearly report that ViLT stage is verification-only and to point users to `eval.py`.
- Added lightweight tests for ViLT config parsing, mocked embedding interface integration, and clear failure behavior without local data.

### Scope boundary
- ViLT path in this stage is verification-only.
- SQuAD, explainability, attacks, ONNX/TensorRT, and edge latency remain out of scope.

## 2026-04-23 - Stage: real explainability for implemented real_full paths

### Completed
- Implemented real_full explainability CLI routing in `scripts/run_explainability.py` for method + run-dir based execution.
- Added text token attribution path for DistilBERT + AG News using Captum Integrated Gradients (method aliases: `shap` / `captum_ig`).
- Added image explainability summaries for MobileNetV2 + CIFAR-10 with Grad-CAM and Score-CAM style overlays.
- Added multimodal explainability summaries:
  - ViLT attention-based layer attention summary.
  - CLIP documented alternative summary using token-ablation similarity-drop where stable attention maps are not exposed in selected backend wrapper.
- Added explainability output schema with supportive-evidence note and structured summary JSON outputs under `outputs/full_real/explainability/`.
- Added lightweight tests for explainability method routing, run-dir validation, unsupported method/model combinations, and supportive note presence.

### Scope boundary
- Explainability outputs are supportive evidence only; not causal proof; not legal/forensic proof.
- SQuAD, attacks, ONNX/TensorRT, and edge latency remain out of scope.

## 2026-04-23 - Explainability routing hardening and artifact indexing

### Completed
- Hardened `scripts/run_explainability.py` real_full routing to infer target from `config_snapshot.yaml` (`dataset.name` + `model.name`) before run-directory name fallback.
- Added `outputs/full_real/explainability/latest_summary.json` write-out for easier downstream artifact collection.
- Added `captum_ig` output directory support for text attribution outputs and aligned validator checks.
- Added a guardrail error for empty text trigger manifests in real_full text explainability.
- Expanded routing tests to cover config-first inference and summary index generation.

### Impact
- real_full explainability is less brittle when run-directory naming deviates from expected defaults.
- downstream scripts/users get a stable latest-summary pointer without needing to parse method-specific paths.

## 2026-04-23 - Explainability completion pass (real methods + schema tests)

### Completed
- Implemented explicit Score-CAM computation for MobileNetV2 explainability (separate from Grad-CAM path) and preserved method-specific output directories.
- Kept DistilBERT text explainability as real Captum Integrated Gradients token attribution with method alias handling (`shap`/`captum_ig`).
- Kept ViLT attention-based explainability summaries under `attention_rollout` with per-layer attention means.
- Kept CLIP explainability as a documented alternative token-ablation similarity-drop summary (no fake attention map generation).
- Added lightweight method-level tests for text/image/multimodal explainability output schema and artifact path generation.
- Updated README/capability/full-mode interfaces/outputs docs to explicitly separate implemented vs partial/unsupported explainability capabilities.

### Scope boundary
- CLIP true attention-rollout maps remain unsupported in the selected wrapper backend.
- SQuAD explainability, attacks explainability, and deployment explainability remain out of scope.


## 2026-04-23 - Stage: real_full attacks and robustness evaluation

### Completed
- Added real_full attack runner (`scripts/run_attacks.py --mode real_full`) with explicit attack selection (`pruning`, `finetuning`, `distillation`) and required run-dir inputs.
- Implemented attack evaluation core for DistilBERT+AG News and MobileNetV2+CIFAR-10:
  - global magnitude pruning,
  - short local finetuning pass,
  - teacher-student distillation pass.
- Added required attack artifacts under `outputs/full_real/attacks/<attack>/<run_name>/` (`attack_metrics.json`, `attack_metrics.csv`, `attack_log.txt`, `config_snapshot.yaml`).
- Added clear unsupported failure for verification-only CLIP/ViLT attack-mutation combinations in this stage.
- Added lightweight tests for real_full attack CLI routing, arg parsing, run-dir validation/missing-input failures, and unsupported combination failures.

### Scope boundary
- CLIP/ViLT mutation attacks remain unsupported in this stage due verification-only architecture.
- ONNX/TensorRT, real edge latency, and SQuAD remain out of scope.


## 2026-04-23 - Stage: real_full ONNX export

### Completed
- Implemented real_full ONNX export runner (`scripts/export_onnx.py --mode real_full`) with run-dir and opset controls.
- Added deployment exporter (`src/deployment/onnx_export.py`) supporting real model export for:
  - DistilBERT + AG News
  - MobileNetV2 + CIFAR-10
- Added ONNX export outputs under `outputs/full_real/deployment/onnx/<run_name>/` with summary/log/input-example artifacts.
- Added clear unsupported failures for CLIP/ViLT verification-only ONNX export combinations in this stage.
- Added lightweight tests for ONNX CLI routing, missing input failures, unsupported combos, mocked success path, and no-fake-file-on-failure behavior.

### Scope boundary
- TensorRT, edge latency benchmarking, and SQuAD remain out of scope for this stage.


## 2026-04-23 - Stage: real_full TensorRT interface

### Completed
- Added TensorRT export interface module (`src/deployment/tensorrt_export.py`) for ONNX-capable paths.
- Added CLI `scripts/export_tensorrt.py` supporting `--onnx` or `--run-dir`, `--output-dir`, `--fp16`, and `--workspace`.
- Implemented required TensorRT artifacts under `outputs/full_real/deployment/tensorrt/<run_name>/` including summary and log files, with `model.engine` only on successful conversion.
- Added clear unsupported failures for CLIP/ViLT verification-only TensorRT conversions in this stage.
- Added lightweight tests for CLI routing, missing file failures, TensorRT-unavailable behavior, mocked success path, and no-fake-engine-on-failure behavior.

### Scope boundary
- Real edge latency benchmarking and SQuAD remain out of scope for this stage.


## 2026-04-23 - Stage: real_full latency benchmarking

### Completed
- Implemented real_full latency benchmarking module (`src/deployment/latency.py`) with backend routing (`pytorch`, `onnxruntime`, `tensorrt`) and real local timing (warmup + measured runs).
- Updated `scripts/benchmark_latency.py` to support `--mode real_full`, backend selection, run-dir input, hardware label, warmup/measured runs, batch size, and backend artifact paths.
- Added latency outputs under `outputs/full_real/deployment/latency/` with JSON/CSV metrics and log stream.
- Added clear failure behavior for missing inputs/dependencies and unsupported multimodal verification-only paths.
- Added lightweight tests for latency CLI routing, missing-file failures, mocked measurement path, backend selection, and output schema presence.

### Scope boundary
- SQuAD and real-output artifact aggregation/refinement remain out of scope for this stage.


## 2026-04-23 - Stage: real_full artifact aggregation from local outputs

### Completed
- Added real_full artifact aggregation helpers in `src/results/real_full_artifacts.py`.
- Updated `scripts/make_tables.py` and `scripts/make_figures.py` with explicit mode separation: `frozen`, `full`, `real_full`.
- Implemented real_full table aggregation from validated local outputs under `outputs/full_real/` for detection, baselines, attacks, explainability alignment, threshold, trigger-size, latency, and deployment export summaries.
- Implemented real_full figure generation (text-based figure summaries consistent with current repository convention).
- Added source tracking artifacts under `outputs/full_real/artifacts/` (`artifact_sources.json`, `artifact_generation_log.txt`).
- Added strict/non-strict behavior for missing source artifacts.
- Added tests for real_full artifact generation, source tracking, missing/null behavior, strict mode failures, frozen mode compatibility, and no silent fallback.

### Scope boundary
- This stage aggregates existing local outputs; it does not claim full paper-faithful reruns.
- Missing local source outputs remain explicitly missing/null unless strict mode is requested.


## 2026-04-23 - Stage: real_full DistilBERT + SQuAD v2.0 verification pipeline

### Completed
- Added real_full SQuAD v2.0 processed JSONL loading from manifest with required-field validation and clear missing/schema failures.
- Added SQuAD text trigger CSV loading from manifest with required-column validation.
- Added DistilBERT + SQuAD real_full verification-focused training/evaluation path using question+context text inputs, watermark signature construction, threshold selection, and cosine verification metrics.
- Added `configs/full_real_distilbert_squad.example.yaml` and updated CLI routing in `train.py` / `eval.py` for `dataset=squad_v2` + `model=distilbert*`.
- Added lightweight tests for config parsing, SQuAD loader/trigger contracts, missing-data failures, unsupported combinations, and metrics schema.

### Scope boundary
- This stage is verification-focused for SQuAD watermark detection and does not implement/claim full QA fine-tuning with EM/F1 metrics.
- CI remains scaffold-only and does not download SQuAD or run heavy optional-dependency training.


## 2026-04-23 - Stage: final cleanup, consistency review, and release-readiness audit

### Completed
- Performed cross-check of docs vs code for scaffold mode, frozen/reference artifact mode, and real_full local mode boundaries.
- Cleaned README mode framing and added a grouped command quick reference covering install, validation, real_full paths, deployment scripts, and artifact generation.
- Corrected wording inconsistencies that could imply obsolete dry-run behavior and clarified CIFAR-10 train/eval availability.
- Added final concise status registry in `docs/repository_capability_report.md` (fully implemented / partial / verification-only / unsupported / prerequisites).
- Added explicit SQuAD scope assumption in `docs/assumptions.md` and updated output-contract wording to avoid overclaiming QA EM/F1 support.

### Scope boundary
- This audit stage does not add new experimental features.
- It preserves source-tracking and non-fabrication guarantees.
