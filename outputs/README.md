# Outputs Contract

`outputs/full_real/` is reserved for user-generated local full-paper runs.

For this stage, implemented real-full paths are:
- `outputs/full_real/text_distilbert_agnews_seed<seed>/`
- `outputs/full_real/text_distilbert_squad_seed<seed>/`
- `outputs/full_real/image_mobilenetv2_cifar10_seed<seed>/`
- `outputs/full_real/multimodal_clip_flickr30k_seed<seed>/`
- `outputs/full_real/multimodal_vilt_flickr30k_seed<seed>/`

Expected files for each implemented path:

```text
outputs/full_real/<run_name>/
  config_snapshot.yaml
  metrics.json
  metrics.csv
  signature.json
  threshold.json
  training_log.txt
  eval_log.txt
  run_metadata.json
  checkpoints/
  metrics/
    train_metrics.json
    eval_metrics.json
```

Text runs additionally include tokenizer artifacts.
CLIP and ViLT Flickr30K stages are verification-only and may omit training checkpoints.
SQuAD stage in this repository is verification-focused and may omit QA-specific metrics such as EM/F1.

Do not commit large outputs/checkpoints to Git.


Explainability outputs for this stage are saved under:

```text
outputs/full_real/explainability/
  shap/
  captum_ig/
  gradcam/
  scorecam/
  attention_rollout/
  explainability_log.txt
  latest_summary.json
```

Attack outputs for this stage are saved under:

```text
outputs/full_real/attacks/
  pruning/<run_name>/
  finetuning/<run_name>/
  distillation/<run_name>/
```

ONNX deployment outputs for this stage are saved under:

```text
outputs/full_real/deployment/onnx/<run_name>/
  model.onnx
  onnx_export_summary.json
  export_log.txt
  input_example.json (optional)
  onnx_validation.json (optional)
```

TensorRT deployment outputs for this stage are saved under:

```text
outputs/full_real/deployment/tensorrt/<run_name>/
  tensorrt_export_summary.json
  tensorrt_export_log.txt
  model.engine (on success)
```

Latency benchmarking outputs for this stage are saved under:

```text
outputs/full_real/deployment/latency/
  <run_name>_<backend>_latency.json
  <run_name>_<backend>_latency.csv
  latency_log.txt
```

Other outputs documented below remain interface contracts for later stages:

```text
outputs/
  full_real/
    text_distilbert_squad_seed0/
    attacks/
      pruning/
      finetuning/
      distillation/
    explainability/
      shap/
      gradcam/
      scorecam/
      attention_rollout/
    deployment/
      onnx/
      tensorrt/
      latency/
```


Artifact aggregation outputs for this stage are saved under:

```text
outputs/full_real/artifacts/
  tables/
  figures/
  artifact_generation_log.txt
  artifact_sources.json
```

`artifact_sources.json` records source files and source mode for each generated table/figure.
