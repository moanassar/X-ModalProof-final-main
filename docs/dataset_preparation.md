# Dataset Preparation Contract (User-Prepared)

This repository does **not** auto-download large datasets in CI.

Required datasets for paper-level reproduction:
- AG News
- SQuAD v2.0
- CIFAR-10
- Flickr30K
- Trigger sources: The Pile, COCO, Visual Genome

Expected structure:

```text
data/
  raw/
    ag_news/
    squad_v2/
    cifar10/
    flickr30k/
    trigger_sources/
      pile/
      coco/
      visual_genome/
  processed/
    text/
      ag_news/
      squad_v2/
    image/
      cifar10/
    multimodal/
      flickr30k/
    triggers/
      text/
      image/
      multimodal/
  manifests/
    datasets_manifest.example.yaml
    datasets_manifest.yaml
```

## Real-full text stage (implemented): DistilBERT + AG News

Required files:
- `data/processed/text/ag_news/train.csv`
- `data/processed/text/ag_news/validation.csv`
- `data/processed/text/ag_news/test.csv`
- `data/processed/triggers/text/ag_news_triggers.csv`

AG News CSV columns:
- `text`, `label`

Text trigger CSV required columns:
- `trigger_id`, `trigger_text`

## Real-full text stage (implemented): DistilBERT + SQuAD v2.0 (verification-focused)

Required files:
- `data/processed/text/squad_v2/train.jsonl`
- `data/processed/text/squad_v2/validation.jsonl`
- `data/processed/text/squad_v2/test.jsonl`
- `data/processed/triggers/text/squad_v2_triggers.csv`

Required JSONL fields per line:
- `id`
- `question`
- `context`
- `answer_text`
- `answer_start`
- `is_unanswerable`
- `split`

Required SQuAD trigger CSV fields:
- `trigger_id`, `trigger_text`

This stage is **verification-focused** for watermark detection and does not claim full QA fine-tuning or EM/F1 reporting.

## Real-full image stage (implemented): MobileNetV2 + CIFAR-10

Supported CIFAR-10 processed input formats:

1) CSV mode:
- `data/processed/image/cifar10/train.csv`
- `data/processed/image/cifar10/validation.csv`
- `data/processed/image/cifar10/test.csv`
- columns: `image_path`, `label`

2) Folder mode:
- `data/processed/image/cifar10/train/<class>/*`
- `data/processed/image/cifar10/validation/<class>/*`
- `data/processed/image/cifar10/test/<class>/*`

Required image trigger file:
- `data/processed/triggers/image/cifar10_triggers.json`

Required trigger fields:
- `trigger_id`
- `trigger_type`
- `patch_path` or `patch_spec`

Optional trigger fields:
- `location`
- `opacity` or `intensity`
- `target_label` or `expected_label`
- `split` or `usage`

Both real_full paths fail clearly when required local files/columns are missing.


## Real-full multimodal stage (implemented): CLIP + Flickr30K (verification-only)

Required files:
- `data/processed/multimodal/flickr30k/train.csv`
- `data/processed/multimodal/flickr30k/validation.csv`
- `data/processed/multimodal/flickr30k/test.csv`

Required CSV columns:
- `image_path`, `caption`, `image_id`, `split`

Required multimodal triggers file (CSV or JSON via manifest):
- `data/processed/triggers/multimodal/flickr30k_clip_triggers.csv` (recommended)

Trigger required fields:
- `trigger_id`
- `trigger_type`
- `trigger_image_path` or `image_path`
- `trigger_caption` or `caption`

This stage is **verification-only** (CLIP embedding extraction + watermark verification), not CLIP fine-tuning.


## Real-full multimodal stage (implemented): ViLT + Flickr30K (verification-only)

Uses the same Flickr30K processed CSV files as CLIP stage and the same trigger schema shape (pair triggers), with ViLT-specific trigger manifest keys when desired:
- `processed.flickr30k_vilt_triggers_csv`
- `processed.flickr30k_vilt_triggers_json`

This stage is **verification-only** (ViLT embedding extraction + watermark verification), not ViLT fine-tuning.

## Rules
- Do not commit large data files.
- Keep raw/processed datasets local.
- GitHub Actions is for scaffold validation only.
- Run full-input validation before real experiments.

## Remaining unimplemented real pipelines
- Full QA fine-tuning + EM/F1 paper-faithful SQuAD reproduction
- ONNX/TensorRT real deployment
- real edge latency benchmarking
