# Data directory

This repository supports three execution modes:

1. Frozen/reference-results regeneration
2. Scaffold/smoke synthetic validation
3. Real paper-level local runs

For real paper-level runs, user must prepare datasets locally.

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

Do not commit large datasets. `data/raw/` and `data/processed/` are gitignored.

See also: `docs/dataset_preparation.md`.
