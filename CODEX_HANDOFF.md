# CODEX_HANDOFF.md

# X-ModalProof Implementation Handoff for Codex

This document is a GitHub-safe implementation specification for reproducing the experiments described in the paper:

**X-ModalProof: Real-Time Explainable Ownership Verification for Multimodal and Edge-Deployed AI Models**

•	## Fast development and validation mode
•	Before any full reproduction run, implement:
•	
•	### Smoke mode
•	- tiny reproducible subset
•	- 1 epoch
•	- minimal outputs
•	- no expensive artifacts
•	- verifies end-to-end pipeline correctness
•	
•	### Debug subset mode
•	- small reproducible subset of the real data
•	- short training schedule
•	- reduced-cost evaluation
•	- used to validate training/evaluation logic before full runs
•	
•	### Full mode
•	- exact paper-oriented settings
•	- full data
•	- full metrics
•	- full artifacts

Formatting choices used in this file:

- GitHub-compatible Markdown tables.
- Plain-text formulas for implementation logic.
- Math blocks only for important equations.
- No complex LaTeX inside tables.
- Missing details are explicitly marked as `not specified in the paper`.
- Assumptions are clearly marked as `recommended assumption` or `inferred, not explicitly stated`.

---

# 1. Paper Identity

## Full title

**X-ModalProof: Real-Time Explainable Ownership Verification for Multimodal and Edge-Deployed AI Models**

## Author

- Mohammad Othman Nassar

## Affiliation

- Cyber Security Department, Amman Arab University, Amman, Jordan

## Publication venue / year

- Not specified in the paper.

## Main problem addressed

The paper addresses ownership verification for AI models deployed in black-box, multimodal, and edge-device settings. The central problem is how a model owner can later verify that a deployed, copied, modified, pruned, fine-tuned, distilled, or remotely accessed model contains the owner's watermark without requiring access to internal parameters.

## Claimed contributions

Directly stated or clearly described in the paper:

- A unified framework called **X-ModalProof** for ownership verification.
- Watermark embedding during model training or fine-tuning.
- Inference-time verification using embedding-level similarity.
- Black-box-style verification without direct access to internal weights.
- Support for text, image, and multimodal models.
- Use of SHAP, Grad-CAM, Score-CAM, and attention-based attribution as supportive explanation tools.
- Edge deployment compatibility using ONNX, TensorRT simulation, and int8 quantization.
- Robustness testing under pruning, fine-tuning, and distillation.
- Evaluation on AG News, SQuAD v2.0, CIFAR-10, Flickr30K, and trigger sets derived from The Pile, COCO, and Visual Genome.
- Internal baseline study using B1, B2, B3, and B4 configurations.

## Type of work

- Model watermarking
- Ownership verification
- Multimodal system
- NLP
- Computer vision
- Vision-language modeling
- Explainable AI
- Edge AI deployment
- Robustness evaluation
- Benchmarking-style experimental paper

---

# 2. One-Paragraph Technical Core

X-ModalProof embeds ownership watermarks into the latent representation space of a target AI model during final fine-tuning. For each modality, a trigger set is constructed: rare phrases for text, subtle or low-frequency visual triggers for images, and semantically unusual image-caption pairs for multimodal models. During training, trigger-induced embeddings are encouraged to align with a stored watermark signature vector while benign inputs are encouraged to remain separated from that signature. At verification time, the deployed model is queried with trigger inputs through an inference interface that exposes embedding-level outputs or pooled latent vectors. Each trigger embedding is compared with the stored signature using cosine similarity. The mean similarity over all triggers is compared with a modality-specific threshold selected on validation data by maximizing F1-score. If the mean similarity is greater than or equal to the threshold, the model is classified as watermarked. SHAP, Grad-CAM, Score-CAM, and attention rollout are then used as post-hoc explanation tools to provide supportive attribution evidence, not causal or legal proof.

---

# 3. Reproduction Goal

## Recommended reproduction target

The best practical target is **partial-to-full experimental reproduction**, because several implementation details are not specified in the paper.

## Successful reproduction should include

- Full implementation of watermark embedding during fine-tuning.
- Signature vector construction.
- Black-box-style inference-time verification using cosine similarity.
- Validation-based threshold selection.
- Text, image, and multimodal trigger handling.
- Internal baselines B1 to B4.
- Robustness tests under pruning, fine-tuning, and distillation.
- SHAP, Grad-CAM, Score-CAM, and attention rollout explanation outputs.
- ONNX export and int8 deployment-style latency benchmarking.
- Generation of paper-style tables and figures.

## Outputs to reproduce

- Watermark detection accuracy table.
- Robustness after fine-tuning, pruning, and distillation.
- Attribution alignment scores.
- Threshold sensitivity table.
- Trigger-set size ablation table.
- Latency table and plot.
- Internal baseline comparison table.
- Radar chart.
- SHAP token attribution visualization.
- Grad-CAM or Score-CAM image heatmap visualization.
- Repeatability table with mean and standard deviation.

## Important limitation

Exact numerical reproduction is unlikely unless the missing details are defined by the implementation team. The paper does not specify exact hyperparameters, exact trigger lists, exact threshold values, exact loss weights, exact random seeds, or exact train-validation-test splits.

---

# 4. Problem Formulation

## Task definition

Given a deployed AI model and a set of owner-controlled trigger inputs, determine whether the model contains the owner's watermark.

## Modalities

The paper uses:

- Text
- Image
- Multimodal image-text

## Inputs

Directly stated:

- Training dataset `D`
- Trigger set `T_m = {t_1, ..., t_k}`
- Modality-specific model `f_theta^(m)(.)`
- Watermark signature vector `S_m`
- Similarity threshold `tau_m`
- Trigger inputs
- Benign non-trigger inputs
- Model embeddings exposed through an inference interface

## Outputs

Directly stated:

- Watermarked model `f_theta*`
- Verification decision:
  - `True`: watermark present
  - `False`: watermark absent
- Attribution map or explanation output

Recommended implementation outputs:

- Mean similarity score.
- Per-trigger similarity scores.
- Threshold used.
- False positive rate.
- Detection accuracy.
- Robustness score.
- Attribution alignment score.
- Latency measurement.

## Label space and target variables

Directly stated:

| Dataset | Task / label information |
|---|---|
| AG News | Four-class news classification |
| CIFAR-10 | Standard 10-class image classification |
| SQuAD v2.0 | Text question answering, used for trigger-based inference verification |
| Flickr30K | Image-caption / multimodal evaluation |

Not specified in the paper:

- Exact AG News label mapping.
- Exact CIFAR-10 preprocessing and label encoding.
- Exact SQuAD v2.0 training or inference task formulation.
- Exact Flickr30K objective: retrieval, matching, contrastive learning, or classification.
- Exact class distribution used in train, validation, and test sets.

## Assumptions

Directly stated:

- The attacker may copy full or partial model weights.
- The attacker may prune the model.
- The attacker may fine-tune the model.
- The attacker may distill the model into a smaller student.
- Verification occurs in a black-box or deployment-style setting.
- Model weights and hidden internal states are not required at verification time.
- Embedding outputs are assumed to be exposed by the deployed inference pipeline.

Recommended assumption:

- The deployed model interface must provide an embedding vector or pooled feature vector. If it exposes only class labels, X-ModalProof cannot be implemented as described.

## Notation glossary

| Symbol / term | Meaning |
|---|---|
| `m` | Modality: text, image, or multimodal |
| `f_theta^(m)(.)` | Modality-specific model or embedding function |
| `theta` | Model parameters |
| `D` | Training dataset |
| `T_m` | Trigger set for modality `m` |
| `t_i` | Individual trigger input |
| `k` | Number of triggers |
| `S_m` | Stored watermark signature vector |
| `e_t` | Embedding produced by the model for trigger `t` |
| `tau_m` | Modality-specific verification threshold |
| `mu` | Mean cosine similarity over trigger queries |
| `gamma` | Margin for separation of benign samples; mentioned conceptually but not numerically specified |
| `L_task` | Primary task loss |
| `L_align` | Alignment loss between trigger embedding and signature |
| `L_sep` | Separation loss for benign embeddings |

## Formal objective

The paper describes watermark embedding as a regularized optimization problem that preserves task utility while increasing consistency of trigger representations and separating benign activations.

Implementation-friendly formula:

```text
total_loss = task_loss
           + lambda_align * alignment_loss
           + lambda_sep * separation_loss
```

Where:

- `task_loss` is cross-entropy, contrastive loss, or another task-specific loss.
- `alignment_loss` encourages trigger embeddings to match the watermark signature.
- `separation_loss` discourages benign inputs from matching the watermark signature.
- `lambda_align` and `lambda_sep` are weighting coefficients.

Not specified in the paper:

- Exact values for `lambda_align`.
- Exact values for `lambda_sep`.
- Exact value for `gamma`.
- Exact mathematical form of the full loss due to incomplete equation rendering in the manuscript.

---

# 5. Method / Model Breakdown

## Full pipeline overview

1. Select modality:
   - Text
   - Image
   - Multimodal
2. Select target model:
   - DistilBERT for text
   - MobileNetV2 for image
   - CLIP for multimodal
   - ViLT for multimodal
3. Load public dataset:
   - AG News
   - SQuAD v2.0
   - CIFAR-10
   - Flickr30K
4. Construct modality-specific trigger set:
   - Rare phrases from The Pile
   - Long-tail visual combinations from COCO
   - Rare image-caption pairs from Visual Genome
5. Insert triggers during final model fine-tuning.
6. Train using task loss plus watermark alignment and separation losses.
7. Compute or update watermark signature vector `S_m`.
8. Select threshold `tau_m` on a validation set by maximizing F1-score.
9. Export model to ONNX.
10. Apply optional int8 quantization.
11. Run trigger queries through deployed inference interface.
12. Extract embeddings.
13. Compute cosine similarity between trigger embeddings and `S_m`.
14. Aggregate scores using the mean.
15. Compare mean score against `tau_m`.
16. Return ownership verification decision.
17. Generate attribution explanations.
18. Evaluate robustness under pruning, fine-tuning, and distillation.
19. Benchmark latency on edge-style deployment.

## Watermark Embedding Module

Directly stated:

- Watermark signals are embedded during training or fine-tuning.
- Text triggers are rare but coherent phrases.
- Image triggers involve subtle changes in low-frequency areas.
- Trigger inputs produce specific activation patterns.
- Model outputs do not need explicit modification.
- Watermark is embedded in latent representation space.

Implementation steps:

```text
For each training batch:
    1. Load normal samples.
    2. Add or include trigger samples according to trigger_ratio.
    3. Construct triggered inputs.
    4. Run model forward pass.
    5. Extract embeddings.
    6. Compute task loss.
    7. Compute alignment loss for trigger embeddings.
    8. Compute separation loss for benign embeddings.
    9. Combine losses.
    10. Backpropagate and update model parameters.
```

## Multimodal AI Backbone

Directly stated:

- CLIP is used because it learns joint vision-language embeddings.
- ViLT is used because it fuses visual and textual inputs through transformer attention without a separate visual backbone.
- DistilBERT and MobileNetV2 are used for modality-specific evaluation.
- The framework uses native model representations rather than adding a separate late-fusion module.

## Feature Fusion Mechanism

Directly stated:

- CLIP performs feature alignment through contrastive learning.
- ViLT uses cross-modal transformer attention.
- Verification operates in the joint embedding space.
- Late fusion is not required.

Implementation guidance:

```text
For CLIP:
    Use the projected image embedding and/or projected text embedding.
    Normalize embeddings before similarity computation.

For ViLT:
    Use the final CLS token or pooled joint representation.
    Use attention rollout for explanation.
```

## Real-Time Verification Engine

Directly stated:

- Uses predefined trigger inputs.
- Extracts embeddings from model interface.
- Compares embeddings to registered signature.
- Uses cosine similarity.
- Aggregates scores using mean.
- Declares watermark if mean similarity is above threshold.

Implementation steps:

```text
Input:
    model
    trigger_set
    signature
    threshold

Procedure:
    scores = []
    for trigger in trigger_set:
        embedding = model.extract_embedding(trigger)
        score = cosine_similarity(embedding, signature)
        scores.append(score)

    mean_score = average(scores)

    if mean_score >= threshold:
        return True
    else:
        return False
```

## Explainability Integration Module

Directly stated:

- SHAP is used for text token attribution.
- Grad-CAM is used for image attribution.
- Score-CAM is also mentioned for image attribution.
- Attention rollout is used for transformer focus.
- Explanations are supportive evidence only.
- Explanations do not establish causality.
- Explanations are not legal proof.

Implementation steps:

```text
For text:
    Run SHAP over tokens.
    Compare high-attribution tokens with trigger tokens.

For image:
    Run Grad-CAM or Score-CAM.
    Compare heatmap with trigger region mask.

For multimodal:
    Run attention rollout or CAM+attention.
    Compare attribution with trigger-related tokens or image regions.
```

---

# 6. Architecture Specification

## DistilBERT

Directly stated:

| Item | Value |
|---|---|
| Model family | Transformer language model |
| Parameter count | Approximately 66 million |
| Initialization | Pretrained checkpoint |
| Datasets | AG News, SQuAD v2.0 |
| Explanation | Token-level attribution using SHAP |
| Watermark embedding | During final fine-tuning |

Not specified in the paper:

- Exact checkpoint name.
- Maximum sequence length.
- Tokenizer details.
- Classifier head structure.
- Dropout.
- Optimizer.
- Learning rate.
- Batch size.

Recommended assumptions:

| Item | Recommended value |
|---|---|
| Checkpoint | `distilbert-base-uncased` |
| AG News head | Sequence classification head with 4 labels |
| SQuAD head | Question-answering head, optional |
| Embedding | CLS-like token from final hidden state |
| Tokenizer | HuggingFace AutoTokenizer |
| Max sequence length | 128 or 256 |
| Padding | Dynamic padding or max-length padding |
| Truncation | Enabled |

## MobileNetV2

Directly stated:

| Item | Value |
|---|---|
| Model family | Lightweight CNN |
| Parameter count | Approximately 3.5 million |
| Initialization | ImageNet backbone |
| Dataset | CIFAR-10 |
| Explanation | Grad-CAM and Score-CAM |
| Deployment | ONNX export for edge latency |

Not specified in the paper:

- Image resolution.
- Normalization.
- Augmentation.
- Target CAM layer.
- Fine-tuning strategy.

Recommended assumptions:

| Item | Recommended value |
|---|---|
| Model | torchvision MobileNetV2 |
| Weights | ImageNet pretrained |
| Input size | 224 x 224 |
| Normalization | ImageNet mean and standard deviation |
| Classifier | Replace final classifier with 10-class head |
| Embedding | Penultimate global average pooled vector |
| CAM layer | Last convolutional block |

## CLIP

Directly stated:

- Used for multimodal evaluation on Flickr30K.
- Uses joint vision-language embeddings.
- Used with unusual image-text pairs such as `green tiger playing guitar`.

Important inconsistency:

| Location in paper | CLIP description |
|---|---|
| Model architecture section | CLIP with ResNet-50 vision encoder and 12-layer transformer text encoder |
| Table 3 | CLIP ViT-B |

Recommended implementation decision:

- Default to CLIP ViT-B because Table 3 explicitly reports `CLIP (ViT-B)`.
- Also allow CLIP RN50 as an optional config.

Recommended assumptions:

| Item | Recommended value |
|---|---|
| Default checkpoint | `openai/clip-vit-base-patch32` |
| Optional checkpoint | CLIP RN50 |
| Dataset | Flickr30K |
| Embedding | Projected image/text embedding |
| Normalization | L2 normalization before cosine similarity |
| Objective | Contrastive image-text matching or embedding verification |

## ViLT

Directly stated:

| Item | Value |
|---|---|
| Model family | Vision-language transformer |
| Visual backbone | No separate visual backbone |
| Transformer | 12-layer shared transformer |
| Parameter count | Approximately 86 million |
| Dataset | Flickr30K |
| Explanation | Attention-based visualization |
| Deployment | ONNX export |

Not specified in the paper:

- Exact checkpoint.
- Fine-tuning objective.
- Image preprocessing.
- Text sequence length.
- Pooling strategy.

Recommended assumptions:

| Item | Recommended value |
|---|---|
| Checkpoint | HuggingFace ViLT base checkpoint |
| Embedding | Final CLS token or pooled joint representation |
| Explanation | Attention rollout |
| Dataset use | Image-text matching or retrieval |

---

# 7. Mathematical Objectives

## 7.1 Important equation: cosine similarity

Use this as the main math block in the Markdown file.

```math
cos(e, S_m) = (e dot S_m) / (norm(e) * norm(S_m))
```

Implementation formula:

```text
cosine_similarity = dot(embedding, signature) / (norm(embedding) * norm(signature))
```

Recommended implementation:

```python
score = torch.nn.functional.cosine_similarity(
    embedding,
    signature,
    dim=-1
)
```

## 7.2 Watermark signature

Directly stated:

The watermark signature is the centroid of trigger-induced latent activations.

Implementation formula:

```text
signature = mean(trigger_embeddings)
signature = normalize(signature)
```

Recommended implementation:

```python
trigger_embeddings = normalize(trigger_embeddings)
signature = trigger_embeddings.mean(dim=0)
signature = normalize(signature)
```

Status:

- Normalizing embeddings is inferred, not explicitly stated.
- Normalization is strongly recommended because cosine similarity is used.

## 7.3 Mean verification score

Implementation formula:

```text
scores = cosine_similarity(trigger_embedding_i, signature)
mean_score = average(scores)
```

Decision rule:

```text
if mean_score >= tau_m:
    watermark_present = True
else:
    watermark_present = False
```

## 7.4 Threshold selection

Directly stated:

- Select threshold `tau_m` using validation data.
- Choose the threshold that maximizes F1-score over trigger and benign samples.

Implementation formula:

```text
tau_m = threshold that maximizes F1(validation_labels, validation_predictions)
```

Recommended implementation:

```text
1. Compute scores for validation trigger samples.
2. Compute scores for validation benign samples.
3. Label trigger samples as 1.
4. Label benign samples as 0.
5. Sweep candidate thresholds.
6. For each threshold, compute F1-score.
7. Select threshold with maximum F1-score.
```

Recommended threshold search:

```text
candidate_thresholds = all unique validation scores
```

Alternative:

```text
candidate_thresholds = range from -1.0 to 1.0 with step 0.001
```

## 7.5 Alignment loss

The exact formula is not specified in the paper. The following is a recommended implementation.

Implementation formula:

```text
alignment_loss = average(1 - cosine_similarity(trigger_embedding, signature))
```

Purpose:

- Minimize alignment loss.
- Push trigger embeddings toward the signature.

Status:

- Inferred, not explicitly stated.

## 7.6 Separation loss

The exact formula is not specified in the paper. The following is a recommended implementation.

Implementation formula:

```text
separation_loss = average(max(0, cosine_similarity(benign_embedding, signature) - gamma))
```

Purpose:

- Minimize separation loss.
- Penalize benign embeddings that become too similar to the signature.

Status:

- Inferred, not explicitly stated.

## 7.7 Total loss

Implementation formula:

```text
total_loss = task_loss
           + lambda_align * alignment_loss
           + lambda_sep * separation_loss
```

Not specified in the paper:

- `lambda_align`
- `lambda_sep`
- `gamma`
- Whether signature is fixed before training or recomputed during training.
- Whether trigger loss is applied every batch or only trigger batches.

Recommended defaults:

| Parameter | Suggested default | Status |
|---|---:|---|
| `lambda_align` | 1.0 | Assumption |
| `lambda_sep` | 0.1 | Assumption |
| `gamma` | 0.2 | Assumption |
| embedding normalization | true | Assumption |

---

# 8. Data Specification

## Dataset summary

| Dataset | Modality | Use in paper |
|---|---|---|
| AG News | Text | Four-class news classification for text watermarking |
| SQuAD v2.0 | Text QA | Trigger-based inference verification |
| CIFAR-10 | Image | Image classification and Grad-CAM watermark explainability |
| Flickr30K | Multimodal | Cross-modal watermark evaluation |
| Trigger Set | Hybrid | Derived from The Pile, COCO long-tail categories, and Visual Genome |

## AG News

Directly stated:

- Four-class news classification.
- Used with DistilBERT.
- Used for LLM/text watermarking.

Not specified in the paper:

- Exact split.
- Validation split.
- Tokenization.
- Maximum sequence length.
- Trigger insertion frequency.

Recommended assumptions:

| Item | Recommended choice |
|---|---|
| Split | Standard AG News train/test split |
| Validation | 10% stratified split from training data |
| Tokenizer | DistilBERT tokenizer |
| Max length | 128 |
| Trigger insertion | 5% to 10% of training samples or batches |

## SQuAD v2.0

Directly stated:

- Text QA dataset.
- Used for trigger-based inference verification.

Not specified in the paper:

- Whether SQuAD is trained or only queried.
- Whether exact-match or QA F1 is evaluated.
- Whether SQuAD appears in final results.

Recommended implementation:

- Treat SQuAD as optional.
- Implement only after AG News reproduction works.
- Use it for embedding-based verification rather than as a required main result.

## CIFAR-10

Directly stated:

- Standard image classification task.
- Used with MobileNetV2.
- Used for Grad-CAM watermark explainability.

Not specified in the paper:

- Image size.
- Normalization.
- Augmentation.
- Validation split.
- Trigger patch location or mask.

Recommended assumptions:

| Item | Recommended choice |
|---|---|
| Input size | 224 x 224 |
| Normalization | ImageNet mean/std |
| Training augmentation | Random crop, horizontal flip |
| Evaluation preprocessing | Deterministic resize and normalize |
| Validation split | 10% from training data |
| Trigger region | Fixed small patch or low-frequency perturbation |

## Flickr30K

Directly stated:

- Image-caption dataset.
- Used for cross-modal watermarks.
- Used with CLIP and ViLT.

Not specified in the paper:

- Split.
- Objective.
- Negative sampling.
- Caption handling.
- Whether all captions per image are used.

Recommended assumptions:

| Item | Recommended choice |
|---|---|
| Split | Karpathy split if available |
| Objective | Image-text retrieval or contrastive matching |
| Captions | Use all available captions or first caption consistently |
| Trigger type | Rare or semantically unusual image-caption pairs |

## Trigger Set

Directly stated:

- Not a hidden/private dataset.
- Publicly derived.
- Text triggers from The Pile.
- Visual triggers from COCO long-tail categories.
- Multimodal triggers from Visual Genome.
- Inserted into AG News, CIFAR-10, and Flickr30K during fine-tuning.
- Detected during inference.

Not specified in the paper:

- Exact trigger list.
- Exact number of main triggers.
- Exact low-frequency threshold.
- Exact visual trigger generation method.
- Exact trigger insertion ratio.
- Exact split of triggers into train/validation/test.

Recommended assumptions:

| Item | Recommended choice |
|---|---|
| Main number of triggers | 30 |
| Trigger-size ablation | 5, 10, 20, 30 |
| Text trigger selection | Low-frequency phrase list saved to JSON |
| Visual trigger mask | Save binary masks for alignment scoring |
| Multimodal trigger list | Save image-caption pair IDs |
| Trigger split | Separate train, validation, and test trigger files |

---

# 9. Training Protocol

## Directly stated training environment

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 3090 |
| Framework | PyTorch 2.0 |
| NLP library | HuggingFace Transformers |
| Vision library | torchvision |
| Explainability | Captum and TorchCAM |
| Deployment | ONNX export and TensorRT simulation |
| Edge devices | Jetson Nano and Raspberry Pi 4 |
| Quantization | int8 |

## Directly stated training behavior

- Watermark embedding is applied during final fine-tuning.
- Trigger inputs are added during fine-tuning.
- Trigger embeddings are aligned with signature vector.
- Primary task performance should be preserved.
- Robustness is evaluated after attacks.

## Not specified in the paper

- Optimizer.
- Learning rate.
- Weight decay.
- Batch size.
- Number of epochs.
- Scheduler.
- Warm-up.
- Gradient clipping.
- Mixed precision.
- Early stopping.
- Exact seeds.
- Number of runs.
- Checkpoint selection rule.
- Trigger ratio.
- Distillation temperature.
- Student model architecture.
- Fine-tuning attack dataset.
- Pruning implementation.

## Recommended default training settings

These are assumptions for reproduction, not paper-stated facts.

| Component | Suggested default |
|---|---|
| Optimizer | AdamW |
| Transformer learning rate | 2e-5 |
| CLIP/ViLT learning rate | 1e-5 or 5e-6 |
| MobileNetV2 classifier LR | 1e-3 |
| MobileNetV2 full fine-tuning LR | 1e-4 |
| Weight decay | 0.01 |
| Text batch size | 16 or 32 |
| Image batch size | 64 or 128 |
| Multimodal batch size | 16 or 32 |
| Transformer epochs | 3 to 5 |
| CIFAR-10 epochs | 30 to 100 |
| Scheduler | Linear warm-up for transformers |
| Mixed precision | Enabled on RTX 3090 |
| Seeds | 0, 1, 2, 3, 4 |
| Trigger ratio | 5% to 10% |
| Checkpoint metric | Validation watermark F1 plus task accuracy |

## Recommended checkpoint rule

Use the checkpoint with:

```text
highest validation watermark F1
subject to acceptable task accuracy degradation
```

Because the paper claims preservation of task utility but does not provide primary task accuracy tables, Codex should log both task and watermark metrics.

---

# 10. Inference Protocol

## Directly stated

- Verification occurs during inference.
- The deployed model is queried with triggers.
- The model returns embeddings or pooled feature vectors.
- Cosine similarity is computed against stored signature.
- Scores are averaged.
- Mean score is compared with threshold.
- Verification is compatible with ONNX and TensorRT-style deployment.

## Inference algorithm

```text
Input:
    deployed_model
    trigger_set
    signature
    tau_m

Procedure:
    scores = []

    for trigger in trigger_set:
        embedding = deployed_model.extract_embedding(trigger)
        embedding = normalize(embedding)
        score = cosine_similarity(embedding, signature)
        scores.append(score)

    mean_similarity = average(scores)

    if mean_similarity >= tau_m:
        decision = True
    else:
        decision = False

Output:
    decision
    mean_similarity
    scores
```

## Recommended output object

```json
{
  "decision": true,
  "mean_similarity": 0.91,
  "threshold": 0.84,
  "num_triggers": 30,
  "per_trigger_scores": [0.88, 0.92, 0.90],
  "modality": "text",
  "model_name": "distilbert",
  "signature_id": "signature_text_agnews_seed0"
}
```

## Not specified in the paper

- Inference batch size.
- Whether embeddings are normalized.
- Whether attribution is generated for every trigger.
- Whether benign inputs are evaluated during normal verification.
- Whether verification uses one threshold per model or one threshold per modality.

## Recommended assumptions

- Batch trigger queries when possible.
- Normalize embeddings before similarity.
- Use one threshold per model/modality pair.
- Run attribution only for positive or near-threshold cases to reduce latency.
- Save all per-trigger scores for auditability.

---

# 11. Evaluation Protocol

## Primary metrics

### Watermark detection accuracy / success rate

Implementation formula:

```text
watermark_success_rate = correctly_detected_watermark_triggers / total_watermark_triggers_tested
```

The paper states that high success rate above 95% indicates reliable detection under clean conditions.

### Robustness to model tampering

Implementation formula:

```text
robustness_drop = clean_detection_accuracy - attacked_detection_accuracy
retained_robustness = attacked_detection_accuracy
```

The paper uses both degradation and retained robustness language. Table 5 reports robustness as retained performance.

### Attribution alignment score

Implementation concept:

```text
alignment_score = overlap between attribution-highlighted area/tokens and known trigger region/tokens
```

For text:

```text
Compare SHAP token importance with trigger token span.
```

For images:

```text
Compare Grad-CAM or Score-CAM heatmap with trigger mask.
```

Not specified:

- Exact overlap formula.
- Whether top-k tokens, IoU, thresholded heatmap overlap, or another metric is used.

Recommended alignment metrics:

| Modality | Recommended metric |
|---|---|
| Text | Top-k token overlap with trigger tokens |
| Image | Intersection-over-union between thresholded heatmap and trigger mask |
| Multimodal | Combined token overlap and region overlap |

### Latency overhead

Directly stated components:

- Base model prediction time.
- Signature matching time.
- Attribution generation time.

Implementation formula:

```text
total_latency = model_inference_time
              + embedding_extraction_time
              + signature_matching_time
              + attribution_generation_time
```

Target:

```text
total_latency < 200 ms
```

## Secondary metrics

- False positive rate.
- Standard deviation across runs.
- Threshold sensitivity.
- Trigger-set-size sensitivity.
- Baseline capability scores.
- Radar-normalized capability scores.

## Baselines

| Baseline | Description |
|---|---|
| B1 | Naive watermarking |
| B2 | Black-box embedding verification |
| B3 | Explainable watermark verification |
| B4 | Full X-ModalProof |

## Attack settings

Directly stated:

| Attack | Description |
|---|---|
| Pruning | Structured pruning, up to 50%; Table 3 uses 30% |
| Fine-tuning | Fine-tuning on unrelated downstream tasks |
| Distillation | Smaller student model replicates original behavior |
| Black-box | API-level access without model parameters |

## Statistical reporting

Directly stated:

- Table 8 reports mean ± standard deviation.

Not specified:

- Number of seeds.
- Confidence intervals.
- Statistical significance tests.

Recommended:

- Use 5 seeds: 0, 1, 2, 3, 4.
- Report mean, standard deviation, and optionally 95% confidence intervals.

---

# 12. Results Inventory

## Table 3: Watermark Detection Accuracy

| Model | Dataset | Clean | After Fine-Tuning | After Pruning 30% | After Distillation |
|---|---|---:|---:|---:|---:|
| DistilBERT | AG News | 97.2 | 93.4 | 88.9 | 85.6 |
| MobileNetV2 | CIFAR-10 | 96.1 | 92.3 | 89.1 | 86.4 |
| CLIP ViT-B | Flickr30K | 95.6 | 91.2 | 88.5 | 85.0 |
| ViLT | Flickr30K | 94.8 | 89.5 | 87.0 | 84.1 |

Target outputs:

- `table3_detection_accuracy.csv`
- `figure2_detection_accuracy.png`

## Table 4: Attribution Alignment Score

| Model | Tool | Alignment Score |
|---|---|---:|
| DistilBERT | SHAP | 87.6 |
| MobileNetV2 | Grad-CAM | 91.2 |
| CLIP | AttnRollout | 89.5 |
| ViLT | CAM+Attn | 90.1 |

Target outputs:

- `table4_attribution_alignment.csv`
- `figure4_attribution_alignment.png`
- SHAP visualization
- CAM overlay visualization

## Table 5: Internal Baseline Comparison

| Capability | B1 Naive WM | B2 Black-Box | B3 Explainable | B4 X-ModalProof |
|---|---:|---:|---:|---:|
| Accuracy | 92 | 95 | 95 | 97 |
| Robustness | 65 | 75 | 75 | 90 |
| Attribution Alignment | N/A | N/A | 80 | 92 |
| Latency ms | 100 | 130 | 160 | 190 |
| Edge Deployment | Partial | Partial | Full | Full |
| Multimodal Support | No | No | Partial | Yes |

Target outputs:

- `table5_internal_baselines.csv`
- `figure8_radar_chart.png`

## Table 6: Threshold Sensitivity

| Threshold Offset | Detection Accuracy | False Positive Rate | Robustness |
|---:|---:|---:|---:|
| -0.05 | 96.8 | 6.5 | 88.7 |
| -0.02 | 97.1 | 4.2 | 89.5 |
| 0.00 selected tau_m | 97.0 | 3.8 | 90.0 |
| +0.02 | 96.6 | 2.9 | 89.2 |
| +0.05 | 95.9 | 2.1 | 87.8 |

Target outputs:

- `table6_threshold_sensitivity.csv`
- Optional line plot.

## Table 7: Trigger-Set Size Ablation

| Number of Triggers | Detection Accuracy | Standard Deviation | Verification Latency ms |
|---:|---:|---:|---:|
| 5 | 92.4 | 2.1 | 150 |
| 10 | 94.8 | 1.5 | 165 |
| 20 | 96.7 | 1.0 | 180 |
| 30 | 97.0 | 0.8 | 190 |

Target outputs:

- `table7_trigger_size.csv`
- Optional accuracy-latency tradeoff plot.

## Table 8: Repeatability Across Seeds

| Configuration | Detection Accuracy | Robustness | Attribution Alignment |
|---|---:|---:|---:|
| B1 | 92.1 +/- 1.3 | 65.4 +/- 2.5 | N/A |
| B2 | 94.3 +/- 1.1 | 75.2 +/- 1.8 | N/A |
| B3 | 95.8 +/- 0.9 | 82.6 +/- 1.5 | 80.3 +/- 1.4 |
| B4 | 97.0 +/- 0.8 | 90.1 +/- 1.2 | 92.2 +/- 1.0 |

Target outputs:

- `table8_repeatability.csv`

Important inconsistency:

- Table 5 reports B3 robustness as 75.
- Table 8 reports B3 robustness as 82.6 +/- 1.5.
- Codex should preserve both values but flag the inconsistency.

---

# 13. Ablation Study Breakdown

## Ablation 1: B1 Naive WM

Changed component:

- Basic trigger-based watermarking only.

Included:

- Fixed triggers inserted during fine-tuning.

Excluded:

- Embedding-space verification.
- Explainability.
- Full robustness mechanisms.
- Multimodal support.

Reported values:

| Metric | Value |
|---|---:|
| Accuracy | 92 |
| Robustness | 65 |
| Latency ms | 100 |

Implementation requirement:

```text
Train model with trigger samples.
Evaluate whether triggers are detected.
Do not use SHAP, Grad-CAM, or attention explanations.
Do not use full multimodal verification.
```

## Ablation 2: B2 Black-Box

Changed component:

- Adds embedding-space cosine verification.

Included:

- Embedding extraction.
- Signature matching.
- Threshold decision.

Excluded:

- Explainability.

Reported values:

| Metric | Value |
|---|---:|
| Accuracy | 95 |
| Robustness | 75 |
| Latency ms | 130 |

Implementation requirement:

```text
Compute trigger embeddings.
Compare embeddings to stored signature.
Use tau_m threshold.
Return binary decision.
```

## Ablation 3: B3 Explainable

Changed component:

- Adds explainability.

Included:

- SHAP for text.
- Grad-CAM for images.
- Attribution alignment score.

Reported values:

| Metric | Value |
|---|---:|
| Accuracy | 95 |
| Robustness Table 5 | 75 |
| Robustness Table 8 | 82.6 +/- 1.5 |
| Attribution Alignment | 80 |
| Latency ms | 160 |

Implementation requirement:

```text
Run B2 verification.
Generate attribution maps.
Compute alignment score against trigger tokens or masks.
```

## Ablation 4: B4 Full X-ModalProof

Changed component:

- Full framework.

Included:

- Watermark embedding.
- Embedding-space verification.
- Explainability.
- Robustness testing.
- Edge deployment.
- Multimodal support.

Reported values:

| Metric | Value |
|---|---:|
| Accuracy | 97 |
| Robustness | 90 |
| Attribution Alignment | 92 |
| Latency ms | 190 |

Implementation requirement:

```text
Run full pipeline with triggers, signature matching, attribution, robustness evaluation, and ONNX/int8 latency benchmark.
```

## Ablation 5: Threshold sensitivity

Changed variable:

- Threshold offset around selected `tau_m`.

Trigger values:

```text
-0.05, -0.02, 0.00, +0.02, +0.05
```

Codex must produce:

- Accuracy.
- False positive rate.
- Robustness.

## Ablation 6: Trigger-set size

Changed variable:

- Number of trigger queries.

Trigger counts:

```text
5, 10, 20, 30
```

Codex must produce:

- Detection accuracy.
- Standard deviation.
- Verification latency.

---

# 14. Baselines and Comparators

## Internal baselines to implement

| Baseline | Type | Must implement? | Feasibility |
|---|---|---|---|
| B1 Naive WM | Internal watermark baseline | Yes | High |
| B2 Black-Box | Embedding verification baseline | Yes | High |
| B3 Explainable | Attribution-enabled baseline | Yes | Medium |
| B4 X-ModalProof | Full framework | Yes | Medium-high |

## External methods discussed

The paper includes qualitative comparison to prior work but avoids direct numerical comparison because external methods differ in assumptions, architectures, and datasets.

External comparators mentioned include:

- LLM watermarking methods.
- TinyML watermarking methods.
- Active intellectual property protection methods.
- Federated ownership verification.
- Explanation as a Watermark.
- WMarkGPT.
- Watermarking vision-language models.
- Graph neural network watermarking.
- PTYNet / Free Fine-tuning.

## Feasibility

| Comparator type | Reproduction recommendation |
|---|---|
| Survey papers | Citation-level only |
| LLM watermarking | Do not reimplement unless specifically required |
| EaaW | Separate repository likely needed |
| WMarkGPT | Separate multimodal LLM pipeline likely needed |
| AGATE-like black-box multimodal watermarking | Separate implementation likely needed |
| TinyML watermarking | Optional qualitative comparison |
| Internal B1-B4 | Primary reproduction target |

---

# 15. Figures, Tables, and Artifacts to Reproduce

## Required tables

| Table | Artifact file |
|---|---|
| Table 1 Functional capabilities | `table1_capabilities.md` |
| Table 2 Dataset summary | `table2_datasets.md` |
| Table 3 Detection accuracy | `table3_detection_accuracy.csv` |
| Table 4 Attribution alignment | `table4_attribution_alignment.csv` |
| Table 5 Internal baselines | `table5_internal_baselines.csv` |
| Table 6 Threshold sensitivity | `table6_threshold_sensitivity.csv` |
| Table 7 Trigger-set size | `table7_trigger_size.csv` |
| Table 8 Repeatability | `table8_repeatability.csv` |

## Required figures

| Figure | Output file | Required computation |
|---|---|---|
| Figure 1 | `figure1_architecture.png` | Draw pipeline diagram |
| Figure 2 | `figure2_accuracy.png` | Plot B1-B4 accuracy |
| Figure 3 | `figure3_robustness_drop.png` | Plot robustness drop |
| Figure 4 | `figure4_alignment.png` | Plot attribution alignment |
| Figure 5 | `figure5_shap_tokens.png` | SHAP token attribution |
| Figure 6 | `figure6_gradcam_overlay.png` | Grad-CAM image overlay |
| Figure 7 | `figure7_latency.png` | Plot B1-B4 latency |
| Figure 8 | `figure8_radar.png` | Normalized radar chart |

## Algorithms to encode

| Algorithm | Required implementation |
|---|---|
| Algorithm 1 | Watermark embedding loop |
| Algorithm 2 | Watermark verification |
| Algorithm 3 | Attribution-based analysis |
| Algorithm 4 | Combined embedding and verification procedure |

---

# 16. Hidden Implementation Dependencies

All items below are **inferred, not explicitly stated**.

## Embedding and similarity

- L2-normalize embeddings before cosine similarity.
- L2-normalize signature vector.
- Store signature with model name, modality, dataset, trigger count, seed, and date.

## Text models

- Use HuggingFace tokenizer.
- Use padding and truncation.
- Use CLS-like representation for DistilBERT.
- Use attention mask correctly during embedding extraction.

## Image models

- Resize CIFAR-10 images to 224 x 224 if using ImageNet-pretrained MobileNetV2.
- Use ImageNet normalization.
- Use final convolutional layer for CAM methods.
- Save trigger masks for attribution alignment.

## Multimodal models

- Use CLIP processor or ViLT processor.
- Keep image-caption alignment intact.
- Normalize CLIP embeddings.
- Use final CLS token for ViLT.

## Training

- Use AdamW for transformer fine-tuning.
- Use deterministic seeds.
- Use mixed precision on RTX 3090.
- Save best checkpoint by validation watermark F1.

## Attacks

- Use structured pruning for convolutional and transformer layers.
- Use same trigger set before and after attacks.
- Use same threshold after attacks unless testing recalibration.
- Use student architectures smaller than teacher for distillation.

## Deployment

- Use ONNX Runtime when TensorRT is unavailable.
- Use warm-up runs before timing.
- Report mean, standard deviation, p50, and p95 latency.
- Use int8 quantization calibration data.

---

# 17. Ambiguities, Gaps, and Risks

## 1. Exact trigger construction is missing

Issue:

- The paper describes trigger sources but not exact triggers.

Why it matters:

- Trigger choice strongly affects watermark accuracy, robustness, and false positives.

Severity:

- Critical.

Recommended workaround:

- Create reproducible trigger files and version them.
- Save trigger source, filtering criteria, and random seed.

## 2. Exact training hyperparameters are missing

Issue:

- Optimizer, learning rate, epochs, batch size, and scheduler are not specified.

Why it matters:

- Detection performance and robustness depend on training setup.

Severity:

- Critical.

Recommended workaround:

- Use standard defaults and run sensitivity analysis.

## 3. Loss weights are missing

Issue:

- `lambda_align`, `lambda_sep`, and `gamma` are not specified.

Why it matters:

- These determine the tradeoff between watermark strength and task utility.

Severity:

- Critical.

Recommended workaround:

- Expose them in config.
- Sweep multiple values.

## 4. Equation rendering is incomplete

Issue:

- Several formulas appear incomplete or missing in the manuscript text.

Why it matters:

- Exact objective cannot be reconstructed with full certainty.

Severity:

- High.

Recommended workaround:

- Implement the practical objective described in this handoff and document it clearly.

## 5. CLIP architecture inconsistency

Issue:

- One section says CLIP uses ResNet-50.
- Table 3 says CLIP ViT-B.

Why it matters:

- Different CLIP backbones have different embeddings, accuracy, and latency.

Severity:

- High.

Recommended workaround:

- Default to CLIP ViT-B.
- Add optional CLIP RN50 config.

## 6. SQuAD v2.0 is mentioned but not reported

Issue:

- SQuAD is listed as a dataset but not included in main results tables.

Why it matters:

- Codex cannot know what exact result to reproduce.

Severity:

- Medium.

Recommended workaround:

- Treat SQuAD as optional.
- Do not block main reproduction on SQuAD.

## 7. Attribution alignment metric is unspecified

Issue:

- The paper says overlap-based scoring but does not define formula.

Why it matters:

- Different overlap metrics produce different scores.

Severity:

- High.

Recommended workaround:

- Implement top-k token overlap for text.
- Implement heatmap-mask IoU for images.
- Report the chosen formula clearly.

## 8. Distillation setup is missing

Issue:

- Student architectures, loss, temperature, and data are not specified.

Why it matters:

- Post-distillation watermark accuracy depends heavily on setup.

Severity:

- High.

Recommended workaround:

- Define smaller student models per modality.
- Use standard KL-divergence distillation.

## 9. Latency protocol is underspecified

Issue:

- Exact timing setup is not described.

Why it matters:

- Edge latency can vary widely.

Severity:

- High.

Recommended workaround:

- Use warm-up runs.
- Report mean, std, p50, p95.
- Record hardware, runtime, batch size, and quantization.

## 10. Primary task utility is not reported

Issue:

- The paper claims task utility is preserved but does not provide task accuracy tables.

Why it matters:

- A watermark method should not destroy base model performance.

Severity:

- High.

Recommended workaround:

- Always report task accuracy or task-specific metric before and after watermarking.

---

# 18. Reproduction Strategy

## Exact reproduction path

Exact reproduction is not possible from the paper alone because several critical details are missing.

## Minimum viable reproduction path

1. Implement DistilBERT on AG News.
2. Create a small fixed text trigger set.
3. Fine-tune DistilBERT with task loss plus watermark losses.
4. Compute signature vector.
5. Select threshold on validation data.
6. Evaluate clean watermark detection.
7. Evaluate false positive rate on benign samples.
8. Add pruning/fine-tuning attacks.
9. Generate SHAP token attribution.
10. Produce Table 3-style and Table 4-style outputs.

## Staged implementation plan

### Stage 1: Core verification library

Implement:

- Embedding extraction.
- Signature computation.
- Cosine scoring.
- Threshold selection.
- Verification decision.

### Stage 2: Text reproduction

Implement:

- AG News loader.
- DistilBERT wrapper.
- Text triggers.
- Text watermark training.
- SHAP explanations.

### Stage 3: Image reproduction

Implement:

- CIFAR-10 loader.
- MobileNetV2 wrapper.
- Visual triggers.
- Grad-CAM / Score-CAM.
- Image attribution alignment.

### Stage 4: Multimodal reproduction

Implement:

- Flickr30K loader.
- CLIP wrapper.
- ViLT wrapper.
- Multimodal triggers.
- Attention rollout.

### Stage 5: Robustness

Implement:

- Structured pruning.
- Fine-tuning attack.
- Distillation attack.

### Stage 6: Deployment

Implement:

- ONNX export.
- int8 quantization.
- ONNX Runtime inference.
- Latency benchmarking.

### Stage 7: Reporting

Implement:

- CSV outputs.
- Markdown tables.
- Figures.
- Radar chart.
- Repeatability runner.

## What can be mocked initially

- TensorRT deployment.
- Physical Jetson Nano timing.
- Physical Raspberry Pi timing.
- Full mining from The Pile, COCO, and Visual Genome.
- SQuAD v2.0 QA reproduction.

## What requires sensitivity testing

- Trigger count.
- Trigger ratio.
- Trigger strength.
- Threshold value.
- `lambda_align`.
- `lambda_sep`.
- `gamma`.
- Pruning rate.
- Distillation temperature.
- Student size.
- Embedding layer choice.

---

# 19. Codebase Blueprint for Codex

Recommended repository structure:

```text
x-modalproof/
  README.md
  CODEX_HANDOFF.md
  requirements.txt
  pyproject.toml
  .gitignore

  configs/
    text_distilbert_agnews.yaml
    image_mobilenetv2_cifar10.yaml
    multimodal_clip_flickr30k.yaml
    multimodal_vilt_flickr30k.yaml

    ablations/
      threshold_sensitivity.yaml
      trigger_size.yaml
      pruning_sweep.yaml

    triggers/
      text_triggers.json
      image_triggers.json
      multimodal_triggers.json

  src/
    data/
      agnews.py
      squad.py
      cifar10.py
      flickr30k.py
      trigger_sources.py
      splits.py

    triggers/
      text_triggers.py
      image_triggers.py
      multimodal_triggers.py
      trigger_registry.py

    models/
      distilbert_model.py
      mobilenetv2_model.py
      clip_model.py
      vilt_model.py
      embedding_extractors.py

    watermark/
      signature.py
      losses.py
      verification.py
      thresholding.py

    training/
      train_text.py
      train_image.py
      train_multimodal.py
      checkpointing.py
      schedulers.py

    attacks/
      pruning.py
      finetune_attack.py
      distillation.py
      quantization.py

    explainability/
      shap_text.py
      gradcam_image.py
      scorecam_image.py
      attention_rollout.py
      alignment.py

    deployment/
      export_onnx.py
      onnx_inference.py
      tensorrt_runner.py
      latency.py

    evaluation/
      metrics.py
      robustness.py
      repeatability.py
      baseline_runner.py

    visualization/
      tables.py
      plots.py
      radar.py
      cam_overlay.py
      shap_plot.py

    utils/
      config.py
      logging.py
      seed.py
      device.py
      io.py

  scripts/
    run_text_agnews.sh
    run_image_cifar10.sh
    run_multimodal_clip.sh
    run_multimodal_vilt.sh
    run_all_baselines.sh
    run_attacks.sh
    run_threshold_sensitivity.sh
    run_trigger_size.sh
    export_all_onnx.sh
    benchmark_latency.sh
    make_all_artifacts.sh

  tests/
    test_signature.py
    test_thresholding.py
    test_losses.py
    test_embedding_shapes.py
    test_trigger_splits.py
    test_verification.py

  outputs/
    checkpoints/
    signatures/
    metrics/
    figures/
    tables/
    logs/
```

---

# 20. Config Schema

Recommended YAML schema:

```yaml
experiment:
  name: distilbert_agnews_b4
  modality: text
  baseline: B4
  seed: 0
  output_dir: outputs/

dataset:
  name: ag_news
  root: data/
  validation_ratio: 0.1
  test_split: standard
  num_workers: 4

preprocessing:
  tokenizer_name: distilbert-base-uncased
  max_length: 128
  image_size: 224
  normalize: imagenet
  train_augmentations: true
  eval_augmentations: false

triggers:
  source: pile_low_frequency
  num_triggers: 30
  trigger_ratio: 0.1
  fixed_trigger_file: configs/triggers/text_triggers.json
  split_triggers: true
  visual_trigger_strength: null
  low_frequency_band: true

model:
  name: distilbert
  checkpoint: distilbert-base-uncased
  num_labels: 4
  embedding_layer: cls
  freeze_backbone: false

watermark:
  signature_dim: auto
  normalize_embeddings: true
  signature_method: centroid
  lambda_align: 1.0
  lambda_sep: 0.1
  margin_gamma: 0.2
  threshold_selection: maximize_f1
  threshold_search_step: 0.001

optimizer:
  name: adamw
  lr: 2.0e-5
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  name: linear_warmup
  warmup_ratio: 0.1

training:
  epochs: 3
  batch_size: 32
  mixed_precision: true
  gradient_clip_norm: 1.0
  early_stopping: false
  checkpoint_metric: validation_watermark_f1

attacks:
  pruning:
    enabled: true
    amount: 0.3
    structured: true
  finetuning:
    enabled: true
    epochs: 1
    unrelated_task: true
  distillation:
    enabled: true
    temperature: 2.0
    alpha: 0.5

explainability:
  enabled: true
  methods: [shap]
  num_samples: 100
  alignment_metric: topk_overlap

deployment:
  export_onnx: true
  quantization: int8
  runtime: onnxruntime
  device_target: jetson_nano
  latency_runs: 100
  warmup_runs: 10

evaluation:
  metrics:
    - detection_accuracy
    - false_positive_rate
    - robustness
    - attribution_alignment
    - latency
  threshold_offsets: [-0.05, -0.02, 0.0, 0.02, 0.05]
  trigger_sizes: [5, 10, 20, 30]

logging:
  use_tensorboard: true
  save_predictions: true
  save_scores: true

reproducibility:
  deterministic: true
  seeds: [0, 1, 2, 3, 4]
```

---

# 21. Function/Class Inventory

## Data classes

| Name | Purpose | Inputs | Outputs |
|---|---|---|---|
| `AGNewsDataModule` | Load AG News | config | train, val, test dataloaders |
| `SQuADDataModule` | Load SQuAD v2.0 | config | QA dataloaders |
| `CIFAR10DataModule` | Load CIFAR-10 | config | train, val, test dataloaders |
| `Flickr30KDataModule` | Load Flickr30K | config | image-text dataloaders |

## Trigger classes

| Name | Purpose | Inputs | Outputs |
|---|---|---|---|
| `TextTriggerGenerator` | Generate rare phrase triggers | corpus, frequency threshold, count | trigger list |
| `ImageTriggerGenerator` | Generate visual triggers | image, mask, strength | triggered image and mask |
| `MultimodalTriggerGenerator` | Generate unusual image-caption triggers | image-caption data | trigger pairs |
| `TriggerRegistry` | Save/load fixed trigger sets | JSON path | trigger objects |

## Model wrappers

| Name | Purpose |
|---|---|
| `DistilBERTWatermarkModel` | Text model with embedding extraction |
| `MobileNetV2WatermarkModel` | Image model with embedding extraction |
| `CLIPWatermarkModel` | CLIP wrapper with projected embeddings |
| `ViLTWatermarkModel` | ViLT wrapper with joint embedding extraction |

Each model wrapper should expose:

```python
forward_task(batch)
extract_embedding(batch)
forward_with_embedding(batch)
```

## Watermark functions

| Name | Purpose |
|---|---|
| `compute_signature` | Compute centroid signature from trigger embeddings |
| `cosine_scores` | Compute cosine similarity scores |
| `select_threshold` | Select tau by validation F1 |
| `verify_watermark` | Return decision and scores |
| `AlignmentLoss` | Align trigger embeddings with signature |
| `SeparationLoss` | Separate benign embeddings from signature |
| `WatermarkLoss` | Combine task, alignment, and separation losses |

## Attack functions

| Name | Purpose |
|---|---|
| `apply_structured_pruning` | Apply pruning to model |
| `run_finetune_attack` | Fine-tune model on unrelated task |
| `distill_model` | Train student model from teacher |
| `quantize_model_int8` | Prepare int8 deployment model |

## Explainability functions

| Name | Purpose |
|---|---|
| `compute_shap_text` | Generate SHAP token attributions |
| `compute_gradcam` | Generate Grad-CAM heatmaps |
| `compute_scorecam` | Generate Score-CAM heatmaps |
| `compute_attention_rollout` | Generate transformer attention rollout |
| `compute_alignment` | Score attribution-trigger overlap |

## Deployment functions

| Name | Purpose |
|---|---|
| `export_to_onnx` | Export model to ONNX |
| `run_onnx_embeddings` | Run ONNX model and return embeddings |
| `benchmark_latency` | Measure latency over repeated runs |

## Visualization functions

| Name | Purpose |
|---|---|
| `plot_detection_accuracy` | Create Figure 2 |
| `plot_robustness_drop` | Create Figure 3 |
| `plot_attribution_alignment` | Create Figure 4 |
| `plot_latency` | Create Figure 7 |
| `plot_radar_chart` | Create Figure 8 |
| `render_shap_tokens` | Create Figure 5 |
| `render_cam_overlay` | Create Figure 6 |

---

# 22. Experiment Matrix

| ID | Description | Config differences | Expected outputs | Runtime complexity |
|---|---|---|---|---|
| E01 | DistilBERT AG News B1 | naive watermark only | accuracy, B1 metrics | Medium |
| E02 | DistilBERT AG News B2 | add cosine verification | threshold, scores | Medium |
| E03 | DistilBERT AG News B3 | add SHAP | attribution outputs | High |
| E04 | DistilBERT AG News B4 | full text pipeline | all text metrics | High |
| E05 | MobileNetV2 CIFAR-10 B1-B4 | image triggers and CAM | image metrics | High |
| E06 | CLIP Flickr30K | multimodal triggers | cross-modal detection | High |
| E07 | ViLT Flickr30K | ViLT embeddings and attention | attention alignment | High |
| E08 | Pruning attack | pruning 30% | post-pruning detection | Medium |
| E09 | Fine-tuning attack | unrelated task fine-tuning | post-finetune detection | High |
| E10 | Distillation attack | student model | post-distillation detection | High |
| E11 | Threshold sensitivity | threshold offsets | Table 6 | Low |
| E12 | Trigger-size ablation | 5, 10, 20, 30 triggers | Table 7 | Low-medium |
| E13 | Latency benchmark | ONNX/int8 | Figure 7 | Medium |
| E14 | Repeatability | seeds 0 to 4 | Table 8 | Very high |
| E15 | Radar chart | normalized Table 5 metrics | Figure 8 | Low |

---

# 23. Verification Checklist

## Data loading checks

- Confirm dataset loads successfully.
- Confirm expected number of classes.
- Confirm train, validation, and test splits are separate.
- Confirm trigger samples are not accidentally included in benign validation.
- Confirm trigger files are deterministic and versioned.
- Confirm image-caption pairs remain aligned.

## Tensor shape checks

- DistilBERT embedding shape is `[batch_size, hidden_dim]`.
- MobileNetV2 embedding shape is `[batch_size, feature_dim]`.
- CLIP embedding shape is `[batch_size, projection_dim]`.
- ViLT embedding shape is `[batch_size, hidden_dim]`.
- Signature shape matches embedding shape.
- Cosine score shape is `[batch_size]` or scalar after aggregation.

## Loss behavior checks

- Alignment loss decreases on trigger batches.
- Trigger cosine similarity increases over training.
- Benign cosine similarity remains lower than trigger similarity.
- Task loss does not explode.
- Main task accuracy does not collapse.

## Small-batch sanity tests

- Overfit 32 normal samples.
- Overfit 8 trigger samples.
- Confirm trigger scores are higher than benign scores.
- Confirm threshold can separate toy trigger and benign validation data.
- Confirm verification returns `True` for watermarked model.
- Confirm verification returns `False` for clean unwatermarked model.

## Metric verification

- Manually compute cosine similarity for one example.
- Manually compute mean similarity.
- Manually compute F1 for threshold selection.
- Verify false positive rate on benign samples.
- Verify robustness drop calculation.
- Verify attribution overlap calculation.

## Reproducibility checks

- Same seed gives same trigger set.
- Same seed gives same split.
- Repeated runs preserve B1 to B4 ranking.
- Saved signatures can be loaded and reused.
- ONNX model returns embeddings close to PyTorch model.

## Table and figure checks

- Table 3 values can be regenerated from CSV.
- Table 4 values can be regenerated from attribution outputs.
- Table 5 values can be regenerated from baseline runs.
- Table 6 values can be regenerated from threshold sweep.
- Table 7 values can be regenerated from trigger-size runs.
- Table 8 values can be regenerated from seed runs.

---

# 24. Exact Deliverables for Codex

## Codex Handoff Spec

### Project goal

Build a reproducible codebase for X-ModalProof, a framework for real-time explainable ownership verification of text, image, and multimodal AI models. The implementation must support watermark embedding, latent signature construction, black-box-style inference-time verification, attribution-based explanation, robustness attacks, ONNX/int8 deployment, latency measurement, and generation of paper-style tables and figures.

### Assumptions

Critical assumptions needed because the paper omits details:

- Use normalized embeddings for cosine similarity.
- Use centroid of trigger embeddings as watermark signature.
- Use validation F1 to select threshold.
- Use 30 triggers as the main default.
- Use 5, 10, 20, and 30 triggers for trigger-size ablation.
- Use 30% pruning for the main pruning attack.
- Use 5 random seeds: 0, 1, 2, 3, 4.
- Use CLIP ViT-B as default, with CLIP RN50 optional.
- Use standard dataset splits unless otherwise configured.
- Use task accuracy as an additional logged metric even though not reported in the paper.

### Required modules

Critical:

- Dataset loaders.
- Trigger generators.
- Model wrappers.
- Embedding extractors.
- Watermark losses.
- Signature computation.
- Threshold selection.
- Verification engine.
- Metrics.
- Internal baseline runner.
- Attack simulations.
- Explainability tools.
- Artifact generation.

Important:

- ONNX export.
- int8 quantization.
- Latency benchmarking.
- Radar chart generation.
- Repeatability runner.

Optional:

- TensorRT-specific runner.
- Physical Jetson Nano runner.
- Physical Raspberry Pi runner.
- Full trigger mining from The Pile, COCO, and Visual Genome.
- SQuAD v2.0 full QA reproduction.

### Required scripts

| Script | Purpose |
|---|---|
| `train_watermarked.py` | Train or fine-tune watermarked model |
| `verify_watermark.py` | Run signature-based verification |
| `run_baselines.py` | Run B1 to B4 |
| `run_attacks.py` | Run pruning, fine-tuning, and distillation |
| `run_threshold_sensitivity.py` | Generate Table 6 |
| `run_trigger_size_ablation.py` | Generate Table 7 |
| `run_explainability.py` | Generate SHAP/CAM outputs |
| `export_onnx.py` | Export models to ONNX |
| `benchmark_latency.py` | Measure latency |
| `make_tables.py` | Generate CSV/Markdown tables |
| `make_figures.py` | Generate plots and visualizations |

### Expected outputs

```text
outputs/
  checkpoints/
    *.pt

  signatures/
    *.pt
    *.json

  metrics/
    table3_detection_accuracy.csv
    table4_attribution_alignment.csv
    table5_internal_baselines.csv
    table6_threshold_sensitivity.csv
    table7_trigger_size.csv
    table8_repeatability.csv

  figures/
    figure2_accuracy.png
    figure3_robustness_drop.png
    figure4_alignment.png
    figure5_shap_tokens.png
    figure6_gradcam_overlay.png
    figure7_latency.png
    figure8_radar.png

  logs/
    *.jsonl
    *.txt
```

### Step-by-step build order

1. Build config loader.
2. Build deterministic seed utility.
3. Build model wrapper interface.
4. Build embedding extraction.
5. Build cosine similarity scoring.
6. Build signature computation.
7. Build threshold selection.
8. Build verification engine.
9. Implement AG News and DistilBERT.
10. Implement text triggers.
11. Implement watermark losses.
12. Run first end-to-end text experiment.
13. Add B1 to B4 baseline modes.
14. Add CIFAR-10 and MobileNetV2.
15. Add visual triggers.
16. Add Grad-CAM and Score-CAM.
17. Add CLIP and Flickr30K.
18. Add ViLT.
19. Add pruning attack.
20. Add fine-tuning attack.
21. Add distillation attack.
22. Add ONNX export.
23. Add int8 quantization.
24. Add latency benchmark.
25. Add table and figure generation.
26. Add repeatability runner.

### Validation steps

Critical:

- Verify embeddings and signatures have matching dimensions.
- Verify trigger similarity is higher than benign similarity.
- Verify threshold maximizes validation F1.
- Verify clean watermarked model returns positive decision.
- Verify clean unwatermarked model has low false positive rate.
- Verify attack evaluation uses same trigger set and threshold.
- Verify attribution maps overlap with trigger masks or tokens.
- Verify ONNX embeddings are close to PyTorch embeddings.

Important:

- Verify task accuracy before and after watermarking.
- Verify repeatability across seeds.
- Verify B4 is better than B1 in detection, robustness, and attribution.
- Verify latency includes warm-up and repeated timing.

Optional:

- Verify TensorRT output matches ONNX Runtime.
- Verify Raspberry Pi timing separately from Jetson timing.

### Priority order

Critical:

- Core watermark verification.
- Trigger handling.
- Signature construction.
- Threshold selection.
- DistilBERT AG News reproduction.
- B1 to B4 baseline logic.

Important:

- Image pipeline.
- Multimodal pipeline.
- Explainability.
- Robustness attacks.
- ONNX export.
- Tables and figures.

Optional:

- Physical edge deployment.
- Full SQuAD.
- Full trigger mining.
- TensorRT-specific deployment.

---

# 25. Final Formatting Rules for This Repository

Use the following conventions:

- Use Markdown tables only for compact numerical or categorical values.
- Do not place complex formulas inside tables.
- Use plain-text formulas for implementation logic.
- Use math blocks only for the most important equations.
- Keep all configuration in YAML.
- Save all experiment outputs to CSV.
- Save all trigger sets to JSON.
- Save all signatures with metadata.
- Save all random seeds.
- Clearly label assumptions in README and experiment logs.
- Do not hard-code paper numbers as model outputs.
- Regenerate paper-style numbers from actual runs.

---

# 26. Additional ML-Specific Requirements

## Hyperparameters explicitly mentioned in the paper

| Hyperparameter / setting | Value |
|---|---|
| Pruning | Up to 50% |
| Main pruning table | 30% |
| Trigger sizes | 5, 10, 20, 30 |
| Threshold offsets | -0.05, -0.02, 0.00, +0.02, +0.05 |
| Latency target | Below 200 ms |
| B1 latency | 100 ms |
| B2 latency | 130 ms |
| B3 latency | 160 ms |
| B4 latency | 190 ms |
| Quantization | int8 |
| DistilBERT parameters | Approximately 66M |
| MobileNetV2 parameters | Approximately 3.5M |
| CLIP parameters | Approximately 150M according to one paper section |
| ViLT parameters | Approximately 86M |
| ViLT transformer depth | 12 layers |
| Hardware | RTX 3090, Jetson Nano, Raspberry Pi 4 |
| Frameworks | PyTorch 2.0, HuggingFace, torchvision |
| Explainability libraries | Captum, TorchCAM |

## Dataset split methodology

Not specified in the paper.

Required implementation behavior:

- Use standard public splits when available.
- Create validation split from training data.
- Prevent trigger leakage.
- Keep trigger train, validation, and test sets separate.
- Save split indices.

## Training-time augmentation vs evaluation-time preprocessing

Not specified in the paper.

Recommended behavior:

| Phase | Recommended behavior |
|---|---|
| Training | Use standard augmentation for image tasks |
| Evaluation | Use deterministic preprocessing only |
| Trigger evaluation | Do not use random augmentation |
| Robustness evaluation | Use same trigger set and same threshold |

## Reported scores

Directly stated:

- Table 8 reports mean ± standard deviation.
- Other tables appear to report aggregate or representative values.

Not specified:

- Whether Tables 3 to 7 are single-run, mean, or best-run.
- Number of runs behind Table 8.

Recommended behavior:

- Report both individual seed values and mean ± std.
- Use same seeds for all baselines.

## Explainability requirements

Directly stated:

- SHAP for text.
- Grad-CAM and Score-CAM for images.
- Attention rollout for transformers.
- Explanation is supportive only.

Implementation requirements:

- Save raw attribution scores.
- Save visualization.
- Save trigger mask or token span.
- Compute alignment score using declared formula.
- Do not claim causal proof.

## Robustness requirements

Directly stated attacks:

- Pruning.
- Fine-tuning.
- Distillation.
- Black-box verification.

Implementation requirements:

- Evaluate clean model first.
- Apply attack.
- Re-evaluate using same trigger set.
- Use same threshold unless explicitly testing recalibration.
- Report detection accuracy after attack.
- Report robustness drop and retained robustness.

## Optimization distinction

Do not confuse:

- The model training optimizer, such as AdamW.
- The proposed watermark embedding objective.
- The verification algorithm.

They are separate parts of the implementation.

---

# Questions Codex Should Not Guess About

Codex should not silently invent answers to the following:

1. Exact trigger phrases.
2. Exact image trigger masks or perturbation strength.
3. Exact multimodal trigger pairs.
4. Exact loss weights.
5. Exact margin value.
6. Exact learning rate.
7. Exact batch size.
8. Exact number of epochs.
9. Exact random seeds used in the paper.
10. Exact CLIP backbone used for reported numbers.
11. Exact attribution alignment formula.
12. Exact student model used for distillation.
13. Exact fine-tuning attack dataset.
14. Exact ONNX/TensorRT latency settings.
15. Exact threshold values.

If these are needed, Codex should expose them as configuration fields and document the chosen defaults.

---

# Safe Defaults Codex May Use

These defaults are acceptable for a practical reproduction attempt:

| Component | Safe default |
|---|---|
| Text model | `distilbert-base-uncased` |
| Image model | torchvision MobileNetV2 pretrained on ImageNet |
| CLIP model | `openai/clip-vit-base-patch32` |
| ViLT model | HuggingFace ViLT base checkpoint |
| Text max length | 128 |
| Image size | 224 x 224 |
| Optimizer | AdamW |
| Transformer learning rate | 2e-5 |
| MobileNetV2 learning rate | 1e-4 full fine-tuning or 1e-3 classifier only |
| Weight decay | 0.01 |
| Threshold selection | Maximize validation F1 |
| Embedding normalization | True |
| Signature method | Centroid of trigger embeddings |
| Main trigger count | 30 |
| Trigger-size ablation | 5, 10, 20, 30 |
| Main pruning | 30% |
| Seeds | 0, 1, 2, 3, 4 |
| Quantization | int8 |
| Deployment fallback | ONNX Runtime |
| Text attribution | SHAP via Captum |
| Image attribution | Grad-CAM / Score-CAM via TorchCAM |
| Multimodal attribution | Attention rollout |

---

# Final Implementation Reminder

The goal is not to hard-code the paper's reported results. The goal is to build a configurable experimental system that can attempt to reproduce those results under clearly documented assumptions.

Every result should be generated from code, saved to CSV, and traceable to:

- model checkpoint
- dataset split
- trigger set
- signature file
- threshold
- seed
- attack setting
- config file
