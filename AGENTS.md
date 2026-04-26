# AGENTS.md

## Purpose
This repository is for reproducing experiments from a research paper.

The goal is to build a clean, modular, and reproducible implementation that follows the paper as closely as possible while making all assumptions explicit.

## Source of truth
Before making any changes, read these files in order:

1. `CODEX_HANDOFF.md`
2. `README.md`
3. `docs/assumptions.md` (if it exists)
4. `docs/reproduction_log.md` (if it exists)

`CODEX_HANDOFF.md` is the main technical implementation brief and should be treated as the primary source of truth for:
- problem definition
- datasets and splits
- preprocessing
- architecture
- loss functions
- training setup
- inference setup
- evaluation protocol
- ablations
- expected outputs

If there is a conflict:
- `CODEX_HANDOFF.md` takes priority for paper-specific technical decisions.
- `AGENTS.md` takes priority for repo behavior and engineering discipline.
- `README.md` takes priority for project usage notes only when it does not conflict with the two files above.

## Core working rules
- Do not silently guess critical paper details.
- When a detail is missing from the paper, use the safest reasonable default and document it clearly.
- Record every non-trivial assumption in `docs/assumptions.md`.
- Record important implementation decisions, blockers, and deviations in `docs/reproduction_log.md`.
- Keep the implementation modular, readable, and reproducible.
- Prefer explicit configuration over hard-coded values.
- Keep experiment logic separate from model code.
- Do not mix training, evaluation, and visualization logic in a single script unless explicitly required.
- Make minimal, testable progress in stages.
- When blocked, do not stop silently: document the blocker, state its impact, and continue with the highest-confidence validated subset of the work.

## Implementation priorities
Build in this order unless `CODEX_HANDOFF.md` says otherwise:

1. repository scaffolding
2. config system
3. data loading and preprocessing
4. model architecture
5. loss/objective implementation
6. training loop
7. evaluation pipeline
8. visualization / artifact generation
9. ablation support
10. reproduction scripts for final experiments

## Expected repository structure
Use or maintain a structure close to this:
repo/
├─ AGENTS.md
├─ CODEX_HANDOFF.md
├─ README.md
├─ configs/
│ ├─ base.yaml
│ ├─ train/
│ ├─ eval/
│ └─ ablations/
├─ src/
│ ├─ data/
│ ├─ models/
│ ├─ losses/
│ ├─ training/
│ ├─ evaluation/
│ ├─ visualization/
│ └─ utils/
├─ scripts/
├─ tests/
├─ docs/
└─ outputs/


## Coding conventions
- Use Python unless the project explicitly requires otherwise.
- Prefer clear function boundaries and small modules.
- Add docstrings to public classes/functions.
- Use descriptive variable names matching the paper where possible.
- Keep all hyperparameters in config files.
- Make random seeds configurable.
- Avoid hidden defaults unless they are framework defaults and documented.
- Keep file names and module names predictable.
- Prefer simple, inspectable implementations over clever abstractions unless complexity is clearly justified.

## Dependency and environment discipline
- Do not add or upgrade dependencies unless needed for the paper or for reproducibility.
- Prefer widely used, stable libraries.
- Document any added package, version constraint, or environment-specific requirement.
- If the implementation depends on CUDA, specific PyTorch versions, external repositories, or system packages, record that clearly in `README.md` and `docs/reproduction_log.md`.

## Reproducibility rules
- Make seed handling explicit.
- Make device selection explicit.
- Log package versions when possible.
- Save configs with outputs/checkpoints.
- Save evaluation outputs in a reproducible format.
- Ensure train/val/test split logic is deterministic when required.
- Keep metrics calculation isolated and testable.
- Save enough metadata with each run so results can be traced back to code, config, and assumptions.

## Run and output discipline
- Each experiment should have a clear run name or experiment ID.
- Store outputs in a dedicated subdirectory such as `outputs/<experiment_id>/`.
- Save at minimum:
  - config snapshot
  - key metrics
  - checkpoints if applicable
  - generated tables/figures
  - a short run summary
- Do not overwrite prior outputs unless explicitly instructed.
- Keep baseline, main-method, and ablation outputs easy to compare.

## Validation requirements
Before considering any step complete, verify the following where relevant:

### Data
- dataset loads without errors
- split sizes match expectations
- class labels / targets are correct
- preprocessing is applied in the correct order
- no obvious train/test leakage

### Model
- tensor shapes are correct
- forward pass works on a small batch
- parameter counts are reasonable
- frozen/trainable modules match the spec

### Training
- loss decreases on a tiny overfit test
- checkpoints save and reload properly
- optimizer and scheduler behave as configured

### Evaluation
- metrics match expected definitions
- outputs are saved in the required format
- tables/plots needed by the paper can be generated

## Assumption handling
Whenever paper details are missing, ambiguous, or inconsistent:

1. note the issue in `docs/assumptions.md`
2. mark whether the choice is:
   - explicit from paper
   - inferred from paper
   - safe default
3. explain why the chosen assumption is reasonable
4. keep the assumption configurable whenever possible

## Experiment discipline
- One config per experiment where practical.
- Name experiments clearly.
- Keep baseline runs separate from proposed-method runs.
- Keep ablation configs isolated and comparable.
- Do not overwrite outputs from previous experiments.
- When reproducing tables or figures from the paper, preserve a clear mapping between paper artifact names and generated files.

## Completion criteria
Do not claim reproduction is complete until:
- the required experiments run end-to-end
- the required metrics are produced
- the required tables/figures/artifacts are generated
- mismatches with the paper are documented
- unresolved critical unknowns are clearly listed

## Behavior on uncertainty
If a missing detail materially affects correctness:
- do not bury the issue
- implement the safest documented assumption
- surface the uncertainty clearly in code comments and `docs/assumptions.md`
- note the impact in `docs/reproduction_log.md`



## Fast iteration workflow
Work in four stages:

1. Full implementation draft
   - Generate the complete repository structure and the best full implementation draft possible from the paper.
   - Include data pipeline, model modules, losses, training loop, evaluation pipeline, configs, artifact generation hooks, and ablation support where feasible.
   - Create smoke, debug, and full run modes from the start.
   - Document all assumptions in `docs/assumptions.md`.
   - Treat this stage as reviewable but not yet validated.

2. Smoke test
   - Run on a tiny reproducible subset.
   - Verify data loading, tensor shapes, forward pass, backward pass, loss computation, checkpoint saving/loading, and metric calculation.

3. Debug subset run
   - Run on a small reproducible subset of the real dataset.
   - Verify short training behavior, evaluation behavior, config handling, and output generation.

4. Full experiment readiness
   - Only after smoke and debug subset runs pass should the project be considered ready for full-scale runs.

## Required commands
Codex should create and maintain commands equivalent to:

- `python scripts/train.py --config configs/smoke.yaml`
- `python scripts/train.py --config configs/debug.yaml`
- `python scripts/train.py --config configs/full.yaml`
- `python scripts/eval.py --config configs/debug.yaml`

## Sampling rules
- Small subsets must be reproducible via fixed seed.
- Preserve class balance when possible.
- Preserve subject/group boundaries when required.
- Do not use naive first-N slicing if it risks leakage or class distortion.

## Done when
A training pipeline is not ready for full runs until:
- the full implementation draft is present and reviewable
- smoke test passes end-to-end
- tiny-batch overfit sanity check passes
- debug subset run completes successfully
- outputs are written to the expected directories
- assumptions, blockers, and deviations are documented



## First action
Start by reading `CODEX_HANDOFF.md`, then scaffold the repository and implement the smallest possible validated path toward the first reproducible experiment.



## Environment setup and testing (Codex)
To keep this repository reproducible in Codex-like clean containers:

1. Install dependencies with:
   - `pip install -r requirements-dev.txt`
2. Run validation commands:
   - `pytest -q`
   - `python scripts/train.py --config configs/smoke.yaml`
   - `python scripts/train.py --config configs/debug.yaml`
   - `python scripts/train.py --config configs/full.yaml`
   - `python scripts/eval.py --config configs/debug.yaml`

If dependency errors occur, treat them as environment bootstrap issues first and resolve by reinstalling `requirements-dev.txt` before changing implementation code.

