# Entropy V2: Logistic Regression Intervention Plan

## Goal

Replace the hand-tuned VIX threshold trigger with a lightweight CPU logistic regression classifier that predicts failure risk from entropy/VIX trajectory features.

The classifier is trained on the trainer side, where rewards are available, and used on the vLLM server side for online intervention decisions during generation.

Primary success metric:

- `train/entropy/treatment_minus_control_success_rate`

Secondary diagnostics:

- `train/entropy/treatment_success_rate`
- `train/entropy/control_success_rate`
- `train/entropy/treatment_noop_rate`
- `train/entropy/intervention_improvement_rate`
- `train/completions/mean_length`
- classifier failure/success trigger frontier

## High-Level Architecture

### Trainer / Worker Side

The trainer owns learning because it has access to rewards.

Responsibilities:

1. Collect non-intervened control completions.
2. Receive raw top-k token logprob traces from the server.
3. Compute entropy trajectories from top-k logprobs.
4. Compute rolling VIX/window features from entropy trajectories.
5. Store whole rollout-level calibration records in FIFO buffers.
6. Expand rollout records into window-level training examples when fitting.
7. Train a lightweight logistic regression classifier on CPU.
8. Choose a probability threshold using a max-success-trigger-rate constraint.
9. Push classifier parameters to the vLLM server.
10. Log classifier quality, calibration frontier, and treatment/control metrics.

Classifier training should run in its own background thread so rollout scoring/training does not block on fitting.

### vLLM Server Side

The server owns online inference because split decisions happen during generation.

Responsibilities:

1. For non-intervention/control requests:
  - Generate normally.
  - Return token IDs, sampled token logprobs, and compact top-k logprob traces.
  - Do not compute reward-dependent calibration.
  - Ideally avoid computing VIX except for optional debug metadata.
2. For intervention/treatment requests:
  - Generate lookahead chunks.
  - Compute local entropy/VIX/window features online.
  - Run classifier inference using the latest weights from the trainer.
  - Detect high-failure-risk windows.
  - Backtrack to the beginning of the detected window.
  - Sample branches and select the best continuation.
  - Return standard completion outputs plus intervention metadata.
3. Expose a classifier update endpoint:
  - `POST /entropy_v2/update_classifier`
  - Receives logistic regression weights, bias, feature normalization stats, decision threshold, feature names, and version.
  - Updates `EntropyHandler` classifier state atomically.
  - Logs model version and threshold.

## Structural Code Changes

The `entropy_v2` package has been copied from `entropy`. Update imports so v2 is isolated:

- `forking.entropy_v2.models`
- `forking.entropy_v2.entropy_handler`
- `forking.entropy_v2.entropy_updates`
- `forking.entropy_v2.features`
- `forking.entropy_v2.vllm_server`
- `forking.entropy_v2.vllm_run`
- `forking.entropy_v2.train`

Create a dedicated shared feature extraction module:

- `src/forking/entropy_v2/features.py`

This module must be imported by both the trainer side and the server side. It is the single source of truth for:

- computing top-k entropy from top-k logprobs
- computing rolling VIX values
- turning entropy/VIX windows into classifier feature vectors
- defining `FEATURE_NAMES`
- normalizing token position / completion length features

Do not duplicate feature construction in `entropy_updates.py` and `entropy_handler.py`. The classifier only makes sense if training-time features and inference-time features are exactly the same.

Update TRL integration to import v2:

- `src/trl/trl/experimental/async_grpo/async_rollout_worker.py`
  - import `EntropyUpdateTracker` from `forking.entropy_v2.entropy_updates`
  - import protocol constants from `forking.entropy_v2.models`
- `src/trl/trl/experimental/async_grpo/async_grpo_trainer.py`
  - import `EntropyUpdateTracker` from `forking.entropy_v2.entropy_updates`

Update v2 scripts:

- `entropy_v2/scripts/run_vllm.sh`
  - launch `forking.entropy_v2.vllm_run`
- `entropy_v2/scripts/run_trainer.sh`
  - launch `forking.entropy_v2.train`

Use a distinct endpoint path to avoid confusion:

```yaml
vllm:
  completions_endpoint: "/v1/entropy_v2/completions"
```

## Data Contract

### Server -> Worker For Control Requests

The non-intervention server path should return compact top-k logprob traces.

Example response metadata:

```json
{
  "entropy": {
    "topk_logprobs": [
      [-0.1, -2.0, -3.1],
      [-0.2, -1.8, -4.0]
    ],
    "topk": 3
  }
}
```

Existing standard fields remain:

- `choices[0].token_ids`
- `choices[0].logprobs.token_logprobs`

The trainer uses `topk_logprobs` to compute entropy and VIX.

### Trainer -> Server Classifier Update

New endpoint:

```text
POST /entropy_v2/update_classifier
```

Payload:

```json
{
  "version": 12,
  "feature_names": ["entropy", "vix", "drift", "drawup"],
  "weights": [0.1, -0.2, 0.3, 0.4],
  "bias": -0.4,
  "feature_mean": [0.2, 0.3, 0.0, 0.5],
  "feature_std": [0.1, 0.2, 0.01, 0.3],
  "threshold": 0.72,
  "max_success_trigger_rate": 0.10
}
```

The server stores these params on `EntropyHandler`.

### Worker -> Server Completion Request

Treatment requests can continue to pass:

```json
{
  "vllm_xargs": {
    "intervene": 1,
    "classifier_version": 12
  }
}
```

Classifier weights should not be passed on every request. v2 should use the HTTP update endpoint as the required synchronization mechanism.

## Data Structures

### Rollout-Level Record

Keep FIFO buffers at the whole-completion level.

```python
@dataclass(frozen=True)
class EntropyWindowFeatures:
    entropy: float
    vix: float
    # ...

    def as_vector(self) -> list[float]:
        ...

@dataclass(frozen=True)
class EntropyClassifierRecord:
    reward: float
    completion_len: int
    features: list[EntropyWindowFeatures]
    feature_names: list[str]
```

The record stores derived per-window classifier feature vectors rather than raw flat examples. Training expands records into examples at fit time.

Feature creation happens once when the control completion is processed by the tracker:

```python
topk_logprobs -> entropy trajectory -> VIX windows -> feature vectors
```

The record is still completion-level, which lets us weight windows by completion and choose later whether to train on all windows or only selected windows.

Separate buffers:

```python
success_records: deque[EntropyClassifierRecord]
failure_records: deque[EntropyClassifierRecord]
```

Suggested config:

```yaml
entropy:
  classifier_max_records_per_class: 2048
  classifier_windows_per_completion: null
  classifier_window_selection: "all"
```

Keeping whole rollout records is preferable because:

- feature extraction can be changed later without changing the collection contract too much
- completion-level weighting is easier
- long completions do not have to dominate training
- we can experiment with all windows, top-k windows, or sampled windows

## Feature Extraction

All feature construction lives in `forking.entropy_v2.features`.

From each completion:

1. Compute entropy trajectory from top-k logprobs.
2. Compute rolling VIX over the same `chunk_size` used by inference.
3. Build sampled window-level feature rows.

The classifier window size should not be a separate concept from the inference window. It should use the same entropy chunk size:

```yaml
entropy:
  threshold_chunk_size: 64
```

This means classifier examples and server-side intervention decisions both describe the same 64-token rolling window.

Fixed `classifier_feature_set: "v1"`:

- `entropy`
- `vix`
- `drift`
- `up_vix`
- `down_vix`
- `drawup`
- `drawdown`
- `token_idx_norm`
- `vix_max_so_far`
- `drawup_max_so_far`
- `entropy_max_so_far`
- `entropy_min_so_far`

The Phase 3 classifier uses the raw 64-token entropy window as its primary input. The VIX/summary feature vector remains useful for diagnostics, ablations, and possible later concatenation with the raw entropy window.

Training expands rollout records into window-level examples:

```python
for record in records:
    y = float(record.reward <= 0)
    for features in select_training_windows(record.features):
        X.append(features)
        y.append(y)
```

Default training should use every sampled window from each completion.

Make window selection parameterized:

```yaml
entropy:
  classifier_windows_per_completion: null  # null means all windows
  classifier_window_selection: "all"       # later: "top_risk", "random", "top_vix"
```

For v1 of the classifier:

- train on all windows by default
- label all windows from a failed completion as failed
- label all windows from a successful completion as successful

This is weak supervision, but it avoids biasing the logistic regression model with the same VIX-like heuristic we are trying to improve on. The model should see the full distribution of quiet, noisy, successful, and failed windows and learn its own separating direction.

Top-risk or sampled-window training can remain a future configurable experiment, but it is not part of the v1 default.

Use per-completion weighting so long completions do not dominate:

```python
window_weight = class_weight / num_windows
```

## Classifier Training

Implement CPU logistic regression in `entropy_updates.py`, with shared feature extraction in `features.py`.

Model:

```text
p_fail = sigmoid(w . x + b)
```

Preferred first version: a tiny CPU `torch.nn.Module` that is effectively a single linear layer:

```python
class EntropyFailureClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
```

This should run on CPU, not the trainer GPU. The model size is tiny and CPU training avoids interfering with GRPO updates.

The "size" of the classifier is mostly the feature dimension:

```yaml
entropy:
  classifier_feature_set: "v1"
```

For now, logistic regression means no hidden layer. If we later want more capacity, add a separate config like:

```yaml
entropy:
  classifier_hidden_dim: null  # null means pure linear/logistic regression
```

Server inference should not require `torch`. The trained parameters are exported as plain lists:

```python
weights: list[float]
bias: float
feature_mean: list[float]
feature_std: list[float]
threshold: float
```

The server computes the same sigmoid manually.

Training loop:

```python
z = X @ w + b
p = sigmoid(z)
loss = weighted_binary_cross_entropy + l2
grad_w = X.T @ (weights * (p - y)) / sum(weights) + l2 * w
grad_b = sum(weights * (p - y)) / sum(weights)
```

Use feature standardization:

```python
X_norm = (X - mean) / std
```

Use class balancing:

```python
success_weight = 0.5 / n_success_windows
failure_weight = 0.5 / n_failure_windows
```

Suggested config:

```yaml
entropy:
  classifier_update_interval: 64
  classifier_min_success_records: 32
  classifier_min_failure_records: 32
  classifier_l2: 0.0
  classifier_train_steps: 200
  classifier_learning_rate: 0.1
  classifier_frontier_caps: [0.01, 0.05, 0.075, 0.10, 0.125]
  classifier_feature_set: "v1"
  classifier_hidden_dim: null
```

## Decision Threshold

Do not use `p_fail > 0.5`.

After training, evaluate predicted probabilities on buffered success/failure examples.

Select classifier threshold using the same operational constraint:

```yaml
entropy:
  max_success_trigger_rate: 0.10
```

Choose threshold that:

1. satisfies `success_trigger_rate <= max_success_trigger_rate`
2. maximizes `failure_trigger_rate`
3. falls back to minimum success trigger rate if no threshold satisfies the cap

Log a frontier:

```text
cap<=0.01 success=... failure=...
cap<=0.05 success=... failure=...
cap<=0.10 success=... failure=...
```

## Background Training Thread

Classifier training should not block rollout scoring.

Trainer-side tracker/manager design:

```python
class EntropyUpdateTracker:
    def update_from_scored_group(...):
        add_control_records()
        classifier_manager.maybe_enqueue_training(...)
        classifier_manager.maybe_install_completed()
        return metrics

class EntropyClassifierManager:
    def maybe_enqueue_training(...):
        if enough_new_records and no_training_job_running:
            submit_background_job()

    def maybe_install_completed(...):
        if job_complete:
            install_classifier_params()
```

Use `ThreadPoolExecutor(max_workers=1)` or a dedicated thread inside `entropy_v2/classifier_manager.py`. Keep the fitting code itself in `entropy_v2/classifier.py`.

Important:

- Snapshot buffers before training.
- Do not train while holding locks.
- Install params atomically when training finishes.
- Phase 3 only installs params on the trainer side for metrics.
- Phase 4 pushes params to the server after install.
- Keep training failures non-fatal to rollout generation.

## Server Classifier Inference

On the vLLM side, `EntropyHandler` stores current classifier params:

```python
self.classifier_params: ClassifierParams | None
```

During intervention:

1. Compute current window features.
2. Standardize using stored mean/std.
3. Compute `p_fail = sigmoid(w . x + b)`.
4. `_find_split_idx` returns the index with highest `p_fail` above threshold.
5. Split at `detected_idx - chunk_size + 1`.
6. Continue to allow up to `max_interventions` per completion.
7. Use classifier failure probability as the primary branch-selection score.

Branch scoring is conceptually the same loop as the current VIX version, but replaces the hand-written VIX score with a classifier forward pass:

1. Generate branch candidates.
2. Extract top-k logprobs from each candidate.
3. Compute entropy trajectory.
4. Compute 64-token rolling VIX/window features.
5. Standardize features using trainer-provided stats.
6. Run classifier forward pass on each branch window.
7. Score the branch by classifier failure probability.

v1 branch scoring should minimize the max classifier failure probability across branch windows:

```text
branch_score = max(p_fail(window) for window in branch_windows)
```

This directly asks: "does this branch still contain a high-risk unstable span?"

Mean probability or mixed probability/smoothness scores can be future experiments, but are not part of the v1 implementation.

Fallback:

- If no classifier params are ready, do not intervene.

Once classifier params are ready, all intervention decisions should be based on classifier probability. The threshold-style VIX trigger is not part of v2 intervention logic except as an optional debugging comparison metric.

## Metrics To Log

Classifier/training metrics:

- `entropy/classifier_ready`
- `entropy/classifier_version`
- `entropy/classifier_train_records_success`
- `entropy/classifier_train_records_failure`
- `entropy/classifier_train_windows_success`
- `entropy/classifier_train_windows_failure`
- `entropy/classifier_loss`
- `entropy/classifier_auc` if easy
- `entropy/classifier_threshold`
- `entropy/classifier_trigger_rate_success`
- `entropy/classifier_trigger_rate_failure`
- `entropy/classifier_youden_j`
- `entropy/classifier_frontier_*`

Treatment/control metrics remain primary:

- `entropy/treatment_success_rate`
- `entropy/control_success_rate`
- `entropy/treatment_minus_control_success_rate`
- `entropy/treatment_noop_rate`
- `entropy/avg_interventions`
- `entropy/intervention_improvement_rate`
- `completions/mean_length`

## Implementation Phases

### Phase 1: v2 Import and Endpoint Isolation

- Ensure all v2 files import from `forking.entropy_v2`.
- Update TRL worker/trainer imports to v2.
- Use `/v1/entropy_v2/completions`.
- Confirm v2 smoke run works.

### Phase 2: Move Control Feature Extraction To Trainer

- Server returns compact top-k logprob traces for control completions.
- `features.py` computes entropy/VIX/features from top-k logprobs.
- Tracker stores completion-level feature records in success/failure FIFO buffers.
- Confirm metrics match previous server-computed VIX.

### Phase 3: Offline Classifier Training In Tracker

- Store rollout-level classifier records.
- Expand records into window examples at training time.
- Train on all sampled windows by default.
- Label all failed-completion windows as failed and all successful-completion windows as successful.
- Fit CPU logistic regression using a one-layer `nn.Module`.
- Log classifier frontier only.
- Do not use classifier for intervention yet.

### Phase 4: Push Classifier Params To Server

- Add `/entropy_v2/update_classifier`.
- Tracker pushes params after training finishes.
- Server stores classifier params.
- Log classifier version on server and worker.

### Phase 5: Classifier-Based Intervention

- Server uses classifier probability in `_find_split_idx`.
- Backtrack to beginning of detected window.
- Keep `max_interventions`.
- Do not intervene until classifier params are ready.
- Use classifier probability as the branch-selection score, initially minimizing max `p_fail` across branch windows.
- Compare classifier treatment/control lift against v1 threshold system.

## Fixed Decisions

- v2 stores derived per-window feature vectors in rollout-level records.
- Feature creation lives in `forking.entropy_v2.features` and is shared by trainer and server.
- Classifier parameter updates use the HTTP endpoint, not per-request `vllm_xargs`.
- Server does not intervene until classifier params are ready.
- Once classifier params are ready, split detection is based on classifier probability.
- Branch selection minimizes max classifier failure probability across branch windows.
- `max_interventions` remains in force.
- Classifier window size is the same as inference `threshold_chunk_size` / VIX window size.
- Training uses all sampled windows by default.
- Window selection remains configurable via `classifier_window_selection` and `classifier_windows_per_completion`.
- `classifier_feature_set: "v1"` is the feature list defined in the Feature Extraction section.

## Future Experiments

- Try mean classifier probability or mixed probability/smoothness branch scoring.
- Try top-risk, random, or top-vix window selection instead of all-window training.

