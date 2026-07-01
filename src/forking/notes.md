Baseline
- scarlet-universe-7

Entropy + VIX
- atomic-lion-18
    entropy:
    threshold_chunk_size: 64
    topk_entropy: 3
    max_interventions: 3
    num_samples: 3
    update_interval: 64
    calibration_batches: 1 
    calibration_ema: 0.95
    max_success_trigger_rate: 0.1
    max_records: 1024

Entropy V2
- wild-glade-30
look at train/entropy/cumulative_treatment_minus_control
and interventions logs
entropy:
  threshold_chunk_size: 48
  topk_entropy: 3
  max_interventions: 4
  num_samples: 4
  max_success_trigger_rate: 0.1
  max_records: 2048
  classifier_update_interval: 64 # update after 64 new completions
  classifier_min_success_records: 128
  classifier_min_failure_records: 128
  classifier_train_steps: 1000
  classifier_learning_rate: 0.01
  classifier_l2: 0.0
  classifier_feature_mode: "combined"
  classifier_hidden_dims: [128, 64, 32]
  classifier_inference_stride: 16
  classifier_frontier_caps: [0.10, 0.15, 0.20, 0.25]


Entropy V2 -- starting at better token within chunk
denim-deluge-31


Entropy V2 -- starting at better token within chunk
drawn-galaxy-33
  max_records: 256