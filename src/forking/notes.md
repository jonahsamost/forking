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
