"""
Utility functions for the Behavioral Drift Detection project.
"""

import os
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Trade strict reproducibility for ~2-3x GPU throughput.
        # benchmark=True lets cuDNN pick the fastest kernel for fixed shapes;
        # deterministic=False allows non-deterministic but faster algorithms.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device(device_str: str = "cuda") -> torch.device:
    """Get PyTorch device."""
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def compute_z_score(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Compute z-scores for drift detection."""
    if std == 0:
        return np.zeros_like(values)
    return (values - mean) / std


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    
    train_end = int(train_ratio * len(data))
    val_end = train_end + int(val_ratio * len(data))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    if isinstance(data, np.ndarray):
        return data[train_idx], data[val_idx], data[test_idx]
    elif isinstance(data, list):
        return ([data[i] for i in train_idx],
                [data[i] for i in val_idx],
                [data[i] for i in test_idx])
    else:
        return data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Save metrics to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Load metrics from YAML file."""
    with open(filepath, 'r') as f:
        metrics = yaml.safe_load(f)
    return metrics
