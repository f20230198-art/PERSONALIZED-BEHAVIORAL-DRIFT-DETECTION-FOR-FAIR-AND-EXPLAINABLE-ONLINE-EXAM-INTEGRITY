"""
Main pipeline for Behavioral Drift Detection in Online Exams.

Usage:
    python main.py --config configs/config.yaml --mode [preprocess|train|evaluate|explain|all]
"""

import argparse
import os
import numpy as np
import torch
from pathlib import Path

# Import modules
from src.utils import load_config, set_seed, get_device, ensure_dir, save_metrics
from src.data_loader import (EdNetDataLoader, SessionCreator, OULADLoader,
                             merge_demographics_with_sessions, pool_demographics,
                             encode_demographics)
from src.feature_extraction import BehavioralFeatureExtractor, save_features, load_features
from src.preprocessing import SyntheticAnomalyGenerator, DataPreprocessor, save_processed_data, load_processed_data
from src.models import LSTMAutoencoder, TransformerAutoencoder, DemographicDiscriminator, compute_reconstruction_error
from src.train import Trainer, compute_drift_scores, compute_blended_drift_scores, train_baseline_models
from src.evaluate import evaluate_model, select_optimal_threshold, compare_models, compute_classification_metrics, compute_precision_at_k
from src.fairness import FairnessAnalyzer, compare_fairness_before_after
from src.explainability import SHAPExplainer, SequentialSHAPExplainer, generate_explanation_report


def preprocess_data(config):
    """Step 1: Load data, create sessions, extract features."""
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80 + "\n")

    # Load EdNet data
    num_students = config['data'].get('num_students', 5000)
    print(f"Loading EdNet-KT2 data ({num_students} students)...")
    ednet_loader = EdNetDataLoader(config['data']['ednet_path'])
    students_data = ednet_loader.load_batch(start_idx=0, batch_size=num_students)

    # Create exam sessions
    print("\nCreating exam sessions...")
    session_creator = SessionCreator(
        min_questions=config['data']['session_min_questions'],
        max_questions=config['data']['session_max_questions']
    )
    sessions, student_ids = session_creator.create_all_sessions(students_data)

    # Load OULAD demographics
    print("\nLoading OULAD demographic data...")
    oulad_loader = OULADLoader(config['data']['oulad_path'])
    oulad_df = oulad_loader.load_student_info()

    # Merge demographics
    print("\nMerging demographics with sessions...")
    merged_sessions = merge_demographics_with_sessions(sessions, student_ids, oulad_df)
    sessions = [s[0] for s in merged_sessions]
    demographics = [s[1] for s in merged_sessions]

    # Pool sparse demographic groups (age_band -> binary, imd_band -> 3 groups)
    print("Pooling demographic groups for fairness analysis...")
    demographics = pool_demographics(demographics)

    # ---- Extract SESSION-LEVEL features (for baselines) ----
    print("\nExtracting session-level features (for baselines)...")
    extractor = BehavioralFeatureExtractor()
    session_features, _ = extractor.extract_features_batch(sessions)
    _, sess_mean, sess_std = extractor.normalize_features(session_features)

    # ---- Extract QUESTION-LEVEL features (for LSTM) ----
    print("\nExtracting question-level features (for LSTM)...")
    max_seq_len = config['model'].get('max_seq_len', 50)
    question_features_clean, seq_lengths_clean = extractor.extract_question_features_batch(
        sessions, max_seq_len=max_seq_len
    )
    # Compute normalization stats on CLEAN data only
    _, q_mean, q_std = extractor.normalize_question_features(
        question_features_clean, seq_lengths_clean
    )

    # ---- Inject synthetic anomalies ----
    print("\nInjecting synthetic anomalies...")
    anomaly_generator = SyntheticAnomalyGenerator(
        contamination_rate=config['data']['synthetic_contamination'],
        seed=config['training']['seed']
    )
    sessions_with_anomalies, labels = anomaly_generator.inject_anomalies(sessions)

    # Re-extract BOTH feature levels from anomalous sessions
    print("\nRe-extracting features after anomaly injection...")
    session_features_final, _ = extractor.extract_features_batch(sessions_with_anomalies)
    session_features_final_norm, _, _ = extractor.normalize_features(
        session_features_final, mean=sess_mean, std=sess_std
    )

    question_features_final, seq_lengths_final = extractor.extract_question_features_batch(
        sessions_with_anomalies, max_seq_len=max_seq_len
    )
    question_features_final_norm, _, _ = extractor.normalize_question_features(
        question_features_final, seq_lengths_final, mean=q_mean, std=q_std
    )

    # ---- Split data CHRONOLOGICALLY per student ----
    # Group sessions by student and sort by time within each student,
    # then split 70/15/15 so train=earliest, val=middle, test=latest.
    # This prevents temporal leakage (model never sees future sessions).
    print("\nSplitting data chronologically per student...")
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']

    # Get the start timestamp of each session for chronological ordering
    session_start_times = []
    for sess in sessions_with_anomalies:
        if 'timestamp_sec' in sess.columns and len(sess) > 0:
            session_start_times.append(sess['timestamp_sec'].min())
        else:
            session_start_times.append(0)

    # Group session indices by student
    from collections import defaultdict
    student_session_map = defaultdict(list)
    for idx, sid in enumerate(student_ids):
        student_session_map[sid].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for sid, sess_indices in student_session_map.items():
        # Sort this student's sessions by start time
        sess_indices_sorted = sorted(sess_indices, key=lambda i: session_start_times[i])
        n_s = len(sess_indices_sorted)
        train_end_s = max(1, int(train_ratio * n_s))
        val_end_s = max(train_end_s + 1, int((train_ratio + val_ratio) * n_s))
        val_end_s = min(val_end_s, n_s)  # Safety bound

        train_idx.extend(sess_indices_sorted[:train_end_s])
        val_idx.extend(sess_indices_sorted[train_end_s:val_end_s])
        test_idx.extend(sess_indices_sorted[val_end_s:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    print(f"  Chronological split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Split session-level features (normalized for ML baselines)
    X_train_sess = session_features_final_norm[train_idx]
    X_val_sess = session_features_final_norm[val_idx]
    X_test_sess = session_features_final_norm[test_idx]

    # Also keep RAW (unnormalized) session features for RuleBased detector
    X_train_sess_raw = session_features_final[train_idx]
    X_val_sess_raw = session_features_final[val_idx]
    X_test_sess_raw = session_features_final[test_idx]

    # Split question-level features (for LSTM)
    X_train_seq = question_features_final_norm[train_idx]
    X_val_seq = question_features_final_norm[val_idx]
    X_test_seq = question_features_final_norm[test_idx]

    # Split lengths
    L_train = seq_lengths_final[train_idx]
    L_val = seq_lengths_final[val_idx]
    L_test = seq_lengths_final[test_idx]

    # Split labels
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    # Split demographics
    demo_train = [demographics[i] for i in train_idx]
    demo_val = [demographics[i] for i in val_idx]
    demo_test = [demographics[i] for i in test_idx]

    # Split student IDs (needed for personalized blended drift scoring)
    student_ids_arr = np.array(student_ids)
    sid_train = student_ids_arr[train_idx]
    sid_val = student_ids_arr[val_idx]
    sid_test = student_ids_arr[test_idx]

    # Encode demographics for adversarial fairness training
    fairness_attr = config.get('fairness_training', {}).get('attribute', 'gender')
    demo_labels_all, demo_label_map = encode_demographics(demographics, attribute=fairness_attr)
    demo_labels_train = demo_labels_all[train_idx]
    demo_labels_val = demo_labels_all[val_idx]
    demo_labels_test = demo_labels_all[test_idx]
    num_demo_groups = len(demo_label_map)
    print(f"  Demographic encoding ({fairness_attr}): {demo_label_map}  ({num_demo_groups} groups)")

    # Save processed data
    print("\nSaving processed data...")
    processed_data = {
        # Question-level (LSTM / Transformer)
        'X_train_seq': X_train_seq,
        'X_val_seq': X_val_seq,
        'X_test_seq': X_test_seq,
        'L_train': L_train,
        'L_val': L_val,
        'L_test': L_test,
        # Session-level (baselines, normalized)
        'X_train_sess': X_train_sess,
        'X_val_sess': X_val_sess,
        'X_test_sess': X_test_sess,
        # Session-level (raw/unnormalized, for RuleBased detector)
        'X_train_sess_raw': X_train_sess_raw,
        'X_val_sess_raw': X_val_sess_raw,
        'X_test_sess_raw': X_test_sess_raw,
        # Labels and demographics
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'demo_train': demo_train,
        'demo_val': demo_val,
        'demo_test': demo_test,
        # Demographic integer labels (for adversarial fairness)
        'demo_labels_train': demo_labels_train,
        'demo_labels_val': demo_labels_val,
        'demo_labels_test': demo_labels_test,
        'num_demo_groups': num_demo_groups,
        'demo_label_map': demo_label_map,
        # Normalization stats
        'sess_mean': sess_mean,
        'sess_std': sess_std,
        'q_mean': q_mean,
        'q_std': q_std,
        'feature_names': extractor.feature_names,
        'question_feature_names': extractor.question_feature_names,
        # Student IDs (for personalized blended drift scoring)
        'sid_train': sid_train,
        'sid_val': sid_val,
        'sid_test': sid_test,
    }

    ensure_dir(config['data']['processed_path'])
    save_path = os.path.join(config['data']['processed_path'], 'processed_data.npz')
    save_processed_data(processed_data, save_path)

    n_total = len(labels)
    print(f"\nPreprocessing complete!")
    print(f"  Total sessions: {n_total}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Anomalies: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"  Sequence features shape: {X_train_seq.shape}")
    print(f"  Session features shape: {X_train_sess.shape}")

    return processed_data


def _build_discriminator(config, processed_data):
    """Build DemographicDiscriminator if adversarial fairness is enabled."""
    ft_cfg = config.get('fairness_training', {})
    if not ft_cfg.get('enabled', False):
        return None
    latent_dim = config['model']['latent_dim']
    num_groups = int(processed_data.get('num_demo_groups', 2))
    disc = DemographicDiscriminator(latent_dim, num_groups)
    print(f"  Adversarial fairness: discriminator ({num_groups} groups, "
          f"lambda={ft_cfg.get('lambda', 0.1)})")
    return disc


def train_models(config, processed_data):
    """Step 2: Train LSTM-AE, Transformer-AE, and baseline models."""
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80 + "\n")

    device = get_device(config['training']['device'])
    ensure_dir(config['output']['models_dir'])

    X_train_seq = processed_data['X_train_seq']
    X_val_seq = processed_data['X_val_seq']
    X_test_seq = processed_data['X_test_seq']
    L_train = processed_data['L_train']
    L_val = processed_data['L_val']
    L_test = processed_data['L_test']
    X_train_sess = processed_data['X_train_sess']

    # ----------------------------------------------------------------
    # CRITICAL: train autoencoders on NORMAL sessions only.
    # Anomaly injection happened before the split, so y_train has labels.
    # Training on contaminated data teaches the AE to reconstruct anomalies
    # too, collapsing the reconstruction-error gap we rely on for detection.
    # ----------------------------------------------------------------
    y_train = processed_data['y_train']
    normal_mask = (y_train == 0)
    X_train_seq_clean = X_train_seq[normal_mask]
    L_train_clean = L_train[normal_mask]
    print(f"  AE training on CLEAN sessions only: "
          f"{normal_mask.sum()}/{len(y_train)} "
          f"({normal_mask.mean()*100:.1f}%)")

    # Filter val set to clean sessions only — early stopping should be based
    # on how well the AE reconstructs NORMAL behavior, not anomalies.
    y_val = processed_data['y_val']
    val_normal_mask = (y_val == 0)
    X_val_seq_clean = X_val_seq[val_normal_mask]
    L_val_clean = L_val[val_normal_mask]
    print(f"  AE validation on CLEAN sessions only: "
          f"{val_normal_mask.sum()}/{len(y_val)} "
          f"({val_normal_mask.mean()*100:.1f}%)")

    # Demographic labels for adversarial fairness (may be None if not present)
    dl_train = processed_data.get('demo_labels_train')
    dl_val = processed_data.get('demo_labels_val')
    dl_test = processed_data.get('demo_labels_test')
    # Convert numpy int types if loaded from npz
    if isinstance(dl_train, np.ndarray):
        dl_train = dl_train.astype(np.int64)
        dl_val = dl_val.astype(np.int64)
        dl_test = dl_test.astype(np.int64)
    dl_train_clean = dl_train[normal_mask] if dl_train is not None else None
    dl_val_clean = dl_val[val_normal_mask] if dl_val is not None else None

    # Student IDs for clean training sessions (needed for blended drift scoring)
    sid_train = processed_data.get('sid_train')
    sid_train_clean = sid_train[normal_mask] if sid_train is not None else None

    preprocessor = DataPreprocessor(config)
    # AE loaders: train on clean-only, validate on clean-only
    train_loader, val_loader, _ = preprocessor.create_data_loaders(
        X_train_seq_clean, X_val_seq_clean, X_test_seq,
        batch_size=config['training']['batch_size'],
        lengths_train=L_train_clean, lengths_val=L_val_clean, lengths_test=L_test,
        demo_labels_train=dl_train_clean, demo_labels_val=dl_val_clean,
        demo_labels_test=dl_test
    )

    # Clean-only tensors for drift score normalization.
    # IMPORTANT: normalizing with contaminated data inflates the mean and
    # shrinks the normal/anomalous separation, causing inverted thresholds.
    X_train_clean_tensor = torch.FloatTensor(X_train_seq_clean).to(device)
    L_train_clean_tensor = torch.LongTensor(L_train_clean).to(device)

    # Clean session-level features for Standard AE normalization
    X_train_sess_clean = X_train_sess[normal_mask]

    # ================================================================
    # 2a. LSTM Autoencoder (skip if checkpoint already exists)
    # ================================================================
    lstm_save_dir = os.path.join(config['output']['models_dir'], 'lstm_ae')
    lstm_ckpt = os.path.join(lstm_save_dir, 'best_model.pth')

    lstm_model = LSTMAutoencoder(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )

    force_retrain = config.get('_force_retrain', False)
    if os.path.exists(lstm_ckpt) and not force_retrain:
        print("=" * 60)
        print("LSTM Autoencoder checkpoint found — skipping training.")
        print("  (run with --force-retrain to ignore checkpoints)")
        print("=" * 60)
        ckpt = torch.load(lstm_ckpt, map_location=device)
        lstm_model.load_state_dict(ckpt['model_state_dict'])
        lstm_model.to(device)
        lstm_history = {}
    else:
        print("=" * 60)
        print("Training LSTM Autoencoder...")
        print("=" * 60)
        print(f"  Architecture: input={config['model']['input_dim']}, "
              f"hidden={config['model']['hidden_dim']}, "
              f"latent={config['model']['latent_dim']}, "
              f"layers={config['model']['num_layers']}")
        lstm_disc = _build_discriminator(config, processed_data)
        lstm_trainer = Trainer(lstm_model, config, device, discriminator=lstm_disc)
        ensure_dir(lstm_save_dir)
        lstm_history = lstm_trainer.train(
            train_loader, val_loader,
            epochs=config['training']['epochs'],
            save_dir=lstm_save_dir
        )
        lstm_trainer.load_checkpoint(lstm_ckpt)

    print("\nComputing LSTM-AE training reconstruction errors (clean sessions only)...")
    lstm_train_errors, _ = compute_drift_scores(
        lstm_model, X_train_clean_tensor, device, lengths=L_train_clean_tensor
    )

    # ================================================================
    # 2b. Transformer Autoencoder (skip if checkpoint already exists)
    # ================================================================
    tf_save_dir = os.path.join(config['output']['models_dir'], 'transformer_ae')
    tf_ckpt = os.path.join(tf_save_dir, 'best_model.pth')
    tf_cfg = config['transformer']

    tf_model = TransformerAutoencoder(
        input_dim=tf_cfg['input_dim'],
        d_model=tf_cfg['d_model'],
        nhead=tf_cfg['nhead'],
        num_layers=tf_cfg['num_layers'],
        latent_dim=tf_cfg['latent_dim'],
        max_seq_len=tf_cfg['max_seq_len'],
        dropout=tf_cfg['dropout']
    )

    if os.path.exists(tf_ckpt) and not force_retrain:
        print("\n" + "=" * 60)
        print("Transformer Autoencoder checkpoint found — skipping training.")
        print("  (run with --force-retrain to ignore checkpoints)")
        print("=" * 60)
        ckpt = torch.load(tf_ckpt, map_location=device)
        tf_model.load_state_dict(ckpt['model_state_dict'])
        tf_model.to(device)
        tf_history = {}
    else:
        print("\n" + "=" * 60)
        print("Training Transformer Autoencoder...")
        print("=" * 60)
        print(f"  Architecture: d_model={tf_cfg['d_model']}, heads={tf_cfg['nhead']}, "
              f"latent={tf_cfg['latent_dim']}, layers={tf_cfg['num_layers']}")
        tf_disc = _build_discriminator(config, processed_data)
        tf_trainer = Trainer(tf_model, config, device, discriminator=tf_disc)
        ensure_dir(tf_save_dir)
        tf_history = tf_trainer.train(
            train_loader, val_loader,
            epochs=config['training']['epochs'],
            save_dir=tf_save_dir
        )
        tf_trainer.load_checkpoint(tf_ckpt)

    print("\nComputing Transformer-AE training reconstruction errors (clean sessions only)...")
    tf_train_errors, _ = compute_drift_scores(
        tf_model, X_train_clean_tensor, device, lengths=L_train_clean_tensor
    )

    # ================================================================
    # 2c. Baseline models (session-level)
    # ================================================================
    print("\n" + "=" * 60)
    print("Training baseline models (session-level features)...")
    print("=" * 60)
    baselines = train_baseline_models(X_train_sess_clean, config)

    std_ae = baselines.get('StandardAutoencoder')
    std_ae_train_errors = None
    if std_ae is not None:
        # Use CLEAN sessions only so the z-score reference isn't inflated by anomalies
        X_sess_tensor = torch.FloatTensor(X_train_sess_clean[:, np.newaxis, :]).to(device)
        std_ae_train_errors = compute_reconstruction_error(std_ae, X_sess_tensor, device)

    # Persist baselines and train errors so evaluate can run standalone
    import pickle
    baselines_path = os.path.join(config['output']['models_dir'], 'baselines.pkl')
    with open(baselines_path, 'wb') as f:
        pickle.dump(baselines, f)
    print(f"Saved baselines to {baselines_path}")

    errors_path = os.path.join(config['output']['models_dir'], 'train_errors.npz')
    np.savez(errors_path,
             lstm_train_errors=lstm_train_errors,
             tf_train_errors=tf_train_errors,
             std_ae_train_errors=std_ae_train_errors if std_ae_train_errors is not None else np.array([]),
             sid_train_clean=sid_train_clean if sid_train_clean is not None else np.array([]))
    print(f"Saved train errors to {errors_path}")

    # Store everything in memory for the current run
    processed_data['lstm_model'] = lstm_model
    processed_data['tf_model'] = tf_model
    processed_data['baselines'] = baselines
    processed_data['lstm_train_errors'] = lstm_train_errors
    processed_data['tf_train_errors'] = tf_train_errors
    processed_data['std_ae_train_errors'] = std_ae_train_errors
    processed_data['sid_train_clean'] = sid_train_clean
    processed_data['lstm_history'] = lstm_history
    processed_data['tf_history'] = tf_history

    print("\nAll model training complete!")
    return processed_data


def _load_trained_models(config, processed_data, device):
    """Load trained models from disk when running evaluate/fairness/explain standalone."""
    import pickle

    print("Loading trained models from disk...")

    # LSTM Autoencoder
    lstm_model = LSTMAutoencoder(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    )
    lstm_ckpt = os.path.join(config['output']['models_dir'], 'lstm_ae', 'best_model.pth')
    ckpt = torch.load(lstm_ckpt, map_location=device)
    lstm_model.load_state_dict(ckpt['model_state_dict'])
    lstm_model.to(device)
    processed_data['lstm_model'] = lstm_model
    print(f"  Loaded LSTM-AE from {lstm_ckpt}")

    # Transformer Autoencoder
    tf_cfg = config['transformer']
    tf_model = TransformerAutoencoder(
        input_dim=tf_cfg['input_dim'],
        d_model=tf_cfg['d_model'],
        nhead=tf_cfg['nhead'],
        num_layers=tf_cfg['num_layers'],
        latent_dim=tf_cfg['latent_dim'],
        max_seq_len=tf_cfg['max_seq_len'],
        dropout=tf_cfg['dropout']
    )
    tf_ckpt = os.path.join(config['output']['models_dir'], 'transformer_ae', 'best_model.pth')
    ckpt = torch.load(tf_ckpt, map_location=device)
    tf_model.load_state_dict(ckpt['model_state_dict'])
    tf_model.to(device)
    processed_data['tf_model'] = tf_model
    print(f"  Loaded Transformer-AE from {tf_ckpt}")

    # Baselines
    baselines_path = os.path.join(config['output']['models_dir'], 'baselines.pkl')
    with open(baselines_path, 'rb') as f:
        processed_data['baselines'] = pickle.load(f)
    print(f"  Loaded baselines from {baselines_path}")

    # Train errors (needed for drift score normalization)
    errors_path = os.path.join(config['output']['models_dir'], 'train_errors.npz')
    errors = np.load(errors_path)
    processed_data['lstm_train_errors'] = errors['lstm_train_errors']
    processed_data['tf_train_errors'] = errors['tf_train_errors']
    std_ae_err = errors['std_ae_train_errors']
    processed_data['std_ae_train_errors'] = std_ae_err if len(std_ae_err) > 0 else None
    sid_clean = errors.get('sid_train_clean', np.array([]))
    processed_data['sid_train_clean'] = sid_clean if len(sid_clean) > 0 else None
    print(f"  Loaded train errors from {errors_path}")

    return processed_data


def _eval_sequence_model(name, model, train_errors, X_val_seq, X_test_seq,
                         L_val, L_test, y_val, y_test, device,
                         all_results, all_scores, all_predictions,
                         sid_train_clean=None, sid_val=None, sid_test=None):
    """Evaluate a sequence autoencoder (LSTM or Transformer) with blended drift scoring."""
    print(f"\nEvaluating {name}...")
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    L_val_tensor = torch.LongTensor(L_val).to(device)

    # Use blended (personalized) scoring if student IDs are available
    if sid_train_clean is not None and sid_val is not None:
        _, val_scores = compute_blended_drift_scores(
            model, X_val_tensor, device, train_errors,
            sid_train_clean, sid_val, L_val_tensor
        )
    else:
        _, val_scores = compute_drift_scores(model, X_val_tensor, device, train_errors, L_val_tensor)

    threshold = select_optimal_threshold(y_val, val_scores, method='f1_weighted')
    print(f"  {name} threshold: {threshold:.4f}")

    # Evaluate on test set with blended scoring
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    L_test_tensor = torch.LongTensor(L_test).to(device)

    if sid_train_clean is not None and sid_test is not None:
        _, test_scores = compute_blended_drift_scores(
            model, X_test_tensor, device, train_errors,
            sid_train_clean, sid_test, L_test_tensor
        )
    else:
        _, test_scores = compute_drift_scores(model, X_test_tensor, device, train_errors, L_test_tensor)

    preds = (test_scores > threshold).astype(int)
    from src.evaluate import compute_classification_metrics
    metrics = compute_classification_metrics(y_test, preds, test_scores)
    all_results[name] = metrics
    all_scores[name] = test_scores
    all_predictions[name] = preds

    print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} | "
          f"F1: {metrics['f1']:.4f} | "
          f"Precision: {metrics['precision']:.4f} | "
          f"Recall: {metrics['recall']:.4f}")
    return threshold


def evaluate_models(config, processed_data):
    """Step 3: Evaluate all models (6 total) with PER-MODEL threshold selection."""
    print("\n" + "="*80)
    print("STEP 3: MODEL EVALUATION")
    print("="*80 + "\n")

    device = get_device(config['training']['device'])

    # Load from disk if running evaluate standalone (after a separate train run)
    if 'lstm_model' not in processed_data:
        processed_data = _load_trained_models(config, processed_data, device)

    X_val_seq = processed_data['X_val_seq']
    X_test_seq = processed_data['X_test_seq']
    L_val = processed_data['L_val']
    L_test = processed_data['L_test']
    X_val_sess = processed_data['X_val_sess']
    X_test_sess = processed_data['X_test_sess']
    X_test_sess_raw = processed_data.get('X_test_sess_raw')
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    baselines = processed_data['baselines']
    std_ae_train_errors = processed_data.get('std_ae_train_errors')
    sid_train_clean = processed_data.get('sid_train_clean')
    sid_val = processed_data.get('sid_val')
    sid_test = processed_data.get('sid_test')

    all_results = {}
    all_scores = {}
    all_predictions = {}

    # ---- 1. LSTM Autoencoder ----
    lstm_threshold = _eval_sequence_model(
        'LSTM-AE (Ours)', processed_data['lstm_model'],
        processed_data['lstm_train_errors'],
        X_val_seq, X_test_seq, L_val, L_test, y_val, y_test, device,
        all_results, all_scores, all_predictions,
        sid_train_clean=sid_train_clean, sid_val=sid_val, sid_test=sid_test
    )

    # ---- 2. Transformer Autoencoder ----
    _eval_sequence_model(
        'Transformer-AE', processed_data['tf_model'],
        processed_data['tf_train_errors'],
        X_val_seq, X_test_seq, L_val, L_test, y_val, y_test, device,
        all_results, all_scores, all_predictions,
        sid_train_clean=sid_train_clean, sid_val=sid_val, sid_test=sid_test
    )

    # ---- 3. Baseline models (each gets its OWN threshold) ----
    for baseline_name, baseline_model in baselines.items():
        print(f"\nEvaluating {baseline_name}...")

        if baseline_name == 'StandardAutoencoder':
            X_val_ae = X_val_sess[:, np.newaxis, :]
            X_test_ae = X_test_sess[:, np.newaxis, :]
            X_val_t = torch.FloatTensor(X_val_ae).to(device)
            _, val_scores_ae = compute_drift_scores(baseline_model, X_val_t, device, std_ae_train_errors)
            ae_threshold = select_optimal_threshold(y_val, val_scores_ae, method='f1_weighted')
            print(f"  Threshold: {ae_threshold:.4f}")
            metrics, scores, preds = evaluate_model(
                baseline_model, X_test_ae, y_test, ae_threshold, device,
                model_type='standard_ae', train_errors=std_ae_train_errors
            )

        elif baseline_name == 'RuleBased':
            # IMPORTANT: pass RAW (unnormalized) features to RuleBased
            # because its thresholds are in original units (seconds, counts)
            rb_features = X_test_sess_raw if X_test_sess_raw is not None else X_test_sess
            metrics, scores, preds = evaluate_model(
                baseline_model, rb_features, y_test, 0, device,
                model_type='rule_based'
            )

        else:
            val_scores_bl = baseline_model.score_samples(X_val_sess)
            bl_threshold = select_optimal_threshold(y_val, val_scores_bl, method='f1_weighted')
            print(f"  Threshold: {bl_threshold:.4f}")
            metrics, scores, preds = evaluate_model(
                baseline_model, X_test_sess, y_test, bl_threshold, device,
                model_type='sklearn'
            )

        all_results[baseline_name] = metrics
        all_scores[baseline_name] = scores
        all_predictions[baseline_name] = preds

        print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f}")

    # Compare all 6 models
    compare_models(all_results)

    ensure_dir(config['output']['metrics_dir'])
    results_path = os.path.join(config['output']['metrics_dir'], 'evaluation_results.yaml')
    save_metrics(all_results, results_path)

    processed_data['evaluation_results'] = all_results
    processed_data['test_scores'] = all_scores
    processed_data['test_predictions'] = all_predictions
    processed_data['threshold'] = lstm_threshold

    print("\nEvaluation complete!")
    return processed_data


def analyze_fairness(config, processed_data):
    """Step 4: Fairness analysis and bias mitigation."""
    print("\n" + "="*80)
    print("STEP 4: FAIRNESS ANALYSIS")
    print("="*80 + "\n")

    y_test = processed_data['y_test']
    demo_test = processed_data['demo_test']

    # Use LSTM-AE scores for fairness (it's the primary personalized model)
    primary_key = 'LSTM-AE (Ours)'
    lstm_scores = processed_data['test_scores'][primary_key]
    lstm_preds = processed_data['test_predictions'][primary_key]
    threshold = processed_data['threshold']

    # Handle demographics (may be stored as numpy array of dicts)
    if isinstance(demo_test, np.ndarray):
        demo_test = demo_test.tolist()

    analyzer = FairnessAnalyzer(
        sensitive_attributes=config['fairness']['sensitive_attributes'],
        config=config
    )

    # Analyze BEFORE calibration
    print("Analyzing fairness with global threshold...")
    fairness_before = analyzer.analyze_fairness(lstm_preds, y_test, demo_test)
    analyzer.print_fairness_report(fairness_before)

    # Calibrate thresholds per attribute using α grid search
    print("\nCalibrating group-specific thresholds (α grid search)...")
    calibrated_predictions = {}
    optimal_alphas = {}
    for attribute in config['fairness']['sensitive_attributes']:
        best_alpha, group_thresholds = analyzer.alpha_grid_search(
            lstm_scores, y_test, demo_test, threshold, attribute
        )
        optimal_alphas[attribute] = best_alpha
        fair_preds = analyzer.apply_fair_predictions(
            lstm_scores, demo_test, attribute, group_thresholds
        )
        calibrated_predictions[attribute] = fair_preds
        print(f"  {attribute}: optimal α = {best_alpha:.2f}")

    # Analyze AFTER calibration — each attribute uses its OWN calibrated predictions.
    # SAFEGUARD: only keep calibrated predictions if they actually improve the
    # fairness metric for that attribute.  If calibration makes things worse
    # (e.g. by adjusting thresholds based on noisy per-group means), we fall
    # back to the global-threshold predictions for that attribute.
    print("\nAnalyzing fairness with calibrated thresholds...")
    fairness_after = {}
    for attribute in config['fairness']['sensitive_attributes']:
        preds_for_attr = calibrated_predictions[attribute]
        attr_metrics = analyzer.analyze_fairness(preds_for_attr, y_test, demo_test)
        calibrated_eo = attr_metrics[attribute]['equalized_odds']['max_difference']
        calibrated_dp = attr_metrics[attribute]['demographic_parity']['ratio']
        before_eo = fairness_before[attribute]['equalized_odds']['max_difference']
        before_dp = fairness_before[attribute]['demographic_parity']['ratio']

        # Keep calibration only if BOTH metrics improve (or at least don't worsen)
        eo_improved = calibrated_eo <= before_eo
        dp_improved = calibrated_dp >= before_dp
        if eo_improved and dp_improved:
            fairness_after[attribute] = attr_metrics[attribute]
        else:
            print(f"  WARNING: Calibration worsened fairness for '{attribute}' "
                  f"(EO: {before_eo:.3f}->{calibrated_eo:.3f}, "
                  f"DP ratio: {before_dp:.3f}->{calibrated_dp:.3f}). "
                  f"Keeping global threshold.")
            fairness_after[attribute] = fairness_before[attribute]

    analyzer.print_fairness_report(fairness_after)
    compare_fairness_before_after(fairness_before, fairness_after)

    fairness_results = {
        'before_calibration': fairness_before,
        'after_calibration': fairness_after
    }
    fairness_path = os.path.join(config['output']['metrics_dir'], 'fairness_results.yaml')
    save_metrics(fairness_results, fairness_path)

    processed_data['fairness_results'] = fairness_results
    print("\nFairness analysis complete!")
    return processed_data


def explain_predictions(config, processed_data):
    """Step 5: Generate SHAP explanations."""
    print("\n" + "="*80)
    print("STEP 5: EXPLAINABILITY (SHAP)")
    print("="*80 + "\n")

    device = get_device(config['training']['device'])

    if 'lstm_model' not in processed_data:
        processed_data = _load_trained_models(config, processed_data, device)

    # Use sequential (question-level) data so SHAP explains the model on
    # the same data shape it was trained on (not the invalid length-1 hack).
    X_train_seq = processed_data['X_train_seq']
    X_test_seq = processed_data['X_test_seq']
    L_train = processed_data['L_train']
    L_test = processed_data['L_test']
    model = processed_data['lstm_model']
    primary_key = 'LSTM-AE (Ours)'
    lstm_scores = processed_data['test_scores'][primary_key]
    lstm_preds = processed_data['test_predictions'][primary_key]
    q_feature_names = processed_data['question_feature_names']

    if isinstance(q_feature_names, np.ndarray):
        q_feature_names = q_feature_names.tolist()

    explainer = SequentialSHAPExplainer(model, q_feature_names, config, device)
    explainer.init_explainer(X_train_seq, L_train)

    explain_dir = os.path.join(config['output']['plots_dir'], 'explanations')
    ensure_dir(explain_dir)

    generate_explanation_report(
        explainer, X_test_seq, lstm_scores, lstm_preds, explain_dir,
        lengths=L_test
    )

    # Move model back to original device after SHAP (which runs on CPU)
    model.to(device)

    print("\nExplanations generated!")
    return processed_data


def main():
    parser = argparse.ArgumentParser(description='Behavioral Drift Detection Pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['preprocess', 'train', 'evaluate', 'fairness', 'explain', 'all'],
                       help='Pipeline mode')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Ignore existing checkpoints and retrain models from scratch')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("BEHAVIORAL DRIFT DETECTION FOR ONLINE EXAM INTEGRITY")
    print("="*80)

    config = load_config(args.config)
    config['_force_retrain'] = args.force_retrain
    print(f"\nConfiguration loaded from {args.config}")

    set_seed(config['training']['seed'])
    print(f"Random seed set to {config['training']['seed']}")

    for dir_key in ['models_dir', 'plots_dir', 'metrics_dir']:
        ensure_dir(config['output'][dir_key])

    processed_data_path = os.path.join(config['data']['processed_path'], 'processed_data.npz')

    if args.mode in ['preprocess', 'all']:
        processed_data = preprocess_data(config)
    else:
        print(f"\nLoading preprocessed data from {processed_data_path}...")
        processed_data = load_processed_data(processed_data_path)

    if args.mode in ['train', 'all']:
        processed_data = train_models(config, processed_data)

    if args.mode in ['evaluate', 'all']:
        processed_data = evaluate_models(config, processed_data)

    if args.mode in ['fairness', 'all']:
        processed_data = analyze_fairness(config, processed_data)

    if args.mode in ['explain', 'all']:
        processed_data = explain_predictions(config, processed_data)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Models: {config['output']['models_dir']}")
    print(f"  Plots: {config['output']['plots_dir']}")
    print(f"  Metrics: {config['output']['metrics_dir']}")
    print("\n")


if __name__ == '__main__':
    main()
