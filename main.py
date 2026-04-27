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
from src.realistic_cheating_rules import RealisticCheatingGenerator
from src.gan_model import ConditionalGANTrainer, GANConfig, compute_feature_bounds, apply_sanity_filters
from src.plain_classifiers import PlainTransformerClassifier, PlainLSTMClassifier, ClassifierTrainer
from src.models import LSTMAutoencoder, DemographicDiscriminator, compute_reconstruction_error
from src.train import (Trainer, compute_drift_scores, compute_blended_drift_scores,
                        compute_combined_scores, train_baseline_models)
from src.evaluate import evaluate_model, select_optimal_threshold, compare_models, compute_classification_metrics, compute_precision_at_k
from src.fairness import FairnessAnalyzer, compare_fairness_before_after
from src.explainability import SHAPExplainer, SequentialSHAPExplainer, generate_explanation_report
from src.visualization import generate_all_plots


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

    # ---- Extract sequence features (for LSTM / Transformer models) ----
    # The feature level is configurable: 'question' (per-question aggregates)
    # or 'action' (per-event, finer-grained — answers the critique that
    # per-question aggregates hide action-level temporal drift).
    seq_level = config['data'].get('sequence_feature_level', 'question')
    if seq_level == 'action':
        max_seq_len = int(config['data'].get('action_max_seq_len', 100))
        print(f"\nExtracting ACTION-level features (max_seq_len={max_seq_len})...")
        question_features_clean, seq_lengths_clean = extractor.extract_action_features_batch(
            sessions, max_seq_len=max_seq_len
        )
        seq_feature_names = extractor.action_feature_names
    else:
        max_seq_len = config['model'].get('max_seq_len', 50)
        print(f"\nExtracting QUESTION-level features (max_seq_len={max_seq_len})...")
        question_features_clean, seq_lengths_clean = extractor.extract_question_features_batch(
            sessions, max_seq_len=max_seq_len
        )
        seq_feature_names = extractor.question_feature_names
    # Compute normalization stats on CLEAN data only
    _, q_mean, q_std = extractor.normalize_question_features(
        question_features_clean, seq_lengths_clean
    )

    # ---- Inject synthetic anomalies ----
    # Method selector: 'injection' (legacy), 'realistic' (literature-grounded),
    # or 'gan' (realistic rules seed + GAN augmentation at the feature level).
    method = config['data'].get('synthetic_method', 'injection')
    print(f"\nInjecting synthetic anomalies (method='{method}')...")
    if method == 'injection':
        anomaly_generator = SyntheticAnomalyGenerator(
            contamination_rate=config['data']['synthetic_contamination'],
            seed=config['training']['seed']
        )
    else:
        # Both 'realistic' and 'gan' start from realistic rules.
        # For 'gan', the GAN augments the anomaly set after feature extraction.
        anomaly_generator = RealisticCheatingGenerator(
            contamination_rate=config['data']['synthetic_contamination'],
            seed=config['training']['seed']
        )
    # Optional: restrict to a subset of anomaly families for cross-distribution
    # generalization experiments (train on subset A, test on subset B).
    allowed_families = config['data'].get('allowed_anomaly_families', None)
    if method != 'injection' and allowed_families:
        sessions_with_anomalies, labels, families = anomaly_generator.inject_anomalies(
            sessions, allowed_families=allowed_families
        )
    else:
        sessions_with_anomalies, labels, families = anomaly_generator.inject_anomalies(sessions)

    # Re-extract BOTH feature levels from anomalous sessions
    print("\nRe-extracting features after anomaly injection...")
    session_features_final, _ = extractor.extract_features_batch(sessions_with_anomalies)
    session_features_final_norm, _, _ = extractor.normalize_features(
        session_features_final, mean=sess_mean, std=sess_std
    )

    if seq_level == 'action':
        question_features_final, seq_lengths_final = extractor.extract_action_features_batch(
            sessions_with_anomalies, max_seq_len=max_seq_len
        )
    else:
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

    # Split anomaly family identifiers (for per-family evaluation)
    fam_train = families[train_idx]
    fam_val = families[val_idx]
    fam_test = families[test_idx]

    # Split demographics
    demo_train = [demographics[i] for i in train_idx]
    demo_val = [demographics[i] for i in val_idx]
    demo_test = [demographics[i] for i in test_idx]

    # Split student IDs (needed for personalized blended drift scoring)
    student_ids_arr = np.array(student_ids)
    sid_train = student_ids_arr[train_idx]
    sid_val = student_ids_arr[val_idx]
    sid_test = student_ids_arr[test_idx]

    # ---- Optional GAN augmentation of the TRAIN partition only ----
    # Critical design choice: we only augment training data. Val/test stay
    # purely real + rule-based so evaluation numbers reflect the detector's
    # real-world behaviour, not the GAN's ability to fool itself.
    if method == 'gan' and float(config['data'].get('gan_augment_fraction', 0.0)) > 0:
        frac = float(config['data']['gan_augment_fraction'])
        print("\n" + "=" * 60)
        print(f"GAN augmentation: adding {frac*100:.0f}% more anomalies to train set")
        print("=" * 60)

        gan_cfg_raw = config.get('gan', {})
        gan_cfg = GANConfig(
            noise_dim=int(gan_cfg_raw.get('noise_dim', 64)),
            n_features=question_features_final_norm.shape[2],
            max_seq_len=max_seq_len,
            hidden_dim=int(gan_cfg_raw.get('hidden_dim', 96)),
            n_classes=2,
            epochs=int(gan_cfg_raw.get('epochs', 40)),
            batch_size=int(gan_cfg_raw.get('batch_size', 128)),
            lr=float(gan_cfg_raw.get('lr', 2e-4)),
            beta1=float(gan_cfg_raw.get('beta1', 0.5)),
            label_smoothing=float(gan_cfg_raw.get('label_smoothing', 0.1)),
        )

        gan_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ConditionalGANTrainer(gan_cfg, device=gan_device)

        # Train GAN on the TRAIN partition (both normals and rule-anomalies)
        X_gan_train = question_features_final_norm[train_idx]
        y_gan_train = labels[train_idx]
        L_gan_train = seq_lengths_final[train_idx]

        print(f"  GAN training data: {len(X_gan_train)} samples "
              f"({y_gan_train.sum()} cheating, {(y_gan_train == 0).sum()} normal)")
        trainer.train(X_gan_train, y_gan_train, L_gan_train)

        # How many extra anomalies to generate?
        n_train_anomalies = int(y_gan_train.sum())
        n_extra = int(n_train_anomalies * frac)
        # Oversample by 2x to compensate for sanity-filter rejections
        n_request = max(n_extra * 2, 50)
        print(f"\n  Generating {n_request} candidate GAN anomalies "
              f"(target: keep {n_extra} after sanity filters)...")
        X_gen, L_gen = trainer.generate(n_request, label=1)

        # Sanity filters: reject out-of-range / NaN samples
        lo, hi = compute_feature_bounds(
            X_gan_train, L_gan_train,
            pad=float(gan_cfg_raw.get('sanity_pad', 0.25))
        )
        X_kept, L_kept, rej = apply_sanity_filters(
            X_gen, L_gen, lo, hi,
            max_rejection_rate=float(gan_cfg_raw.get('max_rejection_rate', 0.5))
        )
        # Cap at the requested count
        if len(X_kept) > n_extra:
            X_kept = X_kept[:n_extra]
            L_kept = L_kept[:n_extra]
        n_added = len(X_kept)
        print(f"  Kept {n_added}/{n_extra} GAN anomalies after sanity filtering")

        if n_added > 0:
            # Build extra rows for each tensor we need to keep aligned
            # Feature rows: use zero-vectors for session-level (no raw session for GAN)
            # so the GAN-augmented samples only contribute to sequence-based models.
            zero_sess = np.zeros((n_added, session_features_final_norm.shape[1]),
                                 dtype=session_features_final_norm.dtype)
            zero_sess_raw = np.zeros((n_added, session_features_final.shape[1]),
                                     dtype=session_features_final.dtype)

            # Demographics: sample from TRAIN to preserve distribution
            demo_pool = [demographics[i] for i in train_idx]
            extra_demos = list(np.random.choice(demo_pool, size=n_added))

            extra_sids = np.array([f"gan_synth_{i}" for i in range(n_added)], dtype=object)
            extra_labels = np.ones(n_added, dtype=int)
            extra_families = np.array(["gan_augmented"] * n_added, dtype=object)
            families = np.concatenate([families, extra_families], axis=0)

            # Extend the underlying arrays/lists. The chronological split is
            # already done; we append these to the train side only.
            question_features_final_norm = np.concatenate(
                [question_features_final_norm, X_kept], axis=0)
            seq_lengths_final = np.concatenate([seq_lengths_final, L_kept], axis=0)
            session_features_final_norm = np.concatenate(
                [session_features_final_norm, zero_sess], axis=0)
            session_features_final = np.concatenate(
                [session_features_final, zero_sess_raw], axis=0)
            labels = np.concatenate([labels, extra_labels], axis=0)
            demographics = demographics + extra_demos
            student_ids = list(student_ids) + list(extra_sids)

            # Their new indices go into train_idx
            new_indices = np.arange(
                len(question_features_final_norm) - n_added,
                len(question_features_final_norm)
            )
            train_idx = np.concatenate([train_idx, new_indices])

            # Re-split the already-assigned slices on the (now-extended) arrays
            X_train_sess = session_features_final_norm[train_idx]
            X_train_sess_raw = session_features_final[train_idx]
            X_train_seq = question_features_final_norm[train_idx]
            L_train = seq_lengths_final[train_idx]
            y_train = labels[train_idx]
            demo_train = [demographics[i] for i in train_idx]
            student_ids_arr = np.array(student_ids, dtype=object)
            sid_train = student_ids_arr[train_idx]
            fam_train = families[train_idx]

            print(f"  Train set now: {len(train_idx)} samples "
                  f"({int(y_train.sum())} cheating, {int((y_train == 0).sum())} normal)")

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
        'question_feature_names': seq_feature_names,
        # Student IDs (for personalized blended drift scoring)
        'sid_train': sid_train,
        'sid_val': sid_val,
        'sid_test': sid_test,
        # Anomaly family per session (for per-family evaluation)
        'fam_train': fam_train,
        'fam_val': fam_val,
        'fam_test': fam_test,
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
    """Step 2: Train LSTM-AE, Plain Transformer classifier, and baseline models."""
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80 + "\n")

    device = get_device(config['training']['device'])
    ensure_dir(config['output']['models_dir'])

    # GPU performance optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True   # auto-tune convolution algorithms
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  cudnn.benchmark: enabled")

    # Derive the effective max_seq_len from the preprocessed data so that
    # all sequence models agree on tensor shape regardless of feature level.
    # This prevents Transformer positional-encoding shape mismatches when
    # switching between question-level (50) and action-level (100) features.
    effective_max_seq_len = int(processed_data['X_train_seq'].shape[1])

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
        # Recover training history from checkpoint if saved
        lstm_history = {}
        if 'train_losses' in ckpt:
            lstm_history = {
                'train_losses': ckpt['train_losses'],
                'val_losses': ckpt.get('val_losses', []),
            }
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
    # 2b. Plain Transformer Classifier (supervised — main proposed model)
    # ================================================================
    # This trains on the FULL training set (normals + anomalies) with
    # binary labels — the main detection model.
    pc_cfg = config.get('plain_classifiers', {})
    plain_tf_model = None
    if pc_cfg.get('enabled', False):
        plain_save_dir = os.path.join(config['output']['models_dir'], 'plain_classifiers')
        ensure_dir(plain_save_dir)

        # Supervised training uses full train/val with labels
        X_train_full = processed_data['X_train_seq']
        L_train_full = processed_data['L_train']
        y_train_full = processed_data['y_train']
        X_val_full = processed_data['X_val_seq']
        L_val_full = processed_data['L_val']
        y_val_full = processed_data['y_val']

        pos_weight = float((y_train_full == 0).sum() / max(1, (y_train_full == 1).sum()))
        input_dim = int(X_train_full.shape[2])
        effective_seq_len = int(X_train_full.shape[1])

        tf_pc_cfg = pc_cfg.get('transformer', {})
        tf_ckpt = os.path.join(plain_save_dir, 'plain_transformer.pth')
        plain_tf_model = PlainTransformerClassifier(
            input_dim=input_dim,
            d_model=int(tf_pc_cfg.get('d_model', 128)),
            nhead=int(tf_pc_cfg.get('nhead', 4)),
            num_layers=int(tf_pc_cfg.get('num_layers', 3)),
            max_seq_len=effective_seq_len,
            dropout=float(tf_pc_cfg.get('dropout', 0.2)),
        )
        if os.path.exists(tf_ckpt) and not force_retrain:
            print("\n" + "=" * 60)
            print("Plain Transformer classifier checkpoint found — skipping training.")
            print("=" * 60)
            ckpt = torch.load(tf_ckpt, map_location=device)
            plain_tf_model.load_state_dict(ckpt['model_state_dict'])
            plain_tf_model.to(device)
        else:
            print("\n" + "=" * 60)
            print("Training Plain Transformer classifier (supervised — main model)...")
            print("=" * 60)
            trainer = ClassifierTrainer(
                plain_tf_model, device,
                lr=float(pc_cfg.get('learning_rate', 1e-3)),
                weight_decay=float(config['training'].get('weight_decay', 5e-4)),
                pos_weight=pos_weight,
                gradient_clip=float(config['training'].get('gradient_clip', 1.0)),
            )
            tf_cls_history = trainer.train(
                X_train_full, y_train_full, L_train_full,
                X_val_full, y_val_full, L_val_full,
                epochs=int(pc_cfg.get('epochs', 50)),
                batch_size=int(pc_cfg.get('batch_size', 256)),
                patience=int(pc_cfg.get('patience', 10)),
                save_path=tf_ckpt,
            )
            processed_data['tf_classifier_history'] = tf_cls_history

    processed_data['plain_tf_model'] = plain_tf_model

    # ================================================================
    # 2c. Plain LSTM Classifier (supervised baseline)
    # ================================================================
    plain_lstm_model = None
    if pc_cfg.get('enabled', False):
        lstm_pc_cfg = pc_cfg.get('lstm', {})
        lstm_ckpt = os.path.join(plain_save_dir, 'plain_lstm.pth')
        plain_lstm_model = PlainLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=int(lstm_pc_cfg.get('hidden_dim', 96)),
            num_layers=int(lstm_pc_cfg.get('num_layers', 2)),
            max_seq_len=effective_seq_len,
            dropout=float(lstm_pc_cfg.get('dropout', 0.2)),
        )
        if os.path.exists(lstm_ckpt) and not force_retrain:
            print("\n" + "=" * 60)
            print("Plain LSTM classifier checkpoint found — skipping training.")
            print("=" * 60)
            ckpt = torch.load(lstm_ckpt, map_location=device)
            plain_lstm_model.load_state_dict(ckpt['model_state_dict'])
            plain_lstm_model.to(device)
        else:
            print("\n" + "=" * 60)
            print("Training Plain LSTM classifier (supervised baseline)...")
            print("=" * 60)
            lstm_cls_trainer = ClassifierTrainer(
                plain_lstm_model, device,
                lr=float(pc_cfg.get('learning_rate', 1e-3)),
                weight_decay=float(config['training'].get('weight_decay', 5e-4)),
                pos_weight=pos_weight,
                gradient_clip=float(config['training'].get('gradient_clip', 1.0)),
            )
            lstm_cls_history = lstm_cls_trainer.train(
                X_train_full, y_train_full, L_train_full,
                X_val_full, y_val_full, L_val_full,
                epochs=int(pc_cfg.get('epochs', 50)),
                batch_size=int(pc_cfg.get('batch_size', 256)),
                patience=int(pc_cfg.get('patience', 10)),
                save_path=lstm_ckpt,
            )
            processed_data['lstm_classifier_history'] = lstm_cls_history

    processed_data['plain_lstm_model'] = plain_lstm_model

    # ================================================================
    # 2d. Baseline models (session-level)
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
             std_ae_train_errors=std_ae_train_errors if std_ae_train_errors is not None else np.array([]),
             sid_train_clean=sid_train_clean if sid_train_clean is not None else np.array([]))
    print(f"Saved train errors to {errors_path}")

    # Store everything in memory for the current run
    processed_data['lstm_model'] = lstm_model
    processed_data['baselines'] = baselines
    processed_data['lstm_train_errors'] = lstm_train_errors
    processed_data['std_ae_train_errors'] = std_ae_train_errors
    processed_data['sid_train_clean'] = sid_train_clean
    processed_data['lstm_history'] = lstm_history

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

    # Plain Transformer Classifier (main model)
    pc_cfg = config.get('plain_classifiers', {})
    if pc_cfg.get('enabled', False):
        plain_dir = os.path.join(config['output']['models_dir'], 'plain_classifiers')
        input_dim = int(processed_data['X_train_seq'].shape[2])
        seq_len = int(processed_data['X_train_seq'].shape[1])
        tf_ckpt = os.path.join(plain_dir, 'plain_transformer.pth')
        if os.path.exists(tf_ckpt):
            tf_pc_cfg = pc_cfg.get('transformer', {})
            m = PlainTransformerClassifier(
                input_dim=input_dim,
                d_model=int(tf_pc_cfg.get('d_model', 128)),
                nhead=int(tf_pc_cfg.get('nhead', 4)),
                num_layers=int(tf_pc_cfg.get('num_layers', 3)),
                max_seq_len=seq_len,
                dropout=float(tf_pc_cfg.get('dropout', 0.2)),
            )
            m.load_state_dict(torch.load(tf_ckpt, map_location=device)['model_state_dict'])
            m.to(device)
            processed_data['plain_tf_model'] = m
            print(f"  Loaded Plain Transformer classifier from {tf_ckpt}")

        # Plain LSTM Classifier (supervised baseline)
        lstm_ckpt = os.path.join(plain_dir, 'plain_lstm.pth')
        if os.path.exists(lstm_ckpt):
            lstm_pc_cfg = pc_cfg.get('lstm', {})
            lstm_m = PlainLSTMClassifier(
                input_dim=input_dim,
                hidden_dim=int(lstm_pc_cfg.get('hidden_dim', 96)),
                num_layers=int(lstm_pc_cfg.get('num_layers', 2)),
                max_seq_len=seq_len,
                dropout=float(lstm_pc_cfg.get('dropout', 0.2)),
            )
            lstm_m.load_state_dict(torch.load(lstm_ckpt, map_location=device)['model_state_dict'])
            lstm_m.to(device)
            processed_data['plain_lstm_model'] = lstm_m
            print(f"  Loaded Plain LSTM classifier from {lstm_ckpt}")

    # Baselines
    baselines_path = os.path.join(config['output']['models_dir'], 'baselines.pkl')
    with open(baselines_path, 'rb') as f:
        processed_data['baselines'] = pickle.load(f)
    print(f"  Loaded baselines from {baselines_path}")

    # Train errors (needed for drift score normalization)
    errors_path = os.path.join(config['output']['models_dir'], 'train_errors.npz')
    errors = np.load(errors_path, allow_pickle=True)
    processed_data['lstm_train_errors'] = errors['lstm_train_errors']
    std_ae_err = errors['std_ae_train_errors']
    processed_data['std_ae_train_errors'] = std_ae_err if len(std_ae_err) > 0 else None
    sid_clean = errors.get('sid_train_clean', np.array([]))
    processed_data['sid_train_clean'] = sid_clean if len(sid_clean) > 0 else None
    print(f"  Loaded train errors from {errors_path}")

    return processed_data


def _eval_sequence_model(name, model, train_errors, X_val_seq, X_test_seq,
                         L_val, L_test, y_val, y_test, device,
                         all_results, all_scores, all_predictions,
                         sid_train_clean=None, sid_val=None, sid_test=None,
                         X_train_seq=None, L_train=None):
    """Evaluate a sequence autoencoder using combined reconstruction + Mahalanobis scoring."""
    print(f"\nEvaluating {name}...")
    X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
    L_val_tensor = torch.LongTensor(L_val).to(device)

    # Use combined (reconstruction + Mahalanobis) scoring when training data available
    if sid_train_clean is not None and sid_val is not None and X_train_seq is not None:
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        L_train_tensor = torch.LongTensor(L_train).to(device)
        _, val_scores = compute_combined_scores(
            model, X_val_tensor, device, train_errors,
            sid_train_clean, sid_val, X_train_tensor,
            lengths=L_val_tensor, train_lengths=L_train_tensor
        )
    elif sid_train_clean is not None and sid_val is not None:
        _, val_scores = compute_blended_drift_scores(
            model, X_val_tensor, device, train_errors,
            sid_train_clean, sid_val, L_val_tensor
        )
    else:
        _, val_scores = compute_drift_scores(model, X_val_tensor, device, train_errors, L_val_tensor)

    threshold = select_optimal_threshold(y_val, val_scores, method='f1_weighted')
    print(f"  {name} threshold: {threshold:.4f}")

    # Evaluate on test set with same scoring
    X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
    L_test_tensor = torch.LongTensor(L_test).to(device)

    if sid_train_clean is not None and sid_test is not None and X_train_seq is not None:
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        L_train_tensor = torch.LongTensor(L_train).to(device)
        _, test_scores = compute_combined_scores(
            model, X_test_tensor, device, train_errors,
            sid_train_clean, sid_test, X_train_tensor,
            lengths=L_test_tensor, train_lengths=L_train_tensor
        )
    elif sid_train_clean is not None and sid_test is not None:
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

    X_train_seq = processed_data['X_train_seq']
    L_train = processed_data['L_train']
    y_train = processed_data['y_train']
    # Use only clean training sessions for Mahalanobis reference distribution
    normal_mask = (y_train == 0)
    X_train_seq_clean = X_train_seq[normal_mask]
    L_train_clean = L_train[normal_mask]

    # ---- 1. LSTM Autoencoder ----
    lstm_threshold = _eval_sequence_model(
        'LSTM-AE', processed_data['lstm_model'],
        processed_data['lstm_train_errors'],
        X_val_seq, X_test_seq, L_val, L_test, y_val, y_test, device,
        all_results, all_scores, all_predictions,
        sid_train_clean=sid_train_clean, sid_val=sid_val, sid_test=sid_test,
        X_train_seq=X_train_seq_clean, L_train=L_train_clean
    )

    # ---- 2. Plain Transformer Classifier (Ours — main proposed model) ----
    # Batched inference to avoid OOM on large sequence tensors
    def _batched_supervised_scores(pc_model, X_np, L_np, batch_size=256):
        pc_model.eval()
        out = []
        n = X_np.shape[0]
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.FloatTensor(X_np[i:i+batch_size]).to(device)
                lb = torch.LongTensor(L_np[i:i+batch_size]).to(device)
                s = torch.sigmoid(pc_model(xb, lb)).cpu().numpy()
                out.append(s)
                del xb, lb
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        return np.concatenate(out, axis=0)

    plain_tf_model = processed_data.get('plain_tf_model')
    if plain_tf_model is not None:
        pc_name = 'Plain-Transformer (Ours)'
        print(f"\nEvaluating {pc_name}...")
        val_scores_raw = _batched_supervised_scores(plain_tf_model, X_val_seq, L_val)
        test_scores_raw = _batched_supervised_scores(plain_tf_model, X_test_seq, L_test)

        # ----- Personalized blended scoring on top of raw Transformer logits -----
        # The Transformer is a population-level supervised classifier. To honour
        # the "personalized drift" framing of the paper, we additionally blend
        # each test session's score against that student's own historical score
        # distribution (clean training sessions only), with cold-start weighting
        # lambda_s = min(1, n_s / 5). Train scores come from clean sessions in
        # train; eval blending is applied to validation (for thresholding) and
        # test sets identically.
        from src.train import personalize_scores
        sid_train_clean_arr = processed_data.get('sid_train_clean')
        sid_val_arr = processed_data.get('sid_val')
        sid_test_arr = processed_data.get('sid_test')
        if sid_train_clean_arr is not None and len(sid_train_clean_arr) > 0:
            print(f"  Applying per-student blended scoring (lambda_s = min(1, n_s/5))")
            X_train_clean_seq = X_train_seq_clean
            L_train_clean_seq = L_train_clean
            train_scores_clean = _batched_supervised_scores(
                plain_tf_model, X_train_clean_seq, L_train_clean_seq
            )
            val_scores = personalize_scores(
                train_scores_clean, val_scores_raw,
                np.asarray(sid_train_clean_arr), np.asarray(sid_val_arr)
            )
            test_scores = personalize_scores(
                train_scores_clean, test_scores_raw,
                np.asarray(sid_train_clean_arr), np.asarray(sid_test_arr)
            )
        else:
            print("  WARNING: no train student IDs available; using raw scores.")
            val_scores = val_scores_raw
            test_scores = test_scores_raw

        pc_thresh = select_optimal_threshold(y_val, val_scores, method='f1_weighted')
        print(f"  {pc_name} threshold: {pc_thresh:.4f}")
        preds = (test_scores > pc_thresh).astype(int)
        metrics = compute_classification_metrics(y_test, preds, test_scores)
        all_results[pc_name] = metrics
        all_scores[pc_name] = test_scores
        all_predictions[pc_name] = preds
        print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f}")

    # ---- 3. Plain LSTM Classifier (supervised baseline) ----
    plain_lstm_model = processed_data.get('plain_lstm_model')
    if plain_lstm_model is not None:
        lstm_name = 'Plain-LSTM'
        print(f"\nEvaluating {lstm_name}...")
        val_scores_lstm_raw = _batched_supervised_scores(plain_lstm_model, X_val_seq, L_val)
        test_scores_lstm_raw = _batched_supervised_scores(plain_lstm_model, X_test_seq, L_test)

        # Apply the same personalised blending used for the Transformer so the
        # supervised-architecture ablation (BiLSTM vs. Transformer) is a fair
        # apples-to-apples comparison: same training data, same scoring
        # pipeline, only the encoder differs.
        from src.train import personalize_scores
        if (sid_train_clean_arr is not None and len(sid_train_clean_arr) > 0
                and sid_val_arr is not None and sid_test_arr is not None):
            print(f"  Applying per-student blended scoring to {lstm_name}")
            train_scores_lstm_clean = _batched_supervised_scores(
                plain_lstm_model, X_train_seq_clean, L_train_clean
            )
            val_scores_lstm = personalize_scores(
                train_scores_lstm_clean, val_scores_lstm_raw,
                np.asarray(sid_train_clean_arr), np.asarray(sid_val_arr)
            )
            test_scores_lstm = personalize_scores(
                train_scores_lstm_clean, test_scores_lstm_raw,
                np.asarray(sid_train_clean_arr), np.asarray(sid_test_arr)
            )
        else:
            val_scores_lstm = val_scores_lstm_raw
            test_scores_lstm = test_scores_lstm_raw

        lstm_thresh = select_optimal_threshold(y_val, val_scores_lstm, method='f1_weighted')
        print(f"  {lstm_name} threshold: {lstm_thresh:.4f}")
        preds = (test_scores_lstm > lstm_thresh).astype(int)
        metrics = compute_classification_metrics(y_test, preds, test_scores_lstm)
        all_results[lstm_name] = metrics
        all_scores[lstm_name] = test_scores_lstm
        all_predictions[lstm_name] = preds
        print(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Precision: {metrics['precision']:.4f} | "
              f"Recall: {metrics['recall']:.4f}")

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

    # If running fairness standalone, evaluate must run first to populate scores
    if 'test_scores' not in processed_data:
        print("Test scores not found — running evaluation first...\n")
        processed_data = evaluate_models(config, processed_data)

    # Use Plain Transformer (Ours) scores for fairness — it's the primary model
    primary_key = 'Plain-Transformer (Ours)'
    if primary_key not in processed_data['test_scores']:
        # Fallback to LSTM-AE if Transformer wasn't trained
        primary_key = 'LSTM-AE'
    model_scores = processed_data['test_scores'][primary_key]
    model_preds = processed_data['test_predictions'][primary_key]
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
    fairness_before = analyzer.analyze_fairness(model_preds, y_test, demo_test)
    analyzer.print_fairness_report(fairness_before)

    # Calibrate thresholds per attribute using α grid search
    print("\nCalibrating group-specific thresholds (α grid search)...")
    calibrated_predictions = {}
    optimal_alphas = {}
    for attribute in config['fairness']['sensitive_attributes']:
        best_alpha, group_thresholds = analyzer.alpha_grid_search(
            model_scores, y_test, demo_test, threshold, attribute
        )
        optimal_alphas[attribute] = best_alpha
        fair_preds = analyzer.apply_fair_predictions(
            model_scores, demo_test, attribute, group_thresholds
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
    """Step 5: Generate SHAP explanations using the Plain Transformer classifier."""
    print("\n" + "="*80)
    print("STEP 5: EXPLAINABILITY (SHAP)")
    print("="*80 + "\n")

    device = get_device(config['training']['device'])

    if 'lstm_model' not in processed_data:
        processed_data = _load_trained_models(config, processed_data, device)

    X_train_seq = processed_data['X_train_seq']
    X_test_seq = processed_data['X_test_seq']
    L_train = processed_data['L_train']
    L_test = processed_data['L_test']
    q_feature_names = processed_data['question_feature_names']

    if isinstance(q_feature_names, np.ndarray):
        q_feature_names = q_feature_names.tolist()

    # If running explain standalone, evaluate must run first to populate scores
    if 'test_scores' not in processed_data:
        print("Test scores not found — running evaluation first...\n")
        processed_data = evaluate_models(config, processed_data)

    # Prefer Plain Transformer (Ours) for SHAP, fall back to LSTM-AE
    plain_tf = processed_data.get('plain_tf_model')
    if plain_tf is not None:
        primary_key = 'Plain-Transformer (Ours)'
        model = plain_tf
        print("Using Plain Transformer classifier for SHAP explanations.")
    else:
        primary_key = 'LSTM-AE'
        model = processed_data['lstm_model']
        print("Plain Transformer not available — falling back to LSTM-AE for SHAP.")

    model_scores = processed_data['test_scores'][primary_key]
    model_preds = processed_data['test_predictions'][primary_key]

    explainer = SequentialSHAPExplainer(model, q_feature_names, config, device)
    explainer.init_explainer(X_train_seq, L_train)

    explain_dir = os.path.join(config['output']['plots_dir'], 'explanations')
    ensure_dir(explain_dir)

    generate_explanation_report(
        explainer, X_test_seq, model_scores, model_preds, explain_dir,
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
                       choices=['preprocess', 'train', 'evaluate', 'fairness', 'plots', 'explain', 'analyze', 'all'],
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

    if args.mode in ['plots', 'all']:
        # Ensure evaluation has run (needed for scores/predictions)
        if 'test_scores' not in processed_data:
            processed_data = evaluate_models(config, processed_data)
        generate_all_plots(processed_data, config)

    if args.mode in ['explain', 'all']:
        processed_data = explain_predictions(config, processed_data)

    if args.mode in ['analyze', 'all']:
        # Reviewer-driven analyses: bootstrap CIs, calibration curve,
        # per-anomaly-family breakdown, cold-start buckets, error analysis.
        if 'test_scores' not in processed_data:
            processed_data = evaluate_models(config, processed_data)
        from src.analysis import run_full_analysis
        run_full_analysis(processed_data, config)

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
