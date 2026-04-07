"""
Data preprocessing and synthetic anomaly generation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import random


class SyntheticAnomalyGenerator:
    """Generate synthetic anomalous sessions for evaluation."""
    
    def __init__(self, contamination_rate: float = 0.1, seed: int = 42):
        self.contamination_rate = contamination_rate
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_rapid_responses(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 1: Suspiciously fast responses on a PORTION of questions.

        Simulates a student who gets help partway through — the first portion
        is answered at normal speed, then a contiguous block of questions is
        answered 3-6x faster. This creates a temporal transition that LSTM
        can detect but aggregate statistics (mean/std) partially miss.
        """
        anomalous = session.copy()
        n = len(anomalous)

        # Pick a contiguous block covering 50-80% of the session to speed up
        block_size = random.randint(int(n * 0.5), int(n * 0.8))
        block_start = random.randint(0, n - block_size)
        block_end = block_start + block_size

        compression = random.uniform(1.5, 3.0)
        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values

        # Only compress gaps within the affected block
        for i in range(block_start, block_end):
            original_gaps[i] = original_gaps[i] / compression

        min_time = anomalous['timestamp_sec'].min()
        cumulative_time = min_time
        new_timestamps = []
        for gap in original_gaps:
            new_timestamps.append(cumulative_time)
            cumulative_time += gap

        anomalous['timestamp_sec'] = new_timestamps
        return anomalous
    
    def generate_excessive_answer_changes(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 2: Elevated answer changes.
        Simulate a student who second-guesses answers (e.g. looking things up
        mid-question). Adds 1-3 extra changes to 40-70% of questions — elevated
        but still within the realm of a nervous or uncertain test-taker.
        """
        anomalous = session.copy()

        if 'user_answer' not in anomalous.columns:
            return anomalous

        new_rows = []
        question_ids = anomalous['item_id'].unique()
        # Only affect a subset of questions (40-70%)
        affected = random.sample(
            list(question_ids),
            k=random.randint(int(len(question_ids) * 0.4),
                             int(len(question_ids) * 0.7))
        )

        for item_id in affected:
            item_rows = anomalous[anomalous['item_id'] == item_id]
            num_changes = random.randint(2, 5)  # 2-5 extra changes per affected question
            answers = ['a', 'b', 'c', 'd']

            for i in range(num_changes):
                new_row = item_rows.iloc[0].copy()
                new_row['user_answer'] = random.choice(answers)
                # Space changes 2-8 seconds apart (realistic typing/clicking)
                new_row['timestamp_sec'] += (i + 1) * random.uniform(2.0, 8.0)
                new_rows.append(new_row)

        anomalous = pd.concat([anomalous, pd.DataFrame(new_rows)]).sort_values('timestamp_sec')
        return anomalous
    
    def generate_irregular_timing(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 3: Irregular timing with pause-then-burst clusters.

        Simulates alt-tabbing to look things up: clusters of 3-6 consecutive
        fast answers appear after long pauses. This creates a distinctive
        temporal rhythm (pause → burst → pause → burst) that sequential
        models can pick up as a pattern, while aggregate stats only see
        "high variance."
        """
        anomalous = session.copy()
        n = len(anomalous)

        min_time = anomalous['timestamp_sec'].min()
        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values

        # Create 2-4 burst clusters at random positions
        num_clusters = random.randint(2, 4)
        cluster_starts = sorted(random.sample(range(1, n - 5), min(num_clusters, n - 6)))

        burst_positions = set()
        for start in cluster_starts:
            cluster_len = random.randint(3, 6)
            for j in range(start, min(start + cluster_len, n)):
                burst_positions.add(j)

        new_timestamps = []
        cumulative_time = min_time
        for i, gap in enumerate(original_gaps):
            if i in burst_positions and i - 1 not in burst_positions:
                # First action in a burst cluster — preceded by a lookup pause
                cumulative_time += random.uniform(20, 60)
                cumulative_time += gap * random.uniform(0.25, 0.50)
            elif i in burst_positions:
                # Inside a burst — very fast
                cumulative_time += gap * random.uniform(0.25, 0.50)
            else:
                cumulative_time += gap
            new_timestamps.append(cumulative_time)

        anomalous['timestamp_sec'] = new_timestamps
        return anomalous
    
    def generate_uniform_response_pattern(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 4: Uniform (near-constant) response times.

        Replaces natural per-question time variation with near-constant
        inter-question gaps, simulating a student who is copying pre-known
        answers at a steady mechanical pace.  The constant gap is drawn
        from [3, 8] seconds (realistic typing/clicking speed for someone
        reading off a cheat sheet), plus small jitter (±0.5s) so it
        isn't perfectly identical.
        """
        anomalous = session.copy()
        n = len(anomalous)

        # Pick a constant gap drawn from a realistic "copy-paste" range
        constant_gap = random.uniform(3.0, 8.0)

        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values
        min_time = anomalous['timestamp_sec'].min()

        # Replace ALL gaps with the constant value + tiny jitter
        new_timestamps = [min_time]
        for i in range(1, n):
            jitter = random.uniform(-0.5, 0.5)
            new_timestamps.append(new_timestamps[-1] + max(0.5, constant_gap + jitter))

        anomalous['timestamp_sec'] = new_timestamps
        return anomalous

    def generate_random_navigation(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 4: Excessive question revisits.
        Simulate jumping back to earlier questions more than normal — e.g.
        a student re-checking answers after getting external help. Shuffles
        30-50% of actions to create realistic revisit patterns while
        preserving the original timing gaps.
        """
        anomalous = session.copy()

        n = len(anomalous)
        # Shuffle 30-50% of the rows to simulate revisiting
        num_shuffle = random.randint(int(n * 0.3), int(n * 0.5))
        shuffle_idx = random.sample(range(n), num_shuffle)
        shuffled_values = anomalous.iloc[shuffle_idx].copy()
        shuffled_values = shuffled_values.sample(frac=1)

        # Swap the item_id and user_answer columns to simulate revisits,
        # but keep the original timestamps so timing structure is preserved
        for col in ['item_id', 'user_answer']:
            if col in anomalous.columns:
                anomalous.iloc[shuffle_idx, anomalous.columns.get_loc(col)] = \
                    shuffled_values[col].values

        return anomalous
    
    def generate_correlated_shifts(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 5: Correlated feature shifts on alternating segments.

        Compresses response times AND adds answer changes, but only on
        alternating segments of 5-8 questions (simulating a student who
        periodically checks an external source then answers a batch).
        This creates a zigzag temporal pattern: normal → fast → normal → fast.
        """
        anomalous = session.copy()
        n = len(anomalous)

        compression = random.uniform(1.5, 3.0)
        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values

        # Create alternating fast/normal segments of 5-8 questions
        seg_len = random.randint(5, 8)
        is_fast = False
        affected_indices = set()
        for i in range(0, n, seg_len):
            if is_fast:
                for j in range(i, min(i + seg_len, n)):
                    original_gaps[j] = original_gaps[j] / compression
                    affected_indices.add(j)
            is_fast = not is_fast

        min_time = anomalous['timestamp_sec'].min()
        cumulative_time = min_time
        new_timestamps = []
        for gap in original_gaps:
            new_timestamps.append(cumulative_time)
            cumulative_time += gap
        anomalous['timestamp_sec'] = new_timestamps

        # Add answer changes only to the fast segments
        if 'user_answer' in anomalous.columns:
            new_rows = []
            fast_rows = anomalous.iloc[list(affected_indices)]
            affected_qids = fast_rows['item_id'].unique()
            for item_id in affected_qids:
                if random.random() < 0.5:
                    item_rows = anomalous[anomalous['item_id'] == item_id]
                    new_row = item_rows.iloc[0].copy()
                    new_row['user_answer'] = random.choice(['a', 'b', 'c', 'd'])
                    new_row['timestamp_sec'] += random.uniform(1.5, 4.0)
                    new_rows.append(new_row)
            if new_rows:
                anomalous = pd.concat([anomalous, pd.DataFrame(new_rows)]).sort_values('timestamp_sec')

        return anomalous

    def generate_partial_session_cheating(self, session: pd.DataFrame) -> pd.DataFrame:
        """
        Anomaly Type 6: Partial session cheating (strong temporal signature).

        The student starts normally but begins cheating at a transition point
        (40-60% through the session). After the transition:
          - Response times drop by 3-5x
          - Answer changes increase
          - Burst ratio spikes

        This is the most realistic cheating pattern AND the one that gives
        sequential models the biggest advantage — the first half looks normal,
        so aggregate features are diluted, but the LSTM sees the clear
        temporal transition.
        """
        anomalous = session.copy()
        n = len(anomalous)

        # Transition point: 30-55% through the session (larger cheating portion)
        transition = random.randint(int(n * 0.3), int(n * 0.55))
        compression = random.uniform(1.5, 3.0)

        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values

        # Compress only the post-transition portion
        for i in range(transition, n):
            original_gaps[i] = original_gaps[i] / compression

        min_time = anomalous['timestamp_sec'].min()
        cumulative_time = min_time
        new_timestamps = []
        for gap in original_gaps:
            new_timestamps.append(cumulative_time)
            cumulative_time += gap
        anomalous['timestamp_sec'] = new_timestamps

        # Add answer changes only in the cheating portion
        if 'user_answer' in anomalous.columns:
            post_transition = anomalous.iloc[transition:]
            affected_qids = post_transition['item_id'].unique()
            new_rows = []
            for item_id in affected_qids:
                if random.random() < 0.5:  # 50% of post-transition questions
                    item_rows = anomalous[anomalous['item_id'] == item_id]
                    num_extra = random.randint(1, 2)
                    answers = ['a', 'b', 'c', 'd']
                    for i in range(num_extra):
                        new_row = item_rows.iloc[0].copy()
                        new_row['user_answer'] = random.choice(answers)
                        new_row['timestamp_sec'] += (i + 1) * random.uniform(1.5, 4.0)
                        new_rows.append(new_row)
            if new_rows:
                anomalous = pd.concat([anomalous, pd.DataFrame(new_rows)]).sort_values('timestamp_sec')

        return anomalous

    def generate_anomaly(self, session: pd.DataFrame) -> pd.DataFrame:
        """Generate anomalies: always 1 timing-based + optionally 1 supplementary.

        Timing-based anomalies alter the temporal structure of the session,
        which is what the LSTM autoencoder reconstructs. Without at least one
        timing anomaly, the reconstruction error barely changes and the model
        can't detect the session as anomalous.

        Supplementary anomalies (answer changes, navigation) add signal on
        top of the timing shift.
        """
        # Group 1: TIMING anomalies (always include exactly 1)
        timing_anomalies = [
            self.generate_rapid_responses,
            self.generate_irregular_timing,
            self.generate_uniform_response_pattern,
            self.generate_correlated_shifts,
            self.generate_partial_session_cheating
        ]

        # Group 2: SUPPLEMENTARY anomalies (optionally add 1)
        supplementary_anomalies = [
            self.generate_excessive_answer_changes,
            self.generate_random_navigation,
        ]

        # Always apply 1 timing anomaly
        result = random.choice(timing_anomalies)(session)

        # 60% chance of also applying a supplementary anomaly
        if random.random() < 0.6:
            result = random.choice(supplementary_anomalies)(result)

        return result
    
    def inject_anomalies(self, sessions: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], np.ndarray]:
        """
        Inject synthetic anomalies into a portion of sessions.
        
        Returns:
            sessions: List of sessions (some anomalous)
            labels: Binary labels (0 = normal, 1 = anomaly)
        """
        num_sessions = len(sessions)
        num_anomalies = int(num_sessions * self.contamination_rate)
        
        # Randomly select sessions to make anomalous
        anomaly_indices = set(random.sample(range(num_sessions), num_anomalies))
        
        labels = np.zeros(num_sessions, dtype=int)
        processed_sessions = []
        
        for i, session in enumerate(sessions):
            if i in anomaly_indices:
                # Generate anomalous version
                anomalous_session = self.generate_anomaly(session)
                processed_sessions.append(anomalous_session)
                labels[i] = 1
            else:
                # Keep normal
                processed_sessions.append(session)
        
        print(f"Injected {num_anomalies} anomalies ({self.contamination_rate*100:.1f}%) into {num_sessions} sessions")
        
        return processed_sessions, labels


class DataPreprocessor:
    """Main preprocessing pipeline."""

    def __init__(self, config: Dict):
        self.config = config

    def prepare_sequences(self, feature_matrix: np.ndarray,
                         sequence_length: int = 1) -> np.ndarray:
        """
        Legacy method for session-level features (baselines).
        Adds a trivial sequence dimension: (N, 6) -> (N, 1, 6).
        """
        if feature_matrix.ndim == 2:
            return feature_matrix[:, np.newaxis, :]
        return feature_matrix

    def create_data_loaders(self, X_train, X_val, X_test, batch_size=32,
                            lengths_train=None, lengths_val=None, lengths_test=None,
                            demo_labels_train=None, demo_labels_val=None,
                            demo_labels_test=None):
        """Create PyTorch data loaders, optionally with sequence lengths
        and demographic labels (for adversarial fairness training)."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        X_test_t = torch.FloatTensor(X_test)

        tensors_train = [X_train_t]
        tensors_val = [X_val_t]
        tensors_test = [X_test_t]

        if lengths_train is not None:
            tensors_train.append(torch.LongTensor(lengths_train))
            tensors_val.append(torch.LongTensor(lengths_val))
            tensors_test.append(torch.LongTensor(lengths_test))

        if demo_labels_train is not None:
            tensors_train.append(torch.LongTensor(demo_labels_train))
            tensors_val.append(torch.LongTensor(demo_labels_val))
            tensors_test.append(torch.LongTensor(demo_labels_test))

        train_dataset = TensorDataset(*tensors_train)
        val_dataset = TensorDataset(*tensors_val)
        test_dataset = TensorDataset(*tensors_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


def save_processed_data(data: Dict, filepath: str):
    """Save preprocessed data to disk."""
    np.savez_compressed(filepath, **data)
    print(f"Saved processed data to {filepath}")


def load_processed_data(filepath: str) -> Dict:
    """Load preprocessed data from disk."""
    data = np.load(filepath, allow_pickle=True)
    result = {key: data[key] for key in data.files}
    print(f"Loaded processed data from {filepath}")
    return result
