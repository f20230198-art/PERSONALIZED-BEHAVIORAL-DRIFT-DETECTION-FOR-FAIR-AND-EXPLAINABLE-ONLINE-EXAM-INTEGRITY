"""
Feature extraction for behavioral drift detection.

Implements 6 behavioral features:
1. Response Time Mean
2. Response Time Std
3. Answer Change Rate
4. Keystroke Rhythm Variance
5. Pause Frequency
6. Question Sequence Deviation
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Tuple
from tqdm import tqdm


class BehavioralFeatureExtractor:
    """Extract behavioral features from exam sessions."""

    def __init__(self):
        self.feature_names = [
            'response_time_mean',
            'response_time_std',
            'answer_change_rate',
            'keystroke_rhythm_variance',
            'pause_frequency',
            'question_sequence_deviation',
            'response_time_entropy',
            'speed_score',
            'burst_ratio',
            'response_time_iqr'
        ]
        self.question_feature_names = [
            'time_per_question',
            'answer_change_count',
            'is_revisited',
            'inter_question_gap',
            'visit_count',
            'relative_position',
            'response_time_zscore',
            'is_burst',
            'cumulative_speed',
            'gap_ratio'
        ]
    
    def extract_response_times(self, session: pd.DataFrame) -> np.ndarray:
        """
        Calculate response time for each question.
        Response time = time between question entry and submission.
        """
        response_times = []
        
        # Group by question (item_id)
        for item_id in session['item_id'].unique():
            item_actions = session[session['item_id'] == item_id].sort_values('timestamp_sec')
            
            if len(item_actions) >= 2:
                # Time from first action to last action for this question
                start_time = item_actions['timestamp_sec'].iloc[0]
                end_time = item_actions['timestamp_sec'].iloc[-1]
                response_time = end_time - start_time
                
                # Sanity check: response time should be positive and reasonable (< 10 minutes)
                if 0 < response_time < 600:
                    response_times.append(response_time)
        
        return np.array(response_times) if response_times else np.array([0.0])
    
    def extract_answer_changes(self, session: pd.DataFrame) -> int:
        """
        Count number of times the student changed their answer.
        Answer changes indicate uncertainty or potential cheating.
        """
        answer_changes = 0
        
        # Group by question
        for item_id in session['item_id'].unique():
            item_actions = session[session['item_id'] == item_id]
            
            # Check if multiple different answers were given
            if 'user_answer' in item_actions.columns:
                answers = item_actions['user_answer'].dropna().tolist()
                if len(answers) > 1:
                    # Count transitions between different answers
                    for i in range(len(answers) - 1):
                        if answers[i] != answers[i + 1]:
                            answer_changes += 1
        
        return answer_changes
    
    def extract_keystroke_rhythms(self, session: pd.DataFrame) -> np.ndarray:
        """
        Calculate inter-keystroke intervals (time between consecutive actions).
        Variance in rhythm can indicate copy-pasting or irregular behavior.
        """
        # Calculate time gaps between all consecutive actions
        time_gaps = session['timestamp_sec'].diff().dropna().values
        
        # Filter out unreasonably large gaps (> 2 minutes = likely pauses, not keystrokes)
        keystroke_intervals = time_gaps[time_gaps < 120]
        
        return keystroke_intervals if len(keystroke_intervals) > 0 else np.array([0.0])
    
    def extract_pauses(self, session: pd.DataFrame, pause_threshold: float = 30.0) -> int:
        """
        Count number of long pauses (> 30 seconds) during the session.
        Unusual pause patterns may indicate suspicious behavior.
        """
        time_gaps = session['timestamp_sec'].diff().dropna().values
        pauses = (time_gaps > pause_threshold).sum()
        return pauses
    
    def extract_question_sequence_deviation(self, session: pd.DataFrame) -> float:
        """
        Measure revisit ratio: fraction of unique questions that the student
        returned to after visiting other questions.

        High revisit ratios may indicate checking answers against external
        sources or second-guessing after looking things up.

        This replaces the previous item_id ordering approach, which assumed
        question IDs were sequentially numbered — not true for EdNet data.
        """
        question_order = []
        seen = set()
        revisited = set()

        for _, row in session.iterrows():
            qid = row['item_id']
            if qid in seen and qid != (question_order[-1] if question_order else None):
                # Student returned to a previously-visited question
                # (consecutive actions on the same question don't count)
                revisited.add(qid)
            if qid not in seen:
                question_order.append(qid)
            seen.add(qid)

        num_unique = len(question_order)
        if num_unique == 0:
            return 0.0

        return len(revisited) / num_unique
    
    def extract_response_time_entropy(self, response_times: np.ndarray) -> float:
        """Compute entropy of response-time distribution (binned).

        Truly random timing (e.g. bot-generated) has high entropy; normal
        studying shows structured patterns with low entropy.
        """
        if len(response_times) < 2:
            return 0.0
        # Bin response times into 10 equal-width bins
        counts, _ = np.histogram(response_times, bins=10)
        counts = counts + 1e-10  # avoid log(0)
        probs = counts / counts.sum()
        return float(scipy_entropy(probs))

    def extract_speed_score(self, session: pd.DataFrame) -> float:
        """Questions answered per minute of session duration."""
        num_questions = len(session['item_id'].unique())
        duration_min = (session['timestamp_sec'].iloc[-1] - session['timestamp_sec'].iloc[0]) / 60.0
        if duration_min <= 0:
            return 0.0
        return num_questions / duration_min

    def extract_burst_ratio(self, session: pd.DataFrame, burst_threshold: float = 2.0) -> float:
        """Fraction of inter-action gaps that are below burst_threshold seconds.

        A high burst ratio indicates machine-like rapid-fire answering.
        """
        gaps = session['timestamp_sec'].diff().dropna().values
        if len(gaps) == 0:
            return 0.0
        return float((gaps < burst_threshold).sum() / len(gaps))

    def extract_features(self, session: pd.DataFrame) -> Dict[str, float]:
        """Extract all 10 behavioral features from a session."""
        # 1. Response Time features
        response_times = self.extract_response_times(session)
        rt_mean = np.mean(response_times)
        rt_std = np.std(response_times)

        # 2. Answer Change Rate
        answer_changes = self.extract_answer_changes(session)
        num_questions = len(session['item_id'].unique())
        answer_change_rate = answer_changes / num_questions if num_questions > 0 else 0.0

        # 3. Keystroke Rhythm Variance
        keystroke_intervals = self.extract_keystroke_rhythms(session)
        keystroke_rhythm_variance = np.var(keystroke_intervals)

        # 4. Pause Frequency
        pauses = self.extract_pauses(session)
        session_duration = session['timestamp_sec'].iloc[-1] - session['timestamp_sec'].iloc[0]
        pause_frequency = pauses / (session_duration / 60) if session_duration > 0 else 0.0

        # 5. Question Sequence Deviation
        sequence_deviation = self.extract_question_sequence_deviation(session)

        # 6. Response Time Entropy (new)
        rt_entropy = self.extract_response_time_entropy(response_times)

        # 7. Speed Score (new)
        speed_score = self.extract_speed_score(session)

        # 8. Burst Ratio (new)
        burst_ratio = self.extract_burst_ratio(session)

        # 9. Response Time IQR (new) — robust spread measure
        if len(response_times) >= 4:
            rt_iqr = float(np.percentile(response_times, 75) - np.percentile(response_times, 25))
        else:
            rt_iqr = rt_std

        features = {
            'response_time_mean': rt_mean,
            'response_time_std': rt_std,
            'answer_change_rate': answer_change_rate,
            'keystroke_rhythm_variance': keystroke_rhythm_variance,
            'pause_frequency': pause_frequency,
            'question_sequence_deviation': sequence_deviation,
            'response_time_entropy': rt_entropy,
            'speed_score': speed_score,
            'burst_ratio': burst_ratio,
            'response_time_iqr': rt_iqr
        }

        return features
    
    def extract_features_batch(self, sessions: List[pd.DataFrame]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract features from a batch of sessions.
        
        Returns:
            feature_matrix: (n_sessions, n_features) array
            feature_dicts: List of feature dictionaries for each session
        """
        feature_dicts = []
        
        for session in tqdm(sessions, desc="Extracting features"):
            features = self.extract_features(session)
            feature_dicts.append(features)
        
        # Convert to matrix
        feature_matrix = np.array([[f[name] for name in self.feature_names] 
                                   for f in feature_dicts])
        
        print(f"Extracted features for {len(sessions)} sessions")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix, feature_dicts
    
    def normalize_features(self, feature_matrix: np.ndarray,
                          mean: np.ndarray = None,
                          std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using z-score normalization.

        If mean and std are provided, use them (for test set).
        Otherwise, compute from data (for train set).
        """
        if mean is None:
            mean = np.mean(feature_matrix, axis=0)
        if std is None:
            std = np.std(feature_matrix, axis=0)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)

        normalized = (feature_matrix - mean) / std

        return normalized, mean, std

    # ------------------------------------------------------------------
    # QUESTION-LEVEL FEATURE EXTRACTION (for LSTM sequential input)
    # ------------------------------------------------------------------

    def extract_question_level_features(self, session: pd.DataFrame) -> Tuple[np.ndarray, int]:
        """
        Extract per-question features for LSTM sequential input (10 features).

        For each unique question (in order of first appearance), computes:
          0: time_per_question      – seconds spent on this question
          1: answer_change_count    – number of answer changes
          2: is_revisited           – 1.0 if question was revisited after other questions
          3: inter_question_gap     – seconds between previous question's end and this start
          4: visit_count            – total actions recorded for this question
          5: relative_position      – normalised position in session (0→1), captures
                                      temporal effects like fatigue or late-session rushing
          6: response_time_zscore   – z-score of this question's time within session
          7: is_burst               – 1.0 if response time < session median / 3
          8: cumulative_speed       – cumulative questions / time up to this point
          9: gap_ratio              – inter_question_gap / session_mean_gap

        Returns:
            features: (Q, 10) array
            num_questions: actual number of unique questions
        """
        NUM_FEATURES = 10

        # Get unique questions in order of first appearance
        question_order = []
        seen = set()
        for _, row in session.iterrows():
            qid = row['item_id']
            if qid not in seen:
                question_order.append(qid)
                seen.add(qid)

        num_questions = len(question_order)
        if num_questions == 0:
            return np.zeros((1, NUM_FEATURES)), 1

        features = np.zeros((num_questions, NUM_FEATURES))

        # Pre-compute per-question time_spent for z-score and burst detection
        time_spent_list = []
        gap_list = []

        for idx, qid in enumerate(question_order):
            q_actions = session[session['item_id'] == qid].sort_values('timestamp_sec')

            # Feature 0: time_per_question
            if len(q_actions) >= 2:
                time_spent = q_actions['timestamp_sec'].iloc[-1] - q_actions['timestamp_sec'].iloc[0]
            elif idx < num_questions - 1:
                next_qid = question_order[idx + 1]
                next_start = session[session['item_id'] == next_qid]['timestamp_sec'].min()
                time_spent = next_start - q_actions['timestamp_sec'].iloc[0]
            else:
                time_spent = 10.0
            time_spent = np.clip(time_spent, 0.1, 600)
            features[idx, 0] = time_spent
            time_spent_list.append(time_spent)

            # Feature 1: answer_change_count
            if 'user_answer' in q_actions.columns:
                answers = q_actions['user_answer'].dropna().tolist()
                changes = sum(1 for i in range(len(answers) - 1) if answers[i] != answers[i + 1])
            else:
                changes = 0
            features[idx, 1] = changes

            # Feature 2: is_revisited
            first_t = q_actions['timestamp_sec'].iloc[0]
            if len(q_actions) > 1:
                last_t = q_actions['timestamp_sec'].iloc[-1]
                between = session[(session['timestamp_sec'] > first_t) &
                                  (session['timestamp_sec'] < last_t) &
                                  (session['item_id'] != qid)]
                features[idx, 2] = 1.0 if len(between) > 0 else 0.0

            # Feature 3: inter_question_gap
            gap = 0.0
            if idx > 0:
                prev_qid = question_order[idx - 1]
                prev_end = session[session['item_id'] == prev_qid]['timestamp_sec'].max()
                gap = first_t - prev_end
                gap = np.clip(gap, 0, 300)
            features[idx, 3] = gap
            gap_list.append(gap)

            # Feature 4: visit_count
            features[idx, 4] = float(len(q_actions))

            # Feature 5: relative_position (0 → 1 across the session)
            features[idx, 5] = idx / max(num_questions - 1, 1)

        # ---------- NEW features (computed after first pass) ----------
        ts_arr = np.array(time_spent_list)
        ts_mean = ts_arr.mean()
        ts_std = ts_arr.std()
        ts_median = np.median(ts_arr)
        gap_arr = np.array(gap_list)
        gap_mean = gap_arr.mean() if gap_arr.mean() > 0 else 1.0
        session_start = session['timestamp_sec'].iloc[0]

        for idx, qid in enumerate(question_order):
            q_actions = session[session['item_id'] == qid].sort_values('timestamp_sec')

            # Feature 6: response_time_zscore (self-relative)
            if ts_std > 0:
                features[idx, 6] = (time_spent_list[idx] - ts_mean) / ts_std
            else:
                features[idx, 6] = 0.0

            # Feature 7: is_burst (time < median / 3)
            features[idx, 7] = 1.0 if time_spent_list[idx] < ts_median / 3.0 else 0.0

            # Feature 8: cumulative_speed (questions/minute up to this point)
            elapsed = q_actions['timestamp_sec'].iloc[-1] - session_start
            if elapsed > 0:
                features[idx, 8] = (idx + 1) / (elapsed / 60.0)
            else:
                features[idx, 8] = 0.0

            # Feature 9: gap_ratio (this gap / session mean gap)
            features[idx, 9] = gap_list[idx] / gap_mean

        return features, num_questions

    def extract_question_features_batch(self, sessions: List[pd.DataFrame],
                                        max_seq_len: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract question-level features for all sessions, padded to max_seq_len.

        Returns:
            all_features: (N, max_seq_len, 6) padded array
            all_lengths:  (N,) actual sequence lengths
        """
        all_features = []
        all_lengths = []

        num_q_features = len(self.question_feature_names)
        for session in tqdm(sessions, desc="Extracting question-level features"):
            feats, length = self.extract_question_level_features(session)
            if length > max_seq_len:
                feats = feats[:max_seq_len]
                length = max_seq_len
            padded = np.zeros((max_seq_len, num_q_features))
            padded[:length] = feats
            all_features.append(padded)
            all_lengths.append(length)

        all_features = np.array(all_features)
        all_lengths = np.array(all_lengths)
        print(f"Extracted question-level features: {all_features.shape}, "
              f"lengths range [{all_lengths.min()}-{all_lengths.max()}]")
        return all_features, all_lengths

    def normalize_question_features(self, features: np.ndarray,
                                    lengths: np.ndarray = None,
                                    mean: np.ndarray = None,
                                    std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Z-score normalize question-level features.
        Stats are computed only from non-padded positions.
        Padded positions are zeroed after normalization.
        """
        if mean is None or std is None:
            if lengths is not None:
                valid = []
                for i, l in enumerate(lengths):
                    valid.append(features[i, :l, :])
                all_valid = np.concatenate(valid, axis=0)
            else:
                all_valid = features.reshape(-1, features.shape[-1])
            mean = np.mean(all_valid, axis=0)
            std = np.std(all_valid, axis=0)
            std = np.where(std == 0, 1.0, std)

        normalized = (features - mean) / std

        if lengths is not None:
            for i, l in enumerate(lengths):
                normalized[i, l:, :] = 0.0

        return normalized, mean, std


def save_features(feature_matrix: np.ndarray, 
                 student_ids: List[str], 
                 demographics: List[Dict],
                 filepath: str):
    """Save extracted features to disk."""
    data = {
        'features': feature_matrix,
        'student_ids': student_ids,
        'demographics': demographics
    }
    np.savez(filepath, **data)
    print(f"Saved features to {filepath}")


def load_features(filepath: str) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """Load extracted features from disk."""
    data = np.load(filepath, allow_pickle=True)
    feature_matrix = data['features']
    student_ids = data['student_ids'].tolist()
    demographics = data['demographics'].tolist()
    print(f"Loaded features from {filepath}")
    return feature_matrix, student_ids, demographics
