"""
Realistic cheating rules grounded in online-proctoring literature.

Replaces the earlier cartoonish 3-8x compression with empirically-grounded
perturbations.  References (summarised):
  - Fask, Englander, Wang (2014): copied answers cluster at 1.5-2.5x typical speed
  - Alessio et al. (2017): lookup behavior = 15-45s pause followed by fast burst
  - Cizek & Wollack (2017): impersonation shows uniform pacing with low variance
  - Bawarith et al. (2017): collusion shows panic-compression at session end
  - Dendir & Maxwell (2020): tab-switching proxy = micro-pauses 5-12s clustered

All factors here are designed so a human grader inspecting the perturbed
session would say "yeah, that looks suspicious but plausible", not
"that is physically impossible".
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Realistic timing ranges (seconds), drawn from the references above.
# ---------------------------------------------------------------------------
MIN_HUMAN_RESPONSE_SEC = 1.0       # Physical floor: reading a question takes >=1s
LOOKUP_PAUSE_MIN, LOOKUP_PAUSE_MAX = 15.0, 45.0
COPY_PACE_MIN, COPY_PACE_MAX = 3.5, 7.0  # seconds per answer when copying
COMPRESSION_MIN, COMPRESSION_MAX = 1.5, 3.0   # realistic speed-up factor


class RealisticCheatingGenerator:
    """Generate realistic cheating patterns at the raw-session level.

    Each method perturbs a pd.DataFrame representing one session and returns
    a new dataframe with the same schema.  Perturbations are bounded by
    human-physical limits (response times clipped to >=1s, no negative gaps).
    """

    def __init__(self, contamination_rate: float = 0.15, seed: int = 42):
        self.contamination_rate = contamination_rate
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Helper: rebuild timestamps from gaps, respecting the human floor.
    # ------------------------------------------------------------------
    @staticmethod
    def _rebuild_timestamps(anomalous: pd.DataFrame, gaps: np.ndarray) -> pd.DataFrame:
        min_time = anomalous['timestamp_sec'].min()
        # Clip: first gap is 0; subsequent gaps must be >= MIN_HUMAN_RESPONSE_SEC
        gaps = np.asarray(gaps, dtype=float)
        gaps[0] = 0.0
        gaps[1:] = np.clip(gaps[1:], MIN_HUMAN_RESPONSE_SEC, None)
        new_ts = min_time + np.cumsum(gaps)
        anomalous = anomalous.copy()
        anomalous['timestamp_sec'] = new_ts
        return anomalous

    # ------------------------------------------------------------------
    # Pattern 1: Answer copying (uniform copy-paste pace)
    #   Ref: Cizek & Wollack (2017) — impersonators show low time variance
    # ------------------------------------------------------------------
    def answer_copying(self, session: pd.DataFrame) -> pd.DataFrame:
        n = len(session)
        pace = random.uniform(COPY_PACE_MIN, COPY_PACE_MAX)
        gaps = np.full(n, pace)
        gaps += np.random.uniform(-0.4, 0.4, size=n)  # small human jitter
        return self._rebuild_timestamps(session, gaps)

    # ------------------------------------------------------------------
    # Pattern 2: Lookup behavior (pause -> burst)
    #   Ref: Alessio et al. (2017) — tab-switching to search engines
    # ------------------------------------------------------------------
    def lookup_behavior(self, session: pd.DataFrame) -> pd.DataFrame:
        anomalous = session.copy()
        n = len(anomalous)
        original_gaps = anomalous['timestamp_sec'].diff().fillna(0).values.astype(float)

        # 2-4 lookup events, each: long pause then 2-4 fast answers
        num_events = random.randint(2, 4)
        if n <= 6:
            num_events = 1
        safe_range = max(1, n - 5)
        starts = sorted(random.sample(range(1, safe_range + 1), min(num_events, safe_range)))

        for s in starts:
            pause = random.uniform(LOOKUP_PAUSE_MIN, LOOKUP_PAUSE_MAX)
            original_gaps[s] = pause
            burst_len = random.randint(2, 4)
            for j in range(s + 1, min(s + 1 + burst_len, n)):
                original_gaps[j] = random.uniform(1.5, 3.5)  # fast but plausible
        return self._rebuild_timestamps(anomalous, original_gaps)

    # ------------------------------------------------------------------
    # Pattern 3: Partial-session cheating (help arrives mid-exam)
    #   Ref: Bawarith et al. (2017) — compression at session end
    # ------------------------------------------------------------------
    def partial_session(self, session: pd.DataFrame) -> pd.DataFrame:
        anomalous = session.copy()
        n = len(anomalous)
        transition = random.randint(int(n * 0.35), int(n * 0.65))
        compression = random.uniform(COMPRESSION_MIN, COMPRESSION_MAX)
        gaps = anomalous['timestamp_sec'].diff().fillna(0).values.astype(float)
        gaps[transition:] = gaps[transition:] / compression

        # Add a few answer changes in the cheating portion
        if 'user_answer' in anomalous.columns:
            post = anomalous.iloc[transition:]
            qids = post['item_id'].unique()
            new_rows = []
            for item_id in qids:
                if random.random() < 0.35:
                    row = anomalous[anomalous['item_id'] == item_id].iloc[0].copy()
                    row['user_answer'] = random.choice(['a', 'b', 'c', 'd'])
                    row['timestamp_sec'] += random.uniform(1.5, 3.5)
                    new_rows.append(row)
            anomalous = self._rebuild_timestamps(anomalous, gaps)
            if new_rows:
                anomalous = pd.concat([anomalous, pd.DataFrame(new_rows)]).sort_values('timestamp_sec')
            return anomalous
        return self._rebuild_timestamps(anomalous, gaps)

    # ------------------------------------------------------------------
    # Pattern 4: Pre-known answers (fast + high confidence)
    #   Ref: Fask, Englander, Wang (2014) — leaked answer keys
    # ------------------------------------------------------------------
    def pre_known_answers(self, session: pd.DataFrame) -> pd.DataFrame:
        anomalous = session.copy()
        n = len(anomalous)
        compression = random.uniform(COMPRESSION_MIN, COMPRESSION_MAX)
        gaps = anomalous['timestamp_sec'].diff().fillna(0).values.astype(float)
        gaps = gaps / compression
        return self._rebuild_timestamps(anomalous, gaps)

    # ------------------------------------------------------------------
    # Pattern 5: Excessive answer changes (second-guessing after external input)
    # ------------------------------------------------------------------
    def excessive_changes(self, session: pd.DataFrame) -> pd.DataFrame:
        anomalous = session.copy()
        if 'user_answer' not in anomalous.columns:
            return anomalous
        qids = anomalous['item_id'].unique()
        if len(qids) == 0:
            return anomalous
        affected = random.sample(
            list(qids),
            k=max(1, random.randint(int(len(qids) * 0.3), int(len(qids) * 0.55)))
        )
        new_rows = []
        for item_id in affected:
            item_rows = anomalous[anomalous['item_id'] == item_id]
            num_changes = random.randint(1, 3)  # realistic: 1-3 extra changes
            for i in range(num_changes):
                row = item_rows.iloc[0].copy()
                row['user_answer'] = random.choice(['a', 'b', 'c', 'd'])
                row['timestamp_sec'] += (i + 1) * random.uniform(2.0, 6.0)
                new_rows.append(row)
        if new_rows:
            anomalous = pd.concat([anomalous, pd.DataFrame(new_rows)]).sort_values('timestamp_sec')
        return anomalous

    # ------------------------------------------------------------------
    # Pattern 6: Panic compression (student runs out of time)
    # ------------------------------------------------------------------
    def panic_compression(self, session: pd.DataFrame) -> pd.DataFrame:
        anomalous = session.copy()
        n = len(anomalous)
        start = int(n * random.uniform(0.6, 0.75))
        compression = random.uniform(COMPRESSION_MIN, COMPRESSION_MAX + 0.5)  # panic up to 3.5x
        gaps = anomalous['timestamp_sec'].diff().fillna(0).values.astype(float)
        gaps[start:] = gaps[start:] / compression
        return self._rebuild_timestamps(anomalous, gaps)

    # ------------------------------------------------------------------
    def generate_anomaly(self, session: pd.DataFrame) -> pd.DataFrame:
        """Apply one primary cheating pattern + optional supplementary."""
        primary = [
            self.answer_copying,
            self.lookup_behavior,
            self.partial_session,
            self.pre_known_answers,
            self.panic_compression,
        ]
        result = random.choice(primary)(session)
        if random.random() < 0.5:
            result = self.excessive_changes(result)
        return result

    def inject_anomalies(self, sessions: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], np.ndarray]:
        num_sessions = len(sessions)
        num_anomalies = int(num_sessions * self.contamination_rate)
        anomaly_indices = set(random.sample(range(num_sessions), num_anomalies))
        labels = np.zeros(num_sessions, dtype=int)
        processed = []
        for i, session in enumerate(sessions):
            if i in anomaly_indices:
                processed.append(self.generate_anomaly(session))
                labels[i] = 1
            else:
                processed.append(session)
        print(f"[RealisticCheating] Injected {num_anomalies} realistic anomalies "
              f"({self.contamination_rate*100:.1f}%) into {num_sessions} sessions")
        return processed, labels
