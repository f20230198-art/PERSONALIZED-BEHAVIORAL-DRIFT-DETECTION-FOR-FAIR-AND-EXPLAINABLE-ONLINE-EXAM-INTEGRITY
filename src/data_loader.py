"""
Data loading module for EdNet-KT2 and OULAD datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm


class EdNetDataLoader:
    """Load and parse EdNet-KT2 student clickstream data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.student_files = sorted(list(self.data_path.glob("*.csv")))
        print(f"Found {len(self.student_files)} student files")
    
    def load_student(self, student_file: Path) -> pd.DataFrame:
        """Load a single student's data."""
        try:
            df = pd.read_csv(student_file)
            df['student_id'] = student_file.stem  # Extract student ID from filename
            return df
        except Exception as e:
            print(f"Error loading {student_file}: {e}")
            return None
    
    def load_batch(self, start_idx: int = 0, batch_size: int = 1000) -> List[pd.DataFrame]:
        """Load a batch of student files using parallel I/O."""
        import concurrent.futures
        
        end_idx = min(start_idx + batch_size, len(self.student_files))
        batch_files = self.student_files[start_idx:end_idx]
        
        # Parallel I/O — overlaps OneDrive latency across files
        n_workers = min(8, len(batch_files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(self.load_student, batch_files),
                total=len(batch_files), desc="Loading students"
            ))
        
        students_data = [df for df in results if df is not None and len(df) > 0]
        return students_data
    
    def load_all(self, max_students: int = None) -> List[pd.DataFrame]:
        """Load all student data (or up to max_students)."""
        if max_students:
            files_to_load = self.student_files[:max_students]
        else:
            files_to_load = self.student_files
        
        students_data = []
        for file in tqdm(files_to_load, desc="Loading all students"):
            df = self.load_student(file)
            if df is not None and len(df) > 0:
                students_data.append(df)
        
        print(f"Successfully loaded {len(students_data)} students")
        return students_data


class SessionCreator:
    """Create pseudo-exam sessions from EdNet clickstream data."""
    
    def __init__(self, min_questions: int = 20, max_questions: int = 50):
        self.min_questions = min_questions
        self.max_questions = max_questions
    
    def create_sessions(self, student_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Segment student activity into exam-like sessions.
        
        Sessions are defined as contiguous sequences of respond actions
        with reasonable time gaps between them.
        """
        sessions = []
        
        # Filter to only 'respond' actions (actual question attempts)
        responds = student_df[student_df['action_type'] == 'respond'].copy()
        
        if len(responds) < self.min_questions:
            return sessions
        
        # Convert timestamp to seconds
        responds['timestamp_sec'] = responds['timestamp'] / 1000.0
        responds = responds.sort_values('timestamp_sec').reset_index(drop=True)
        
        # Compute time gaps between consecutive actions (in minutes)
        responds['time_gap'] = responds['timestamp_sec'].diff() / 60.0
        
        # Define session boundaries: gaps > 30 minutes indicate session breaks
        SESSION_GAP_THRESHOLD = 30  # minutes
        session_breaks = responds[responds['time_gap'] > SESSION_GAP_THRESHOLD].index.tolist()
        
        # Add start and end boundaries
        boundaries = [0] + session_breaks + [len(responds)]
        
        # Extract sessions
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            session = responds.iloc[start:end].copy()
            
            # Only keep sessions with sufficient questions
            if len(session) >= self.min_questions:
                # Limit to max_questions if exceeded
                if len(session) > self.max_questions:
                    session = session.iloc[:self.max_questions]
                
                sessions.append(session)
        
        return sessions
    
    def create_all_sessions(self, students_data: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
        """Create sessions for all students."""
        all_sessions = []
        all_student_ids = []
        
        for student_df in tqdm(students_data, desc="Creating sessions"):
            sessions = self.create_sessions(student_df)
            student_id = student_df['student_id'].iloc[0]
            
            for session in sessions:
                all_sessions.append(session)
                all_student_ids.append(student_id)
        
        print(f"Created {len(all_sessions)} sessions from {len(students_data)} students")
        return all_sessions, all_student_ids


class OULADLoader:
    """Load OULAD demographic data."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_student_info(self) -> pd.DataFrame:
        """Load student demographic information."""
        filepath = self.data_path / "studentInfo.csv"
        df = pd.read_csv(filepath)
        
        # Select relevant demographic columns
        demo_cols = ['id_student', 'gender', 'region', 'highest_education',
                     'imd_band', 'age_band', 'disability']
        
        df = df[demo_cols].copy()
        df = df.rename(columns={'id_student': 'student_id'})
        
        print(f"Loaded {len(df)} students with demographic data")
        return df
    
    def get_demographic_groups(self, df: pd.DataFrame, attribute: str) -> Dict[str, pd.DataFrame]:
        """Split students into groups by demographic attribute."""
        groups = {}
        for value in df[attribute].unique():
            groups[value] = df[df[attribute] == value]
        return groups


def pool_demographics(demographics: List[Dict]) -> List[Dict]:
    """
    Pool sparse demographic groups into broader categories for fairness analysis.

    Fixes two problems:
      1. age_band '55<=' has too few samples for meaningful fairness analysis.
         Pool into binary: 'Under 35' and '35+'.
      2. imd_band has 10 subgroups with high variance.
         Pool into 3 groups: 'Low' (0-30%), 'Medium' (30-70%), 'High' (70-100%).
    """
    # Mapping for age bands -> binary
    age_band_map = {
        '0-35': 'Under 35',
        '35-55': '35+',
        '55<=': '35+',
    }

    # Mapping for IMD bands -> 3 groups
    imd_band_map = {
        '0-10%': 'Low',
        '10-20': 'Low',    # note: OULAD uses '10-20' without %
        '10-20%': 'Low',
        '20-30%': 'Low',
        '30-40%': 'Medium',
        '40-50%': 'Medium',
        '50-60%': 'Medium',
        '60-70%': 'Medium',
        '70-80%': 'High',
        '80-90%': 'High',
        '90-100%': 'High',
    }

    pooled = []
    for demo in demographics:
        demo_copy = dict(demo)

        # Pool age_band
        if 'age_band' in demo_copy:
            raw = str(demo_copy['age_band'])
            demo_copy['age_band'] = age_band_map.get(raw, raw)

        # Pool imd_band
        if 'imd_band' in demo_copy:
            raw = str(demo_copy['imd_band'])
            if raw in ('nan', 'None', ''):
                demo_copy['imd_band'] = 'Unknown'
            else:
                demo_copy['imd_band'] = imd_band_map.get(raw, 'Unknown')

        pooled.append(demo_copy)

    return pooled


def encode_demographics(demographics: List[Dict],
                        attribute: str = 'gender') -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode a demographic attribute as integer labels for adversarial training.

    Args:
        demographics: List of demographic dictionaries
        attribute: Which attribute to encode (e.g. 'gender')

    Returns:
        labels: (N,) integer array
        mapping: {group_name: integer_label}
    """
    unique_values = sorted(set(str(d.get(attribute, 'unknown')) for d in demographics))
    value_to_idx = {v: i for i, v in enumerate(unique_values)}
    labels = np.array([value_to_idx[str(d.get(attribute, 'unknown'))] for d in demographics])
    return labels, value_to_idx


def merge_demographics_with_sessions(sessions: List[pd.DataFrame],
                                     student_ids: List[str],
                                     oulad_df: pd.DataFrame) -> List[Tuple[pd.DataFrame, Dict]]:
    """
    Merge OULAD demographic data with EdNet sessions.
    
    Note: EdNet and OULAD have different student IDs, so we'll randomly assign
    OULAD demographics to EdNet students for demonstration purposes.
    In a real scenario, you'd need overlapping datasets or matched demographics.
    """
    # Random assignment for demonstration
    np.random.seed(42)
    oulad_sample = oulad_df.sample(len(sessions), replace=True).reset_index(drop=True)
    
    merged_sessions = []
    for i, session in enumerate(sessions):
        demographics = oulad_sample.iloc[i].to_dict()
        merged_sessions.append((session, demographics))
    
    print(f"Merged demographics with {len(merged_sessions)} sessions")
    return merged_sessions
