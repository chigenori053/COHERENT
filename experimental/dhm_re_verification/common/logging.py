"""
Logging Utilities

Standardizes CSV output format for all experiment stages.
Adheres to Spec Section 6.1 and 6.2.
"""

import os
import csv
import time
from typing import Dict, Any, List

class ExperimentLogger:
    def __init__(self, experiment_name: str, base_dir: str = "experimental/dhm_re_verification/logs"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.sample_csv_path = os.path.join(base_dir, f"{experiment_name}_samples_{timestamp}.csv")
        self.agg_csv_path = os.path.join(base_dir, f"{experiment_name}_aggregate_{timestamp}.csv")
        
        self._init_sample_csv()

    def _init_sample_csv(self):
        headers = [
            "stage_id", "sample_id", "input_sequence", "predicted_sequence",
            "is_exact_match", "margin", "memory_layer", "conflict_flag", "vector_dimension"
        ]
        with open(self.sample_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_sample(self, row_data: Dict[str, Any]):
        """
        Log a single sample row.
        Expected keys match headers.
        """
        row = [
            row_data.get("stage_id", self.experiment_name),
            row_data.get("sample_id"),
            row_data.get("input_sequence"),
            row_data.get("predicted_sequence"),
            row_data.get("is_exact_match"),
            f"{row_data.get('margin', 0.0):.4f}",
            row_data.get("memory_layer", "D"),
            row_data.get("conflict_flag", False),
            row_data.get("vector_dimension")
        ]
        with open(self.sample_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_aggregate(self, stats: Dict[str, Any]):
        """
        Log aggregate statistics.
        headers: stage_id, vector_dimension, num_samples, exact_match_rate, 
                 static_rate, dynamic_rate, causal_rate, false_static_count, avg_margin, min_margin
        """
        headers = [
            "stage_id", "vector_dimension", "num_samples", "exact_match_rate",
            "static_rate", "dynamic_rate", "causal_rate", "false_static_count", 
            "avg_margin", "min_margin"
        ]
        
        # Check if file exists to write headers
        file_exists = os.path.exists(self.agg_csv_path)
        
        with open(self.agg_csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            row = [
                stats.get("stage_id", self.experiment_name),
                stats.get("vector_dimension"),
                stats.get("num_samples"),
                f"{stats.get('exact_match_rate', 0.0):.4f}",
                f"{stats.get('static_rate', 0.0):.4f}",
                f"{stats.get('dynamic_rate', 0.0):.4f}",
                f"{stats.get('causal_rate', 0.0):.4f}",
                stats.get("false_static_count", 0),
                f"{stats.get('avg_margin', 0.0):.4f}",
                f"{stats.get('min_margin', 0.0):.4f}"
            ]
            writer.writerow(row)
