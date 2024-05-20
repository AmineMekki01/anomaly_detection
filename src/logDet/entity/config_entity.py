from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataParams:
    path_to_data: Path
    hdfs_file_name: str
    log_file_name:  str
    labels_file_name: str
    log_file_path: Path
    label_file_path: Path
    test_ratio: float
    train_ratio: float
    train_sessions_path: Path
    test_sessions_path: Path
    random_partition: bool
    train_anomaly_ratio: float

