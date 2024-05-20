from pathlib import Path
from src.logDet.constants import CONFIG_FILE_PATH
from src.logDet.utils.common_functions import read_yaml
from src.logDet.entity.config_entity import DataParams
import os


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH):
        self.config = read_yaml(Path(config_filepath))
    
    def get_data_params(self) -> DataParams:
        data_params = self.config.data_params
        data_params_config = DataParams(
            path_to_data=Path(data_params.path_to_data),
            hdfs_file_name=data_params.hdfs_file_name,
            log_file_name=data_params.log_file_name,
            labels_file_name=data_params.labels_file_name,
            log_file_path= Path(os.path.join(data_params.path_to_data, data_params.hdfs_file_name, "raw", data_params.log_file_name)),
            label_file_path=Path(os.path.join(data_params.path_to_data, data_params.hdfs_file_name, "raw", data_params.labels_file_name)),
            test_ratio=data_params.test_ratio,
            train_ratio=data_params.train_ratio,
            train_sessions_path=Path(os.path.join(data_params.path_to_data, data_params.hdfs_file_name, "processed", "train_sessions.pkl")),
            test_sessions_path=Path(os.path.join(data_params.path_to_data, data_params.hdfs_file_name, "processed", "test_sessions.pkl")),
            random_partition=data_params.random_partition,
            train_anomaly_ratio=data_params.train_anomaly_ratio
        ) 
        return data_params_config