"""
    Dataloader for log data.
    This module provides functions to load log data from different datasets.

Authors:
    Amine MEKKI <amine.mekki.contact@gmail.com>
"""

import re
from collections import OrderedDict, defaultdict
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from src.logDet.utils.common_functions import decision, dump_pickle
from src.logDet import logger

def process_hdfs(
    log_file: str,
    label_file: str,
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    train_anomaly_ratio: float = 1.0,
    random_partition: bool = False,
    train_session_path: Optional[Path] = None,
    test_session_path: Optional[Path] = None,
    **kwargs
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Load HDFS structured log into train and test data.

    Arguments
    ---------
        log_file : str
            Path to the HDFS log file.
        label_file : str
            Path to the label file.
        train_ratio : float, optional
            The ratio of training data. Default is None.
        test_ratio : float, optional
            The ratio of testing data. Default is None.
        train_anomaly_ratio : float, optional
            The ratio of anomalies in training data. Default is 1.
        random_partition : bool, optional
            Whether to randomly partition the data. Default is False.
        train_session_path : Path, optional
            Path to save the training sessions.
        test_session_path : Path, optional
            Path to save the test sessions.

    Returns
    -------
        session_train : dict
            Training data sessions.
        session_test : dict
            Testing data sessions.
    """
    logger.info(f"Loading HDFS logs from {log_file}.")
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)

    logger.debug(f"Log data shape: {struct_log.shape}")
    logger.debug(f"Label data shape: {label_data.shape}")
    
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))

    session_dict = parse_log_file(struct_log)
    assign_labels(session_dict, label_data_dict)
    
    session_train, session_test = split_data(
        session_dict, label_data_dict, train_ratio, test_ratio, train_anomaly_ratio, random_partition
    )
    if train_session_path:
        dump_pickle(session_train, train_session_path)
    
    if test_session_path:
        dump_pickle(session_test, test_session_path)

    return session_train, session_test

def parse_log_file(struct_log: pd.DataFrame) -> Dict[str, Dict]:
    """
    Parse the structured log file into sessions.

    Arguments
    ---------
        struct_log : pd.DataFrame
            The structured log data.

    Returns
    -------
        session_dict : dict
            Parsed sessions from the log data.
    """
    logger.info("Parsing log file into sessions.")

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}

    for row in struct_log.itertuples(index=False):
        blkId_list = re.findall(r"(blk_-?\d+)", getattr(row, "Content"))
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in session_dict:
                session_dict[blk_Id] = defaultdict(list)
            session_dict[blk_Id]["templates"].append(getattr(row, "EventTemplate"))
            
    logger.debug(f"Parsed {len(session_dict)} sessions from the log file.")
    return session_dict

def assign_labels(session_dict: Dict[str, Dict], label_data_dict: Dict[str, int]) -> None:
    """
    Assign labels to the parsed sessions.

    Arguments
    ---------
        session_dict : dict
            Parsed sessions from the log data.
        label_data_dict : dict
            Dictionary mapping block IDs to labels.
    """
    logger.info("Assigning labels to sessions.")
    for blk_Id in session_dict.keys():
        session_dict[blk_Id]["label"] = label_data_dict[blk_Id]
    logger.debug("Labels assigned to all sessions.")

def split_data(
    session_dict: Dict[str, Dict],
    label_data_dict: Dict[str, int],
    train_ratio: Optional[float],
    test_ratio: Optional[float],
    train_anomaly_ratio: float,
    random_partition: bool
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Split the data into training and testing sets.

    Arguments
    ---------
        session_dict : dict
            Parsed sessions from the log data.
        label_data_dict : dict
            Dictionary mapping block IDs to labels.
        train_ratio : float, optional
            The ratio of training data.
        test_ratio : float, optional
            The ratio of testing data.
        train_anomaly_ratio : float
            The ratio of anomalies in training data.
        random_partition : bool
            Whether to randomly partition the data.

    Returns
    -------
        session_train : dict
            Training data sessions.
        session_test : dict
            Testing data sessions.
    """
    session_idx = list(range(len(session_dict)))
    if random_partition:
        logger.info("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    session_labels = np.array([label_data_dict[session_id] for session_id in session_ids])

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if session_dict[k]["label"] == 0 or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }

    session_test = {k: session_dict[k] for k in session_id_test}

    train_anomaly = 100 * sum(v["label"] for v in session_train.values()) / len(session_train)
    test_anomaly = 100 * sum(v["label"] for v in session_test.values()) / len(session_test)

    logger.info(f"# train sessions: {len(session_train)} ({train_anomaly:.2f}%)")
    logger.info(f"# test sessions: {len(session_test)} ({test_anomaly:.2f}%)")

    return session_train, session_test