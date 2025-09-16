import random
import numpy as np
import torch
import logging
import sys
import os


def set_seed(seed):
    """
    设置随机种子以确保可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 适用于多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name, log_dir="./log", level=logging.INFO):
    """
    配置日志记录器

    Args:
        name (str): 日志记录器的名称。
        log_dir (str): 存放日志文件的目录。
        level (int): 日志记录的级别 (e.g., logging.INFO, logging.DEBUG)。

    Returns:
        logging.Logger: 配置好的日志记录器对象。
    """
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f'{name}.log')

    # 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建文件处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 创建流处理器，用于将日志输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # 获取/创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加处理器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
