import torch
import optuna
import dgllife
import dgl
import platform
import logging
from datetime import datetime


def setup_logger(log_filename):
    # 配置日志记录器，使用FileHandler来控制日志文件的写入模式
    log_formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=log_filename, mode='w')
    file_handler.setFormatter(log_formatter)

    # 获取默认的日志记录器并添加文件处理器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 记录硬件和软件信息
    logger.info(f"# Author: Ben Gao")
    logger.info(f"# Shanghai Institute of Organic Chemistry, ")
    logger.info(f"# University of Chinese Academy of Sciences, ")
    logger.info(f"# Chinese Academy of Sciences")
    logger.info(f"# gaoben20@mails.ucas.ac.cn")
    logger.info(f"====================================================")
    logger.info(f"Information on computing hardware and main software")
    logger.info(f"====================================================")
    logger.info(f"Python Version: {platform.python_version()}")
    logger.info(f"Torch Version: {torch.__version__}")
    logger.info(f"DGL Version: {dgl.__version__}")
    logger.info(f"DGL LifeSci Version: {dgllife.__version__}")
    logger.info(f"Optuna Version: {optuna.__version__}")
    logger.info(f"====================================================")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Memory: {gpu_props.total_memory / 1e9:.2f} GB")
    else:
        logger.info("GPU: None")
        
    logger.info(f"CPU: {platform.processor()}")

def log_results_with_time_and_date(model_name, best_trial, y_train, r2_train, rmse_train, mae_train, y_test, r2_test, rmse_test, mae_test, training_start_time, training_end_time, training_time):
    logger = logging.getLogger()
    # 记录开始时间
    logger.info(f"====================================================")
    logger.info(f"Hyperparametric optimisation results for the model")
    logger.info(f"====================================================")
    logger.info(f'Model Name: {model_name}')
    logger.info(f'Model Parameter:')
    
    logger.info('  Params: ')
    for key, value in best_trial.params.items():
        logger.info(f'    {key}: {value}')
    logger.info(f"====================================================")
    logger.info(f"Size of the train dataset: {len(y_train)}")
    logger.info(f"Size of the test dataset: {len(y_test)}")
    logger.info(f'Performance of the model:')
    # Training Set Metrics
    logger.info(f'Training Set Metrics:')
    logger.info(f'  R2: {r2_train}')
    logger.info(f'  RMSE: {rmse_train}')
    logger.info(f'  MAE: {mae_train}')

    # Test Set Metrics
    logger.info(f'Test Set Metrics:')
    logger.info(f'  R2: {r2_test}')
    logger.info(f'  RMSE: {rmse_test}')
    logger.info(f'  MAE: {mae_test}')
    logger.info(f'Total loss on the train set: {best_trial.value}')
    logger.info(f"====================================================")
    # 记录结束时间和总时间
    logger.info(f'Start Time: {training_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'End Time: {training_end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Training Time: {training_time:.2f} seconds')
    logger.info(f"====================================================")