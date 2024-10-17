import logging
import os
from datetime import datetime

def setup_logger(model_name, experiment_name):
    """
    Setup a logger that logs messages to both the console and a file.

    Args:
        experiment_name (str): Name of the experiment for logging.

    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # 로그 저장 디렉토리 생성
    log_dir = f'/app/paper-implementations/vision/experiments/{model_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 로그 파일 이름 설정
    log_filename = f"{log_dir}/{experiment_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

    # 로거 설정
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    # 콘솔 핸들러 설정 (출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 파일 핸들러 설정 (파일 저장)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 핸들러 등록
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# 예시로 logger 호출하기
if __name__ == '__main__':
    logger = setup_logger('test_experiment')
    logger.info('This is an info message for the experiment.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')