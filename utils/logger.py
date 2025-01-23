import logging
from typing import Any, Optional, List, Tuple, Dict
from termcolor import colored
from datetime import datetime

class ChainWatchLogger:
    def __init__(self, language: str = 'en'):
        self.language = language
        self.logger = logging.getLogger('chainwatch')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'chainwatch_{timestamp}.log'
        
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(
            log_filename,
            maxBytes=10*1024*1024,
            backupCount=5
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def log(self, msg: str, level: str = 'info', color: str = 'white') -> None:
        try:
            colored_msg = colored(msg, color)
            if level == 'error':
                self.logger.error(colored_msg)
            elif level == 'warning':
                self.logger.warning(colored_msg)
            elif level == 'debug':
                self.logger.debug(colored_msg)
            else:
                self.logger.info(colored_msg)
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def batch_log(self, messages: List[Tuple[str, str, str]]) -> None:
        for msg, level, color in messages:
            self.log(msg, level, color)

    def log_dict(self, data: Dict[str, Any], prefix: str = '') -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                self.log(f"{prefix}{key}:", 'info', 'white')
                self.log_dict(value, prefix + '  ')
            else:
                self.log(f"{prefix}{key}: {value}", 'info', 'white')

    def log_rate_limit(self, endpoint: str, wait_time: float) -> None:
        self.log(
            f"Rate limit reached for {endpoint}. Waiting {wait_time:.2f}s",
            level='warning',
            color='yellow'
        )

    def error(self, msg: str) -> None:
        self.log(msg, 'error', 'red')

    def warning(self, msg: str) -> None:
        self.log(msg, 'warning', 'yellow')

    def info(self, msg: str) -> None:
        self.log(msg, 'info', 'cyan')

    def debug(self, msg: str) -> None:
        self.log(msg, 'debug', 'white')
