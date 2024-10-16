import logging
from multiprocessing import Manager

class ProgressHandler(logging.Handler):
    def __init__(self, progress_dict, setup_name, run_id, total_generations):
        super().__init__()
        self.progress_dict = progress_dict
        self.setup_name = setup_name
        self.run_id = run_id
        self.total_generations = total_generations

    def emit(self, record):
        if record.levelno == logging.INFO and "GEN:" in record.msg:
            current_gen = int(record.msg.split("GEN:")[1].strip())
            self.progress_dict[f"{self.setup_name}_{self.run_id}"] = current_gen / self.total_generations

def get_logger(log_path: str, logname: str, send_to_console: bool, progress_dict=None, setup_name=None, run_id=None, total_generations=None) -> logging.Logger:
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    
    if send_to_console:
        c_main_handler = logging.StreamHandler()
        c_main_handler.setLevel(logging.DEBUG)
    else:
        c_main_handler = logging.StreamHandler()
        c_main_handler.setLevel(logging.INFO)
    
    f_main_handler = logging.FileHandler(f"{log_path}/{logname}.log", mode="w")
    
    lformat = logging.Formatter(
        "%(process)d - %(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"
    )
    c_main_handler.setFormatter(lformat)
    f_main_handler.setFormatter(lformat)
    
    logger.addHandler(c_main_handler)
    logger.addHandler(f_main_handler)

    if progress_dict is not None:
        progress_handler = ProgressHandler(progress_dict, setup_name, run_id, total_generations)
        progress_handler.setLevel(logging.INFO)
        logger.addHandler(progress_handler)

    return logger

def fs_compatible_time(dt) -> str:
    return dt.strftime('%m-%d-%Y_%H-%M-%S')