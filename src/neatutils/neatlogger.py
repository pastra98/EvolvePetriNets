import logging

def fs_compatible_time(dt) -> str:
    return dt.strftime('%m-%d-%Y_%H-%M-%S')


def get_logger(log_path: str, logname: str, send_to_console: bool) -> logging.Logger:
    # set up main logger for the entire execution
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG) # root level of logger, handlers cannot go deeper, default for all handlers
    # set level of console handler, only relevant for run loggers when printing gen info
    c_main_handler = logging.StreamHandler()
    if send_to_console:
        c_main_handler.setLevel(logging.DEBUG) # means that gen info will be printed
    else:
        c_main_handler.setLevel(logging.INFO) # means that only a new gen will be printed
    # file handler always writes in debug mode, meaning gen info will be included
    f_main_handler = logging.FileHandler(f"{log_path}/{logname}.log", mode="w")
    # Create formatters and add it to handlers
    lformat = logging.Formatter(
        "%(process)d - %(asctime)s - %(name)s - %(levelname)s\n%(message)s\n"
    )
    c_main_handler.setFormatter(lformat)
    f_main_handler.setFormatter(lformat)
    # Add handlers to the logger
    logger.addHandler(c_main_handler)
    logger.addHandler(f_main_handler)
    return logger
