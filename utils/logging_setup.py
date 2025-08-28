import logging, os

def setup_logger(name="bot", level=logging.INFO, path=os.path.join("memory", "bot.log")):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = logging.FileHandler(path)
    fh.setFormatter(fmt); fh.setLevel(level)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt); sh.setLevel(level)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger
