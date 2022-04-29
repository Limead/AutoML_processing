import logging
import datetime

def Logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("[%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
    
    logPath = "./ML_log/ML_log" + datetime.datetime.today().strftime("%Y%m%d") + ".log"
    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(formatter)
    
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    return logger