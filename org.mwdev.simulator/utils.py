from enum import Enum


class LogLevel(Enum):
    DEBUG = 0,
    TRACE = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4


class Logger:

    def __init__(self, log_level:LogLevel):
        self.log_level = log_level

    def log(self, log_level: LogLevel, message:str):
        if log_level >= self.log_level:
            print(message)

