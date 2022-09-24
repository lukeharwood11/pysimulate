from enum import Enum

import numpy as np


class LogLevel(Enum):
    DEBUG = 0,
    TRACE = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4


class Logger:

    def __init__(self, log_level: LogLevel):
        self.log_level = log_level

    def log(self, log_level: LogLevel, message: str):
        if log_level >= self.log_level:
            print(message)


class CollisionSet:

    def __init__(self, mini_batch_size):
        self.data = np.zeros(mini_batch_size)
        self._mini_batch_size = mini_batch_size

    def set_collision(self, index):
        self.data[index] = 1

    def crash_list(self):
        return self.data == 1

    def collision_at(self, index):
        return self.data[index] == 1

    def full_collision(self):
        return self.data == np.ones(len(self.data))

    def clear(self, mini_batch_size):
        """
        mini_batch_size can be smaller than self.mini_batch_size
        :param mini_batch_size:
        :return:
        """
        self.data = np.zeros(mini_batch_size)
