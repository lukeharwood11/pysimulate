from abc import ABC, abstractmethod


class Agent(ABC):
    """
    - Abstract agent class
    """
    @abstractmethod
    def update(self, inputs, keys_pressed=None):
        """
        - Given input from the simulation make a decision
        :param inputs:
        :param keys_pressed:
        :return:
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path:
        :return:
        """
        pass
