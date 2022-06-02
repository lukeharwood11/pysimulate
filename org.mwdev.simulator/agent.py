from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    """
    - Abstract agent class
    """
    @abstractmethod
    def update(self, inputs, keys_pressed=None) -> int:
        """
        - Given input from the simulation make a decision
        :param inputs:
        :param keys_pressed:
        :return direction: int [0 - num_outputs)
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: nothing
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: nothing
        """
        pass
