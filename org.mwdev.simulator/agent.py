from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    """
    - Abstract agent class
    """
    @abstractmethod
    def update(self, inputs, reward_collision=False, wall_collision=False, keys_pressed=None) -> list[int]:
        """
        - Given input from the simulation make a decision
        :param wall_collision: whether the car collided with the wall
        :param reward_collision: whether the car collided with a reward
        :param inputs: sensor input as a numpy array
        :param keys_pressed: a map of pressed keys
        :return direction: int [0 - num_outputs)
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: None
        """
        pass

    @abstractmethod
    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: None
        """
        pass
