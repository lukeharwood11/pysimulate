import numpy as np
from abc import ABC
from agent import Agent

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, \
    Dropout
from torch.optim import Adam, SGD


class Experience:

    def __init__(self, current_state, current_action, resulting_reward, resulting_state):
        """
        Holds the memory for replay
        :param current_state: the current state of the model
        :param current_action: the action that was chosen
        :param res_reward: the resulting reward
        :param res_state: the resulting state
        """
        self.current_state = current_state
        self.current_action = current_action
        self.res_reward = resulting_reward
        self.next_state = resulting_state


class ReplayMemory:
    """
    - Serves as a queue that has a max_size
    - Max size is by default None and should be added by the user
    """

    def __init__(self, max_size=None):
        self._max_size = max_size
        self._arr = []
        self._size = 0

    def add_experience(self, ex: Experience):
        if self._size == self._max_size and self._max_size is not None:
            self._arr.pop(0)
            self._size -= 1
        self._arr.append(ex)
        self._size += 1

    def get_random_experiences(self, num_samples):
        np.random.shuffle(self._arr)
        return self._arr[0:num_samples if num_samples < self._size else self._size]


class CNN(Module):

    def __init__(self, num_inputs, num_outputs, learning_rate=0.001):
        super(CNN, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.layer1 = Sequential(
            torch.nn.Conv2d(num_inputs, 64, kernel_size=3, stride=1, padding=1),
        )

        self.layer2 = Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.layer3 = Sequential(
            torch.nn.Conv2d(32, num_outputs, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

    def get_optimizer(self):
        return Adam(params=self.parameters(), lr=self.learning_rate)


class QLearningParams:

    def __init__(self, wall_collision_value, reward_collision_value, other_value):
        self.wall = wall_collision_value
        self.reward = reward_collision_value
        self.other = other_value


class QLearningAgent(ABC, Agent):

    def __init__(self, alpha, y, epsilon, num_inputs, num_actions, batch_size, replay_mem_max, save_after=None,
                 load_latest_model=False, training_model=True, model_path=None, train_each_step=False, debug=False):
        # initialize Agent parent class
        super(QLearningAgent, self).__init__(num_inputs=num_inputs, num_outputs=num_actions)
        # Q learning hyperparameters
        self.alpha = alpha
        self.y = y
        self.epsilon = epsilon
        self.batch_size = batch_size
        # Q learning replay memory
        self.replay_memory = ReplayMemory(max_size=replay_mem_max)
        # private state
        self._current_state = None
        self._current_action = None
        self._rewarded_currently = False
        self._collision_count = 0
        # load/save/training properties
        self._save_after = save_after
        self._load_latest_model = load_latest_model
        self._training_model = training_model  # boolean
        self._model_path = model_path
        self._train_each_step = train_each_step
        # debug private attributes
        self._debug = debug
        # Q learning rewards
        self._qlearn_params = QLearningParams(wall_collision_value=-20, reward_collision_value=20, other_value=2)

    def update(self, inputs, reward_collision, wall_collision, keys_pressed=None) -> list[int]:
        """
        - Given input from the simulation make a decision
        :param wall_collision: whether the car collided with the wall
        :param reward_collision: whether the car collided with a reward
        :param inputs: sensor input as a numpy array
        :param keys_pressed: a map of pressed keys
        :return direction: int [0 - num_outputs)
        """
        reward = self._get_reward(reward_collision, wall_collision)

        # Change internal states
        self._handle_collision(wall_collision)
        self._handle_reward(reward, reward_collision)
        self._handle_training()

        return []

    def _get_reward(self, reward_collision, wall_collision):
        if wall_collision:
            return self._qlearn_params.wall
        elif reward_collision:
            return self._qlearn_params.reward
        else:
            return self._qlearn_params.other

    def _handle_training(self):
        if self._training_model:
            if self._train_each_step:
                self._train_model()

    def _handle_collision(self, wall_collision):
        if wall_collision:
            if self._collision_count % self._save_after == 0:
                self._save_model_increment()
            if self._training_model:
                self._train_model()

    def _handle_reward(self, reward, reward_collision):
        if reward_collision:
            if self._rewarded_currently:
                # if the car is sitting on a reward, punish it with "other" value
                reward -= self._qlearn_params.reward - self._qlearn_params.other
            self._rewarded_currently = True
        else:
            self._rewarded_currently = False

    def save_model_increment(self):
        """
        Save the current model to a unique location representing the current iteration
        :return: None
        """
        pass

    def _train_model(self):
        pass

    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: None
        """
        pass

    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: None
        """
        pass
