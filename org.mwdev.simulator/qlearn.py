import os
from typing import List

import numpy as np
from abc import ABC

from keras.optimizer_v2.adam import Adam
from numpy import ndarray

from agent import Agent

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


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


class QLearningParams:
    def __init__(self, wall_collision_value, reward_collision_value, other_value):
        self.wall = wall_collision_value
        self.reward = reward_collision_value
        self.other = other_value


class QLearningAgent(Agent):

    def __init__(self, alpha, alpha_decay, y, epsilon, num_sensors, num_actions, batch_size, replay_mem_max,
                 save_after=None,
                 load_latest_model=False, training_model=True, model_path=None, train_each_step=False, debug=False, other_inputs=0):
        # initialize Agent parent class
        # add one to num_inputs for current speed
        super(QLearningAgent, self).__init__(num_inputs=num_sensors+other_inputs, num_outputs=num_actions)
        # Q learning hyperparameters
        self.alpha = alpha
        self.alpha_decay = alpha_decay
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
        # build Sequential tensorflow model
        self._model = Sequential()
        self._model.add(Dense(32, input_shape=(None, self.num_inputs)))
        self._model.add(Dense(64))
        self._model.add(Dense(self.num_outputs, activation='linear'))
        self._model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))
        # Q learning rewards
        self._qlearn_params = QLearningParams(wall_collision_value=-20, reward_collision_value=20, other_value=-2)

    def update(self, inputs, reward_collision=False, wall_collision=False, keys_pressed=None) -> list[int]:
        """
        - Given input from the simulation make a decision
        :param wall_collision: whether the car collided with the wall
        :param reward_collision: whether the car collided with a reward
        :param inputs: sensor input as a numpy array
        :param keys_pressed: a map of pressed keys (ignore, n/a)
        :return direction: int [0 - num_outputs)
        """
        reward = self._get_reward(reward_collision, wall_collision)
        # Change internal states
        self._handle_collision(wall_collision)
        self._handle_reward(reward, reward_collision)
        self._handle_experience(reward, inputs)
        self._handle_training()
        actions = self._model.predict(inputs.reshape(1, self.num_inputs))
        action = np.argmax(actions)
        print("Reward", reward)
        self._current_action = action
        print(action)
        return [action]

    def _get_reward(self, reward_collision, wall_collision):
        if wall_collision:
            return self._qlearn_params.wall
        elif reward_collision:
            return self._qlearn_params.reward
        else:
            return self._qlearn_params.other

    def _handle_experience(self, reward, inputs):
        if self._current_state is not None:
            self.replay_memory.add_experience(Experience(
                current_state=self._current_state,
                current_action=self._current_action,
                resulting_reward=reward,
                resulting_state=inputs
            ))
        self._current_state = inputs

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
            self._collision_count += 1

    def _handle_reward(self, reward, reward_collision):
        if reward_collision:
            if self._rewarded_currently:
                # if the car is sitting on a reward, punish it with "other" value
                reward -= self._qlearn_params.reward - self._qlearn_params.other
            self._rewarded_currently = True
        else:
            self._rewarded_currently = False

    def _train_model(self):
        """
        - Get [self.batch_size] number of experiences and train on those experiences
        """
        batch = self.replay_memory.get_random_experiences(self.batch_size)
        X_train, y_train = np.zeros((self.batch_size, self.num_inputs)), np.zeros(self.batch_size)
        for i, experience in enumerate(batch):
            current_state = np.array(experience.current_state)  # TODO reshape
            # predict the q_values
            q_value_prediction = self._model.predict(current_state.reshape(1, self.num_inputs))
            # set the target to be what the experience actually was
            q_target = experience.res_reward
            resulting_state = np.array(experience.next_state)  # TODO reshape
            q_target = q_target + self.y * np.max(self._model.predict(resulting_state.reshape(1, self.num_inputs))[0])
            # adjust the weights (no other q_vals are impacted)
            q_value_prediction[0][experience.current_action] = q_target
            X_train[i] = current_state
            y_train[i] = q_value_prediction[0][experience.current_action]

        self._model.fit(X_train, y_train, verbose=2)

    def _save_model_increment(self):
        """
        Save the current model to a unique location representing the current iteration
        :return: None
        """
        self._model.save_model("assets/models/model_" + str(self._collision_count) + ".h5")

    def save_model(self, path):
        """
        - Save the brain of the agent to some file (or don't)
        :param path: the path to the model
        :return: None
        """
        self._model.save_weights(os.path.join(path, 'latest.h5'))

    def init_default_model_weights(self):
        self._model.load_model("assets/models/latest.h5")

    def load_model(self, path):
        """
        - Load the brain of the agent from some file (or don't)
        :param path: the path to the model
        :return: None
        """
        self._model.load_weights(path)
