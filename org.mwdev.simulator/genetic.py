from abc import ABC

import numpy as np
import pygame

from agent import Agent
from simulation import Simulation
from example import Car
import os
from utils import CollisionSet

class GeneticCar(Car):

    def __init__(self, driver, debug=False, acceleration_multiplier=.5, normalize=True):
        super(GeneticCar, self).__init__(driver, debug, acceleration_multiplier, normalize)
        self.number = 0
        self.parents = []

    @staticmethod
    def generate_cars(drivers, params=None):
        if params is None:
            params = {
                'acceleration_multiplier': .5,
                'normalize': True,
                'debug': False
            }
        return np.array([GeneticCar(
            driver=driver, **params
        ) for driver in drivers])

    def reset(self, simulation):
        """
        - Resets the car properties, so it is ready for another episode
        :return: None
        """
        self.velocity.reset_velocity(
            x=simulation.start_pos[0],
            y=simulation.start_pos[1],
            angle=180,
            speed=0
        )

    def __add__(self, other):
        """
        Perform cross over with other
        :param other:
        :return:
        """
        return self.driver + other.driver

    def __lt__(self, other):
        return self.odometer < other.odometer

    def __gt__(self, other):
        return self.odometer > other.odometer

    def __eq__(self, other):
        return self.odometer == other.odometer


class GeneticAlgorithmSimulation(Simulation, ABC):

    def __init__(self,
                 debug=True,
                 fps=None,
                 num_episodes=None,
                 cars: list[GeneticCar] = None,
                 track_offset=(0, 0),
                 screen_size=(1400, 800),
                 track_size=(1400, 800),
                 batch_size=100,
                 mini_batch_size=3
                 ):
        super(GeneticAlgorithmSimulation, self).__init__(
            debug=debug,
            fps=fps,
            num_episodes=num_episodes,
            caption="Genetic Algorithm",
            car=None,
            track_offset=track_offset,
            screen_size=screen_size,
            track_size=track_size
        )
        self.mini_batch_size = mini_batch_size
        self.current_index = mini_batch_size
        self.mini_batch = cars[:mini_batch_size]
        # stores the odometer readings from cars that have "competed"
        self.batch_results = np.zeros(batch_size)
        self.batch_size = batch_size
        self.cars: np.array = cars
        self.car_death_count = 0
        self.collision_set = CollisionSet(self.mini_batch_size)

    def update_display(self, keys_pressed):
        """
        :param keys_pressed:
        :return:
        """
        self.window.fill((0, 0, 0))
        if self._track_bg is not None:
            self.window.blit(self._track_bg, self._track_offset)
        if self._track_rewards is not None and self._debug:
            self.window.blit(self._track_rewards, self._track_offset)
        self.car.update_sensors(self.window, self)
        self.handle_collision()
        # Only call reset when all cars have 'died'
        if self.collision_set.full_collision():
            # All three have collided with the wall, and we can now reset
            self.reset()
        for car, collision in zip(self.mini_batch, self.collision_set.crash_list()):
            car.step(reward=False, collision=collision, keys_pressed=keys_pressed)
            car.update(simulation=self)
        self.update_and_display_labels()
        self.op_display()

    def handle_collision(self) -> None:
        """
        Only works if the track_border is not None
        :return: whether the vehicle hit a wall
        """
        for i, car in enumerate(self.mini_batch):
            if self._track_border is not None \
                    and car is not None \
                    and car.current_image is not None:
                car_mask = pygame.mask.from_surface(car.image)
                x, y = (self.car.velocity.x + (.5 * car.image.get_width()),
                        self.car.velocity.y + (.5 * car.image.get_height()))
                col = self.border_mask.overlap(car_mask, (x, y))
                if col is not None:
                    self.collision_set.set_collision(i)
            else:
                raise Exception("Track border is none.")

    def reset(self):
        """
        resets the minibatch to begin the next one
        Called whenever the car crashes and the next mini_batch begins
        :return:
        """
        # restart the collision set
        self.collision_set.clear()
        for car in self.mini_batch:
            # Technically this is redundant # TODO remove
            car.reset(simulation=self)
        # If the batch is done
        if self.current_index >= len(self.cars):
            self.clean_up_iteration()
        else:
            next_index = self.current_index + self.mini_batch_size
            # grab the next mini_batch of cars
            self.mini_batch = self.cars[
                              self.current_index: next_index if next_index < len(self.cars) else len(self.cars) - 1]
            self.current_index += self.mini_batch_size

    def clean_up_iteration(self):
        """
        - Clean up the current iteration for the next one
        - Find the highest ranking cars, then crossover, then mutate
        :return:
        """
        self._iteration_num += 1
        sorted = np.argsort(self.batch_results)
        parent_cars = self.cars[sorted[-2:]]
        driver1, driver2 = parent_cars[0].driver, parent_cars[1].driver
        new_drivers = driver1.mutate(driver2, num_mutations=self.batch_size - 2)
        new_cars = GeneticCar.generate_cars(new_drivers, params=self.car_params)

    def init_car_start_pos(self):
        """
        sets the start position of the car
        :return:
        """
        self.start_pos = (875, 100)

    def init_track(self) -> (str, str, str):
        """
        Should set the images of the track (paths to the images):
        called in the constructor of the simulation class
        - track border
        - track bg
        - track rewards
        :return: the path to the tracks in the order 'border, background (design), rewards'
        """
        return \
            os.path.join("assets", "track-border.png"), \
            os.path.join("assets", "track.png"), None


class GeneticNeuralNetwork:

    def __init__(self):
        pass

    def cross_over(self, other):
        """

        :param other:
        :return:
        """
        pass

    def mutate(self):
        """

        :return:
        """
        pass


class GeneticAlgorithmDriver(Agent, ABC):

    def __init__(self, num_inputs, num_outputs):
        super().__init__(num_inputs, num_outputs)

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

    def mutate(self, other, num_mutations):
        """
        :param other: another driver
        :param num_mutations: the number of mutations to add to itself
        :return:
        """
        return np.array([self, other] + [self.cross_over_mutation(other) for _ in num_mutations],
                        dtype=GeneticAlgorithmDriver)

    def cross_over_mutation(self, other):
        pass

    @staticmethod
    def generate_drivers(num_drivers):
        """
        create a list of random Drivers
        :param num_drivers:
        :return:
        """
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
