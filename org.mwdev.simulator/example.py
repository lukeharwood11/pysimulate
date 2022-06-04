from abc import ABC

from agent import Agent
from simulation import Simulation
from vehicle import Vehicle, Sensor
from pygame import (K_UP, K_DOWN, K_LEFT, K_RIGHT)
import os


class DefaultSimulation(Simulation, ABC):

    def __init__(self,
                 debug=True,
                 fps=None,
                 num_episodes=None,
                 caption: str = None,
                 car: Vehicle = None,
                 track_offset=(0, 0),
                 screen_size=(1400, 800),
                 track_size=(1400, 800)):
        super(DefaultSimulation, self).__init__(debug=debug,
                                                fps=fps,
                                                num_episodes=num_episodes,
                                                caption=caption,
                                                car=car,
                                                track_offset=track_offset,
                                                screen_size=screen_size,
                                                track_size=track_size)

    def init_car_start_pos(self):
        """
        sets the start position of the car
        :return:
        """
        self._start_pos = (800, 120)

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
            os.path.join("assets", "track.png"), \
            os.path.join("assets", "track-rewards.png")


class Car(Vehicle, ABC):

    def __init__(self, driver, sensor_depth, debug=False, acceleration_multiplier=1.2):
        super(Car, self).__init__(
            image_path=os.path.join("assets", "car.png"),
            driver=driver,
            scale=1,
            debug=debug,
            sensor_depth=sensor_depth
        )
        self.acceleration_multiplier = acceleration_multiplier

    def save_car(self):
        """
        - Should save the model of the driver and any other important information
        :return: None
        """
        pass

    def reset(self):
        """
        - Resets the car properties, so it is ready for another episode
        :return: None
        """
        pass

    def accelerate(self):
        """
        :return:
        """
        self.velocity.speed += self.acceleration_multiplier
        pass

    def brake(self):
        """
        :return:
        """
        pass

    def step(self, reward: bool, collision: bool, keys_pressed):
        """
        - Given the reward, collision info and the current input from the sensors, move the car
        - In this implementation of step() the inputs consist of the sensor depths with the last two values of inputs
        being collision (input[-2]) and reward (input[-1]) with 1 being a collision/reward True and 0 being False
        :param keys_pressed: pygame keys pressed
        :param reward: whether the car is over a reward
        :param collision: whether the car is over a collision
        :return: None
        """
        i = self.get_input()
        i.extend([1 if collision else 0, 1 if reward else 0])
        direction = self.driver.update(inputs=i, keys_pressed=keys_pressed)
        if direction == 0:
            self.turn(left=True)
        elif direction == 1:
            self.accelerate()
        elif direction == 2:
            self.turn(right=True)
        elif direction == 3:
            self.brake()


class GameControlDriver(Agent, ABC):

    def __init__(self, num_inputs, num_outputs):
        """
        Default controls for human driver
        :param num_inputs: the number of inputs (sensors/collision/reward)
        :param num_outputs: the number of outputs (driver-actions/left/right/etc.)
        """
        super().__init__(num_inputs, num_outputs)

    def update(self, inputs, keys_pressed=None) -> int:
        if keys_pressed[K_LEFT]:
            return 0
        if keys_pressed[K_UP]:
            return 1
        if keys_pressed[K_RIGHT]:
            return 2
        if keys_pressed[K_DOWN]:
            return 3

    def save_model(self, path):
        """
        do None
        :param path: n/a
        :return: n/a
        """
        pass

    def load_model(self, path):
        """
        do None
        :param path: n/a
        :return: n/a
        """
        pass


def main():
    NUM_SENSORS = 10
    car = Car(
        driver=None,
        sensor_depth=200,
        debug=True,
        acceleration_multiplier=1.2
    )
    simulation = DefaultSimulation(
        debug=True,
        fps=60,
        num_episodes=None,
        caption="Default Simulation",
        car=car,
        track_offset=(0, 0),
        screen_size=(1400, 800),
        track_size=(1400, 800)
    )
    driver = GameControlDriver(
        num_inputs=NUM_SENSORS,
        num_outputs=4
    )
    car.driver = driver
    simulation.simulate()


if __name__ == "__main__":
    main()
