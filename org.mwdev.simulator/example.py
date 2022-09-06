from abc import ABC

from agent import Agent
from simulation import Simulation
from vehicle import Vehicle, Sensor, SensorBuilder
from pygame import (K_UP, K_DOWN, K_LEFT, K_RIGHT, transform)
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
            os.path.join("assets", "track.png"), \
            os.path.join("assets", "track-rewards.png")


class Car(Vehicle, ABC):

    def __init__(self, driver, sensor_depth, debug=False, acceleration_multiplier=.5):
        super(Car, self).__init__(
            image_path=os.path.join("assets", "grey-car.png"),
            driver=driver,
            scale=1,
            debug=debug,
            sensor_depth=sensor_depth
        )
        self.acceleration_multiplier = acceleration_multiplier
        self.model_path = os.path.join("assets", "models")

    def configure_image(self):
        self.image = transform.rotate(self.image, -90)
        self.image = transform.smoothscale(self.image, (34, 17))

    def save_car(self):
        """
        - Should save the model of the driver and any other important information
        :return: None
        """
        if self.driver is not None:
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)
            self.driver.save_model(self.model_path)

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

    def accelerate(self):
        """
        Accelerate the car
        :return: None
        """
        if self.velocity.speed < self.max_speed:
            self.velocity.speed += self.acceleration_multiplier

    def turn(self, left=False, right=False):
        if left:
            self.velocity.turn(self.velocity.speed * .6)
        if right:
            self.velocity.turn(self.velocity.speed * -.6)

    def brake(self):
        """
        Slow down the car or stop if the speed is less than a threshold value
        :return: None
        """
        if self.velocity.speed > 1:
            self.velocity.speed -= self.acceleration_multiplier
        else:
            self.velocity.speed = 0

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
        accel = False
        if direction.count(0) > 0:
            self.turn(left=True)
        if direction.count(1) > 0:
            self.accelerate()
            accel = True
        if direction.count(2) > 0:
            self.turn(right=True)
        if direction.count(3) > 0:
            self.brake()
        if not accel:
            self.deccelerate()

    def deccelerate(self):
        self.velocity.speed = .98 * self.velocity.speed


class GameControlDriver(Agent, ABC):

    def __init__(self, num_inputs, num_outputs):
        """
        Default controls for human driver
        :param num_inputs: the number of inputs (sensors/collision/reward)
        :param num_outputs: the number of outputs (driver-actions/left/right/etc.)
        """
        super().__init__(num_inputs, num_outputs)

    def update(self, inputs, reward_collision=False, wall_collision=False, keys_pressed=None) -> list[int]:
        """
        - Encode the inputs to integers 0 - 3
        :param wall_collision: n/a
        :param reward_collision: n/a
        :param inputs: the input from the car sensors (n/a)
        :param keys_pressed: the keys pressed from the user
        :return: a list of output encodings (0 - 3) representing requested movement
        """
        ret = []
        if keys_pressed[K_LEFT]:
            ret.append(0)
        if keys_pressed[K_UP]:
            ret.append(1)
        if keys_pressed[K_RIGHT]:
            ret.append(2)
        if keys_pressed[K_DOWN]:
            ret.append(3)
        return ret

    def save_model(self, path):
        """
        do nothing
        :param path: n/a
        :return: n/a
        """
        pass

    def load_model(self, path):
        """
        do nothing
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
        acceleration_multiplier=.5
    )

    simulation = DefaultSimulation(
        debug=True,
        fps=60,  # None means simulation fps is not tracked (Suggested for training)
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
    # Create Sensors
    sb = SensorBuilder(
        sim=simulation,
        depth=500,
        default_value=None,
        color=(255, 0, 0),
        width=2,
        pointer=True,
        car_size=car.image.get_size()
    )
    sensors = sb.generate_sensors(sensor_range=(-90, 90, 5))
    # sensors = sb.generate_sensors([0])
    # Attach sensors to car
    car.init_sensors(sensors=sensors)
    # Throw driver in the vehicle
    car.driver = driver
    simulation.simulate()


if __name__ == "__main__":
    main()
