from abc import ABC, abstractmethod

import numpy as np
import pygame
from pygame import image
from pygame.sprite import Sprite
from pygame import transform
from agent import Agent
from vector import Vector, Velocity
from simulation import Simulation


class Vehicle(ABC, Sprite):
    """
    The car class is responsible for handling the inputs
    and moving through the simulation
    """

    def __init__(self,
                 image_path: str = None,
                 driver: Agent = None,
                 scale: int = 1,
                 debug: bool = False,
                 sensor_depth: int = 100):
        # public attributes
        super().__init__()
        self.driver = driver
        self.death_count = 0
        self.sensors: list[Sensor] = []
        # private attributes
        self._image_path = image_path
        # cache each rotation upon initialization to increase efficiency
        self._image_angle_cache = []
        self._scale = scale
        self._debug = debug
        self._sensor_depth = sensor_depth
        # vehicle info
        self.velocity = Velocity(x=0, y=0, angle=0)

    def init_car_image(self):
        self.image = image.load(self._image_path).convert()
        # todo optimize image processing
        # for i in range(359):
        #     pass

    # Unimplemented Feature
    def rotate_center(self, angle):
        """
        Called at initialization to populate the self._image_angle_cache with rotated images
        :param angle: the angle to add to cache
        :return: nothing
        """
        rotated_image = transform.rotate(self.image, angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=self.rect.topleft).center)
        self._image_angle_cache.append(rotated_image)

    def blit_rotate_center(self, window, top_left):
        """
        Rotate the image at the center and blit to the 'window' surface
        :param window: pygame.surface.Surface
        :param top_left: the top left of the vehicle
        :return: nothing
        """
        rotated_image = transform.rotate(self.image, self.velocity.angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=top_left).center)
        self.image = rotated_image
        window.blit(rotated_image, new_rect.topleft + (20, 20))

    @staticmethod
    def scale(img, factor):
        nw = img.get_rect().width * factor
        nh = img.get_rect().height * factor
        n_image = transform.scale(img, (nw, nh))
        return n_image

    def init_sensors(self, sensors):
        """
        - Initializes the sensors of the vehicle
        :param sensors:
        :return:
        """
        self.sensors = sensors

    def update_sensors(self, window, simulation: Simulation):
        """
        Calculate the values for the sensors of the vehicle
        :param window: the pygame window
        :param simulation: the simulation that the car is functioning in
        :return: nothing
        """
        pass

    def rotate(self, left=False, right=False):
        """
        rotate the vehicle and update the image (update the velocity 'angle' attribute)
        :param left: whether the car is turning left
        :param right: whether the car is turning right
        :return: nothing
        """
        pass

    @abstractmethod
    def save_car(self):
        """
        - Should save the model of the driver and any other important information
        :return: nothing
        """
        pass

    @abstractmethod
    def reset(self):
        """
        - Resets the car properties so it is ready for another episode
        :return: nothing
        """
        pass

    @abstractmethod
    def step(self, reward: bool, collision: bool, keys_pressed):
        """
        - Given the reward, collision info and the current input from the sensors, move the car
        :param keys_pressed: pygame keys pressed
        :param reward: whether the car is over a reward
        :param collision: whether the car is over a collision
        :return: nothing
        """
        pass


class Sensor:

    def __init__(self, sensor_depth: int,
                 angle: int,
                 default_val: int = None,
                 offset: np.array = np.array([0, 0]),
                 line_width: int = 2,
                 line_color: (int, int, int) = (255, 0, 0)
                 ):
        """
        :param sensor_depth: The length of the sensor
        :param angle: the angle (relative to the agent)
        :param default_val: the value of the sensor if the sensor doesn't encounter an object.
                            If left None, the default_val param is set to the sensor_depth
        :param offset: the value to offset the creation of the sensor (assuming the sensor will originate from the
                       top-left corner of the vehicle)
        :param line_width: the width of the sensor line when visible
        :param line_color: the color of the sensor line when visible (default color RED)
        """
        self.sensor_depth = sensor_depth
        self.angle = angle
        self.default_val = default_val
        if default_val is None:
            self.default_val = self.sensor_depth
        # surface will be initialized later
        self.surf = None
        self.offset = offset
        self.line_width = line_width
        self.line_color = line_color

    def generate_image(self):
        pass

    def update(self, window: pygame.surface.Surface, simulation: Simulation):
        """
        Update the surface and the value of the sensor
        :param window: pygame window
        :param simulation: Simulation
        :return: the line coordinates as a tuple
        """
        car = simulation.car

        # update the value of the surface
        surface = pygame.surface.Surface(window.get_size())
        x1 = car.velocity.x + self.offset[0]
        y1 = car.velocity.y + self.offset[1]
        x2, y2 = car.velocity.get_transform_pos(self.sensor_depth, self.angle, offset=self.offset)
        pygame.draw.line(surface, self.line_color, (x1, y1), (x2, y2), width=self.line_width)
        self.surf = surface

        # update the value of the sensor
        self.update_value(simulation=simulation)
        # return the line coordinates
        return x1, y1, x2, y2

    def update_value(self, simulation):
        """
        Check for collisions between the simulation track_border and the sensor surface
        and update the value of the sensor as either the default_val (no collision) or
        the distance between the collision and the vehicle
        Note: Error will be thrown if the track_border is None
        :param simulation: Simulation
        :return: nothing
        """
        track_mask = simulation.track_mask
        mask = pygame.mask.from_surface()

