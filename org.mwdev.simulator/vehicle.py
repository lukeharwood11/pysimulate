from abc import ABC, abstractmethod

import numpy as np
import pygame
from pygame import image, draw
from pygame.sprite import Sprite
from pygame import transform
from sklearn.preprocessing import normalize

from agent import Agent
from vector2d import Vector2D, Velocity, Angle
from simulation import Simulation


class Vehicle(ABC):
    """
    The car class is responsible for handling the inputs
    and moving through the simulation
    """

    def __init__(self,
                 num_outputs: int,
                 image_path: str = None,
                 driver: Agent = None,
                 scale: int = 1,
                 debug: bool = False,
                 sensor_depth: int = 100,
                 max_speed: int = 20,
                 normalize: bool = True):
        # public attributes
        super().__init__()
        self.num_outputs = num_outputs
        self.driver = driver
        self.death_count = 0
        self.sensors: list[Sensor] = []
        # private attributes
        self._normalize = normalize
        self._image_path = image_path
        # cache each rotation upon initialization to increase efficiency
        self._image_angle_cache = []
        self._scale = scale
        self._debug = debug
        self._sensor_depth = sensor_depth
        # vehicle info
        self.velocity = Velocity(x=0, y=0, angle=0)
        self.odometer = 0
        self.max_speed = max_speed
        self.current_image = None
        self.image = None
        self.init_car_image()

    def _get_vehicle_input(self):
        """
        Overload to add other inputs
        :return: a numpy array of the values from the sensors
        """
        np_array = np.array([sensor.value for sensor in self.sensors])
        norm = np.linalg.norm(np_array)
        return np_array/norm if self._normalize else np_array

    def print_sensor_values(self):
        for i, sensor in enumerate(self.sensors):
            print("Sensor {}:".format(i), sensor.value)

    def init_car_image(self):
        self.image = image.load(self._image_path)
        self.configure_image()
        # todo optimize image processing
        # for i in range(359):
        #     self._image_angle_cache.append(self.rotate_center(i))

    # Unimplemented Feature
    def rotate_center(self, angle):
        """
        Called at initialization to populate the self._image_angle_cache with rotated images
        :param angle: the angle to add to cache
        :return: None
        """
        rotated_image = transform.rotate(self.image, angle)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=self.rect.topleft).center)
        self._image_angle_cache.append(rotated_image)

    def blit_rotate_center(self, window, top_left):
        """
        Rotate the image at the center and blit to the 'window' surface
        :param window: pygame.surface.Surface
        :param top_left: the top left of the vehicle
        :return: None
        """
        # TODO implement cache using angle
        rotated_image = transform.rotate(self.image, self.velocity.angle.value).convert_alpha()
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft=top_left).center)
        self.current_image = rotated_image
        x, y = new_rect.topleft[0] + self.image.get_width() / 2, new_rect.topleft[1] + self.image.get_height() / 2
        window.blit(rotated_image, (x, y))

    def update(self, simulation: Simulation):
        """
        update necessary data and send back to simulation
        :param simulation: the simulation the vehicle exists in
        :return: None
        """
        window = simulation.window
        # account for reoccurring events (such as velocity update)
        self.update_pos()
        self.odometer += self.velocity.speed  # update odometer as the distance moved each step
        self.blit_rotate_center(window, (self.velocity.x, self.velocity.y))

    def update_pos(self):
        self.velocity.transform()

    @staticmethod
    def scale(img, factor):
        nw = img.get_rect().width * factor
        nh = img.get_rect().height * factor
        n_image = transform.scale(img, (nw, nh))
        return n_image

    def init_sensors(self, sensors):
        """
        - Initializes the sensors of the vehicle
        :param sensor_builder:
        :param sensors:
        :return:
        """
        self.sensors = sensors

    def update_sensors(self, window, simulation: Simulation):
        """
        Calculate the values for the sensors of the vehicle
        :param window: the pygame window
        :param simulation: the simulation that the car is functioning in
        :return: None
        """
        # get (x1,y1,x2,y2) tuples for all sensor positions
        s = [sensor.update(window=window, simulation=simulation) for sensor in self.sensors]
        if self._debug:
            self.display_sensor(car_pos=simulation.get_vehicle_image_position(), window=window)

    # TODO: Display cannot just pull from one collision point
    def display_sensor(self, car_pos, window: pygame.surface.Surface):
        for s in self.sensors:
            if s.coords is not None:
                pygame.draw.line(surface=window,
                                 color=s.line_color,
                                 start_pos=(car_pos),
                                 end_pos=(s.coords[0] if s.collision_point is None else s.collision_point[0],
                                          s.coords[1] if s.collision_point is None else s.collision_point[1]),
                                 width=s.line_width)

    def get_input(self) -> list[int]:
        """
        retrieve the inputs for the vehicle as a list of (normalized) floats representing the distance
        of the vehicle to the nearby walls.
        :return:
        """
        return [s.value / self._sensor_depth if s != 0 else 0 for s in self.sensors]

    @abstractmethod
    def configure_image(self):
        """
        - perform transformations to vehicle
        - configure 'zero-angle-position' and sizing
        :return:
        """
        pass

    @abstractmethod
    def accelerate(self):
        """
        Make the car go faster, or don't
        :return:
        """
        pass

    @abstractmethod
    def brake(self):
        """
        Make the car go slower, or don't
        :return:
        """
        pass

    @abstractmethod
    def turn(self, left=False, right=False):
        """
        - For custom operation override this method
        rotate the vehicle and update the image (update the velocity 'angle' attribute)
        :param left: whether the car is turning left
        :param right: whether the car is turning right
        :return: None
        """
        pass

    @abstractmethod
    def save_car(self):
        """
        - Should save the model of the driver and any other important information
        :return: None
        """
        pass

    @abstractmethod
    def reset(self, simulation: Simulation):
        """
        - Resets the car properties, so it is ready for another episode
        :return: None
        """
        pass

    @abstractmethod
    def step(self, reward: bool, collision: bool, keys_pressed):
        """
        - Given the reward, collision info and the current input from the sensors, move the car
        :param keys_pressed: pygame keys pressed
        :param reward: whether the car is over a reward
        :param collision: whether the car is over a collision
        :return: None
        """
        pass

    @abstractmethod
    def get_external_inputs(self):
        """
        :return: the number of inputs (not including sensors) that are used
        i.e. velocity
        """
        pass


class SensorBuilder:

    def __init__(self, depth: int, sim: Simulation, default_value=None, color=(0, 0, 0), width=2, pointer=True):
        """
        Convenience class for building sensors
        :param depth: the sensor depth
        :param color: the color of the sensor
        :param width: the width of the sensor
        :param pointer: whether the sensor has an end point
        """
        self.simulation = sim
        self.depth = depth
        self.color = color
        self.width = width
        self.pointer = pointer
        self.default_value = default_value
        self.num_sensors = -1
        self.masks = []  # list of 359 masks representing all angles
        self._sensor_position = np.zeros((360, self.depth, 2))  # list of 359 sensor positions
        self.sensor_depth_array = np.array([self.depth, self.depth])
        self.mask_debug_images = []
        self.offset = 0, 0
        self.generate_sensor_points()

    def get_sensor_pos(self, car_angle, sensor_angle):
        """

        :param car_angle: the angle of the vehicle
        :param sensor_angle: the angle of the sensor relative to the car
        :param offset:
        :return: An array of shape ([sensor_depth], 2)
        """
        # Get the sensor position given the current angle and position of the car
        sensor_absolute_angle = car_angle + sensor_angle
        sensor_pos = self._sensor_position[sensor_absolute_angle.round().value]
        return sensor_pos + self.simulation.get_vehicle_image_position()

    def get_sensor_offset(self):
        """
        :return: get offset for sensor masks
        """
        car_v = self.simulation.car.velocity
        return (car_v.x - self.depth + self.simulation.car.image.get_width() / 2,
                car_v.y - self.depth + self.simulation.car.image.get_height() / 2)

    def generate_sensors(self, sensor_angles: list[int] = None, sensor_range=None) -> list:
        """
        - Generate a cached list of sensors
        :param sensor_range: an optional parameter allowing for sensor values to be defined by a range
        :param sensor_angles - list of angles for sensors
        :param range - (start_angle, end_angle, step)
        """
        sensors = []
        if range is not None and sensor_angles is None:
            sensor_angles = range(*sensor_range)
        for angle in sensor_angles:
            self.num_sensors = len(sensor_angles)
            sensors.append(
                Sensor(
                    sb=self, sensor_depth=self.depth, angle=angle, default_val=self.default_value,
                    offset=np.array(self.offset), line_width=self.width, line_color=self.color, pointer=self.pointer
                )
            )
        return sensors

    def generate_sensor_points(self):
        """
        - Generate a cached list of sensor masks to be used to receive input
        """
        for angle in range(360):
            a = Angle.create(angle)
            for step in range(self.depth):
                x, y = Vector2D.point_at_angle(step, angle=a, offset=np.array(self.offset))
                x2, y2 = x, -y
                self._sensor_position[angle, step] = np.array([x2, y2])


class Sensor:

    def __init__(self, sb: SensorBuilder, sensor_depth: int,
                 angle: int,
                 default_val: int = None,
                 offset: np.array = np.array([0, 0]),
                 line_width: int = 2,
                 line_color: (int, int, int) = (255, 0, 0),
                 pointer: bool = False
                 ):
        """
        :param sb: SensorBuilder that created the sensor (issue #12)
        :param sensor_depth: The length of the sensor
        :param angle: the angle (relative to the agent)
        :param default_val: the value of the sensor if the sensor doesn't encounter an object.
                            If left None, the default_val param is set to the sensor_depth.
        :param offset: the value to offset the creation of the sensor (assuming the sensor will originate from the
                       top-left corner of the vehicle)
        :param line_width: the width of the sensor line when visible
        :param line_color: the color of the sensor line when visible (default color RED)
        """
        self.sensor_builder = sb
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
        self.value = 1  # sensor_depth / sensor_depth
        self.coords = np.zeros(2)
        self.pointer = pointer
        self.collision_point = None

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
        # coords has a shape of ([sensor_depth], 2)
        sensor_array = self.sensor_builder.get_sensor_pos(car.velocity.angle, Angle(self.angle))

        # update the value of the sensor
        self.update_value(sensor_array=sensor_array, simulation=simulation)
        # np.array[x2, y2]
        self.coords = sensor_array[self.sensor_depth - 1]
        # return the line coordinates
        # x2, y2
        return self.coords[0], self.coords[1]

    def update_value(self, sensor_array, simulation):
        """
        Check for collisions between the simulation track_border and the sensor surface
        and update the value of the sensor as either the default_val (no bit) or
        the distance between the bit and the vehicle
        Note: Error will be thrown if the track_border is None
        :param simulation: Simulation
        :return: None
        """
        car_v: Velocity = simulation.car.velocity
        border_mask = simulation.border_mask
        angle = car_v.angle + Angle(self.angle)
        # print("car angle: ", car_v.angle)
        # print("mask angle: ", angle)
        for point in sensor_array:
            valid_point = 0 <= point[0] < simulation.track_border_width \
                          and 0 <= point[1] < simulation.track_border_height
            if not valid_point: continue
            bit = border_mask.get_at((point[0], point[1]))
            if bit == 1:
                if self.pointer:
                    x, y = point[0] - 5 / 2, point[1] - 5 / 2
                    draw.circle(simulation.window, (255, 255, 255), (x, y), 5)
                self.value = car_v.distance_between(
                    other=Vector2D(x=point[0], y=point[1]),
                    offset=(simulation.car.image.get_width() / 2,
                            simulation.car.image.get_height() / 2)
                )
                self.collision_point = point
                break
            else:
                self.value = self.default_val
                self.collision_point = None
