from math import sin, cos, tan, sqrt, radians, degrees
import numpy as np


class Angle:

    def __init__(self, starting_val=0):
        self.value = starting_val

    def update(self, dA):
        self.value = self._calc_difference(dA)

    def _calc_difference(self, dA):
        """
        - other: Angle
        """
        value = self.value
        value += dA
        return value % 360

    def greater_than_180(self):
        return self.value > 180

    def ref(self):
        if self.value >= 180:
            return Angle(360 - self.value)
        return self

    def __add__(self, other):
        diff = self.value + other.value
        return Angle(self._calc_difference(diff))

    def __sub__(self, other):
        diff = self.value - other.value
        return Angle(self._calc_difference(diff))

    def __eq__(self, other):
        return other.value == self.value

    def __mul__(self, other):
        return Angle(other.value * self.value)

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __le__(self, other):
        return self.value <= other

    def __ge__(self, other):
        return self.value >= other

    def __str__(self):
        return "Value: {0}".format(self.value)

    def to_radians(self):
        return radians(self.value)

    def sin(self):
        return sin(radians(self.value))

    def cos(self):
        return cos(radians(self.value))

    def tan(self):
        return tan(radians(self.value))

    @staticmethod
    def create(angle):
        return Angle(angle % 360)


class Vector2D:
    """
    - Convenience class to hold 2d data
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_magnitude(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    @staticmethod
    def calc_magnitude(x, y):
        return sqrt(x**2 + y**2)

    def distance_between(self, other, offset=np.array([0, 0])) -> float or None:
        """
        :param other: Vector
        :param offset: offset of current vector
        :return: the distance between the two points or None if other is None
        """
        if other is None:
            return None
        else:
            dy = other.y - (self.y + offset[1])
            dx = other.x - (self.x + offset[0])
            return Vector2D.calc_magnitude(dx, dy)


class Velocity(Vector2D):

    def __init__(self, speed=0, x=0, y=0, angle=0):
        """
        :param x: the current x position
        :param y: the current y position
        :param angle: the current angle the object is moving in
        :param speed: the current speed
        """
        super(Velocity, self).__init__(x, y)
        self.angle = Angle(starting_val=angle)
        self.speed = speed

    def turn(self, da):
        """
        :param da: the change in the angle
        :return: None
        """
        self.angle.update(da)

    def transform(self):
        """
        :param distance:
        :return:
        """
        dx = cos(radians(self.angle.value)) * self.speed
        dy = -sin(radians(self.angle.value)) * self.speed
        self.x += dx
        self.y += dy

    def get_transform_pos(self, distance, angle: Angle, offset=np.array([0, 0])):
        """
        Projects a point [distance] length away at an [angle] relative to the current direction of the vector
        :param distance: distance of the point
        :param angle: angle relative to the current angle of the car
        :param offset: the offset of the position to account for rect positioning in pygame
        :return: (x:int, y:int) -> tuple of the coordinates of the projected position
        """
        theta = self.angle - angle
        sign = 1 if theta.greater_than_180() else -1
        theta = theta.ref()

        dy = theta.sin() * distance if 90 <= self.angle.value <= 180 or 270 < self.angle.value < 360 else theta.cos() * distance
        dx = theta.cos() * distance if 90 <= self.angle.value <= 180 or 270 < self.angle.value < 360 else theta.cos() * distance

        return self.x + (dx * sign), self.y + (dy * sign)

    def reset_velocity(self, x=0, y=0, angle=0, speed=0):
        self.x = x
        self.y = y
        self.angle = Angle.create(angle)
        self.speed = speed
