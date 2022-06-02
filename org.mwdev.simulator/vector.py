from math import sin, cos, tan, sqrt, radians, degrees
import numpy as np


class Vector:
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
        :return: the distance between the two points
        """
        if other is None:
            return None
        else:
            dy = other.y - (self.y + offset[1])
            dx = other.x - (self.x + offset[0])
            return Vector.calc_magnitude(dx, dy)


class Velocity(Vector):

    def __init__(self, speed=0, x=0, y=0, angle=0):
        """
        :param dx: the current change in x
        :param dy: the current change in y
        :param angle: the current angle the object is moving in
        """
        super(Velocity, self).__init__(x, y)
        self.angle = angle
        self.speed = speed

    def turn(self, da):
        """
        :param da: the change in the angle
        :return: nothing
        """
        self.angle += da

    def transform(self, distance):
        """
        :param distance:
        :return:
        """
        dx = cos(radians(self.angle)) * distance
        dy = -sin(radians(self.angle)) * distance
        self.x += dx
        self.y += dy

    def get_transform_pos(self, distance, angle, offset=np.array([0, 0])):
        """
        Projects a point [distance] length away at an [angle] relative to the current direction of the vector
        :param distance: distance of the point
        :param angle: angle relative to the current angle of the car
        :param offset: the offset of the position to account for rect positioning in pygame
        :return: (x:int, y:int) -> tuple of the coordinates of the projected position
        """
        if angle > 0:
            return (self.x - sin(radians(self.angle - angle)) * distance) + offset[0], \
                   (self.y - cos(radians(self.angle - angle)) * distance + offset[1])
        else:
            return (self.x + sin(radians(self.angle - angle)) * distance) + offset[0], \
                   (self.y + cos(radians(self.angle - angle)) * distance + offset[1])
