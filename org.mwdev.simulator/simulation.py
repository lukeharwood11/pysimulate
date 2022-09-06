import os
from abc import ABC, abstractmethod
from time import time

import numpy as np
import pygame


class Label:

    def __init__(self, position, text="", size=12, font=None, color=(0, 0, 0), refresh_count=None, background=None,
                 anti_alias=False):
        """
        - Custom label class for rendering labels
        :param position:
        :param text:
        :param size:
        :param font:
        :param color:
        :param refresh_count:
        :param background:
        :param anti_alias:
        """
        self.font = font
        self.original_text = text
        self.text = text
        self.color = color
        self.size = size
        self.position = position
        self.background = background
        self.anti_alias = anti_alias
        self.refresh_count = refresh_count
        self.current_count = 0

        if font is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), self.size)

    def render(self, window):
        text = self.font.render(self.text, self.anti_alias, self.color, self.background)
        window.blit(text, self.position)

    def append_text(self, text, refresh_count=None):
        if refresh_count is not None:
            self.refresh_count = refresh_count
        if self.refresh_count is None or self.current_count >= self.refresh_count:
            self.text = self.original_text + text
            self.current_count = 0
        else:
            self.current_count += 1

    def update_text(self, text, refresh_count=None):
        if refresh_count is not None:
            self.refresh_count = refresh_count
        if self.refresh_count is None or self.current_count >= self.refresh_count:
            self.original_text = text
            self.text = text
        else:
            self.current_count += 1


class Simulation(ABC):

    def __init__(self,
                 debug=True,
                 fps=None,
                 num_episodes=None,
                 caption: str = None,
                 car=None,
                 track_offset=(0, 0),
                 screen_size=(1400, 800),
                 track_size=(1400, 800)):
        """

        Simulation should hold only the information relevant to the actual simulation and not any information
        about the agent driving (the ai or human)

        Rewards should be initialized within the agent itself and not the simulator

        :param debug: whether or not sensors/rewards/etc. will be shown
        :param fps: None if simulation is not run based on fps (speed of while-loop) otherwise fps of simulator
        :param num_episodes: the number of episodes (crashes) before the simulation dies
                            - None if the simulator runs forever
        """
        self.fps_label = None
        pygame.init()

        # private attributes
        self._screen_dim = screen_size
        self._track_dim = track_size
        self._fps = fps
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # track information
        self.start_pos = None
        self._track_border = None
        self.track_border_width = None
        self.track_border_height = None
        self.border_mask = None
        self._track_bg = None
        self._track_rewards = None
        self.rewards_mask = None
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._clock = pygame.time.Clock()
        self._iteration_num = 0
        self._debug = debug
        self._max_episodes = num_episodes
        self._caption = caption
        self._track_offset = track_offset
        # a list of rects that have been updated since the last screen refresh
        self._rect_changed = []
        # helper attribute for calculating the actual fps
        self.current_timestamp = None
        self._calc_fps = 0

        # public attributes
        self.window = pygame.display.set_mode(self._screen_dim)
        # this value should be overridden by child class
        self.start_pos = (0, 0)
        # initialize car later
        self.car = car

        # handle init
        self.init_display()
        self.init_fps_label()
        self.init_car_start_pos()
        self.convert_images()  # initializes the track

    @abstractmethod
    def init_car_start_pos(self):
        """
        sets the start position of the car
        :return:
        """
        pass

    @abstractmethod
    def init_track(self) -> (str, str, str):
        """
        Should set the images of the track (paths to the images):
        called in the constructor of the simulation class
        - track border
        - track bg
        - track rewards
        :return: the path to the tracks in the order 'border, background (design), rewards'
        """
        pass

    def convert_images(self):
        """
        called in constructor for converting images
        :return:
        """
        border, track, rewards = self.init_track()
        self._track_border = pygame.image.load(border)
        self._track_border = pygame.transform.smoothscale(self._track_border, self._track_dim).convert_alpha()
        self._track_bg = pygame.image.load(track).convert()
        self._track_bg = pygame.transform.smoothscale(self._track_bg, self._track_dim).convert()
        self._track_rewards = pygame.image.load(rewards)
        self._track_rewards = pygame.transform.smoothscale(self._track_rewards, self._track_dim).convert_alpha()

        if self._track_border is not None:
            self.track_border_width = self._track_border.get_width()
            self.track_border_height = self._track_border.get_height()
        if border is not None:
            self.border_mask = pygame.mask.from_surface(self._track_border)
        if rewards is not None:
            self.rewards_mask = pygame.mask.from_surface(self._track_rewards)

    def get_vehicle_offset(self):
        return None if self.car is None else (self.car.image.get_width() / 2, self.car.image.get_height() / 2)

    def get_vehicle_image_position(self):
        """
        :return: The absolute position of the image of the vehicle (in relation to the window)
        """
        return np.array((self.car.velocity.x + (self.car.image.get_width() / 2) + 12,
                  self.car.velocity.y + (self.car.image.get_height() / 2) + 12))

    def init_display(self):
        if self._caption is None:
            self._caption = "Racing Simulation"

    def init_fps_label(self):
        self.fps_label = Label((10, 10), "FPS: 0", size=30, font=None, color=(0, 0, 0), background=None,
                               anti_alias=False)

    def simulate(self):
        """
        main 'game-loop' for simulation
        :return: None
        """
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        print("Begin simulation init...")
        run = True
        pygame.display.set_caption(self._caption)
        if self.car is not None:
            print("Initializing car image conversion...")
            self.car.image = self.car.image.convert()
            self.car.reset(simulation=self)
        print("Done!")
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        while run:
            if self._fps is not None and run:
                self._clock.tick(self._fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    if self.car is not None:
                        print("Saving Car...")
                        self.car.save_car()
                    break
            t = self.current_timestamp
            self.current_timestamp = time()
            if t is not None:
                self.calculate_fps(self.current_timestamp - t)
            keys_pressed = pygame.key.get_pressed()
            self.update_display(keys_pressed)

    def calculate_fps(self, time_elapsed):
        # convert to seconds (from milliseconds)
        t = time_elapsed
        # save to attribute
        self._calc_fps = round(1 / t)

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
        reward = self.handle_reward()
        if reward:
            pass
        collision = self.handle_collision()
        if collision:
            print("YAY")
            self.reset()
        self.car.step(reward, collision, keys_pressed)
        self.car.update(simulation=self)
        # print(self.car.velocity.angle)
        # print(self.car.get_input())
        self.fps_label.append_text(str(self._calc_fps), self._calc_fps / 2)
        self.fps_label.render(self.window)
        self.update_debug_display(reward, collision)
        print("Sensor Values:", self.car.print_sensor_values())
        self.op_display()

    # TODO implement
    def op_display(self):
        """
        optimize what elements are getting displayed to the screen
        if no elements are updated the screen shouldn't update
        :return:
        """
        # fixme -> add check to see what elements have been updated
        # pygame.display.update(self._rect_changed)
        pygame.display.update()

    def update_debug_display(self, reward: bool, collision: bool):
        """
        if debug is active then display elements
        :return:
        """
        pass

    def reset(self):
        """
        Called whenever the car crashes and the simulation starts over
        :return:
        """
        self.car.reset(simulation=self)

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Handlers for display/collision detection
    # - - - - - - - - - - - - - - - - - - - - - - - - - -

    def handle_collision(self) -> bool:
        """
        Only works if the track_border is not None
        :return: whether the vehicle hit a wall
        """
        if self._track_border is not None \
                and self.car is not None \
                and self.car.current_image is not None:
            car_mask = pygame.mask.from_surface(self.car.image)
            x, y = (self.car.velocity.x + (.5 * self.car.image.get_width()),
                    self.car.velocity.y + (.5 * self.car.image.get_height()))
            col = self.border_mask.overlap(car_mask, (x, y))
            if col is not None:
                return True
        return False

    def handle_reward(self) -> bool:
        """
        Only works if the track_rewards is not None
        :return: whether the vehicle is touching a reward
        """
        if self._track_rewards is not None \
                and self.car is not None \
                and self.car.current_image is not None:
            car_mask = pygame.mask.from_surface(self.car.image)
            car_pos = self.get_vehicle_image_position()
            col = self.rewards_mask.overlap(car_mask, (car_pos[0], car_pos[1]))
            if col is not None:
                return True
        return False

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Display for application aesthetics and debugging
    # - - - - - - - - - - - - - - - - - - - - - - - - - -

    def display_sensor_values(self, start_pos: (int, int)):
        pass
