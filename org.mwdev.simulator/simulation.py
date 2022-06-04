import os
from abc import ABC, abstractmethod
from time import time
import pygame


class Label:

    def __init__(self, position, text="", size=12, font=None, color=(0, 0, 0), background=None, anti_alias=False):
        self.font = font
        self.text = text
        self.color = color
        self.size = size
        self.position = position
        self.background = background
        self.anti_alias = anti_alias

        if font is None:
            self.font = pygame.font.Font(pygame.font.get_default_font(), self.size)

    def render(self, window):
        text = self.font.render(self.text, color=self.color, antialias=self.anti_alias, background=self.background)
        window.blit(text, self.position)


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
        pygame.init()

        # private attributes
        self._screen_dim = screen_size
        self._track_dim = track_size
        self._fps = fps
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # track information
        self.start_pos = None
        self._track_border = None
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
        self._track_border = pygame.image.load(border).convert_alpha()
        self._track_bg = pygame.image.load(track).convert()
        self._track_rewards = pygame.image.load(rewards).convert_alpha()

        if border is not None:
            self.border_mask = pygame.mask.from_surface(self._track_border)
        if rewards is not None:
            self.rewards_mask = pygame.mask.from_surface(self._track_rewards)

    def init_display(self):
        if self._caption is None:
            self._caption = "Racing Simulation"

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
        collision = self.handle_collision()
        self.car.step(reward, collision, keys_pressed)
        self.car.update(simulation=self)
        self.update_debug_display(reward, collision)
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
        if self._track_border is not None:
            return True
        return False

    def handle_reward(self) -> bool:
        """
        Only works if the track_rewards is not None
        :return: whether the vehicle is touching a reward
        """
        if self._track_rewards is not None:
            return True
        return False

    # - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Display for application aesthetics and debugging
    # - - - - - - - - - - - - - - - - - - - - - - - - - -

    def display_sensor_values(self, start_pos: (int, int)):
        pass
