import os
from abc import ABC, abstractmethod
from time import time
from components import Label, TimedLabelQueue, ArrowKeysDisplay
import numpy as np
import pygame

from vehicle import Vehicle


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

        Rewards should be initialized within the agent itself and not the src

        :param debug: whether or not sensors/etc. will be shown
        :param fps: None if simulation is not run based on fps (speed of while-loop) otherwise fps of src
        :param num_episodes: the number of episodes (crashes) before the simulation dies
                            - None if the src runs forever
        """
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
        self._iteration_num = 1
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
        # labels
        self.fps_label = None
        self.iteration_count_label = None
        self.car_label = None
        self.car_speed_label = None
        self.odometer_label = None
        self.label_manager = TimedLabelQueue(self.window)
        self.arrow_display = ArrowKeysDisplay(
            unselected_color=(255, 255, 255),
            selected_color=(255, 0, 255),
            border=(0, 0, 0),
            scale=1,
            position=None
        )
        # this value should be overridden by child class
        self.start_pos = (0, 0)
        # initialize car later
        self.car = car
        self._car_top_distance = 0

    def initialize(self):
        pygame.init()
        assert self.start_pos is not None
        assert self.car is not None
        assert self.car.driver is not None
        # handle init
        self.init_display()
        self.init_iteration_count_label()
        self.init_fps_label()
        self.init_car_label()
        self.init_car_stat_labels()
        self.convert_images()  # initializes the track

    def init_car_stat_labels(self):
        self.odometer_label = Label(position=(600, self._screen_dim[1] - 50), text=" Pixels", size=40, font=None,
                                    color=(255, 255, 255), refresh_count=None, background=None, anti_alias=True)
        # Position is arbitrary as it will be moved
        self.car_speed_label = Label(position=(0, 0), text=" PPF", size=40, font=None,
                                     color=(0, 0, 0), refresh_count=None, background=None, anti_alias=True)

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
        assert border is not None and track is not None and rewards is not None
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
                               anti_alias=True)

    def init_car_label(self):
        self.car_label = Label((10, self._screen_dim[1] - 40), "Longest Drive: ", size=30, font=None, color=(0, 0, 0),
                               background=(255, 255, 255), anti_alias=True)

    def init_iteration_count_label(self):
        self.iteration_count_label = Label((1100, 10), "Iteration: ", size=30, font=None, color=(0, 0, 0),
                                           background=(255, 255, 255), anti_alias=True)

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
            self.reset()
        self.car.step(reward, collision, keys_pressed)
        self.car.update(simulation=self)
        self.update_debug_display(reward, collision)
        self.update_and_display_labels()
        self.update_and_display_arrow_display()
        self.op_display()

    @abstractmethod
    def update_and_display_arrow_display(self):
        pass

    def update_and_display_labels(self):
        self.fps_label.append_text(str(self._calc_fps), self._calc_fps / 2)
        self._car_top_distance = max(self._car_top_distance, self.car.odometer)
        self.car_label.append_text(str(round(self._car_top_distance)), refresh_count=None)
        self.car_speed_label.append_text(str(round(self.car.velocity.speed)), self._calc_fps / 2, append_to_front=True)
        self.odometer_label.append_text(str(round(self.car.odometer)), refresh_count=None, append_to_front=True)
        self.iteration_count_label.append_text(str(self._iteration_num))
        self.iteration_count_label.render(self.window)
        self.car_label.render(self.window)
        self.car_speed_label.render(self.window, position=(self.car.velocity.x + 10, self.car.velocity.y - 40))
        self.odometer_label.render(self.window)
        self.fps_label.render(self.window)
        self.label_manager.render()

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
        self._iteration_num += 1
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


class CarTestSimulation(Simulation, ABC):

    def __init__(self,
                 debug=True,
                 fps=None,
                 num_episodes=None,
                 caption: str = None,
                 car: Vehicle = None,
                 track_offset=(0, 0),
                 screen_size=(1400, 800),
                 track_size=(1400, 800)):
        super(CarTestSimulation, self).__init__(debug=debug,
                                                fps=fps,
                                                num_episodes=num_episodes,
                                                caption=caption,
                                                car=car,
                                                track_offset=track_offset,
                                                screen_size=screen_size,
                                                track_size=track_size)
        self.border_path = None
        self.track_path = None
        self.rewards_path = None

    def set_track_paths(self, border_path, track_path, rewards_path):
        """
        Initialize the paths to the track images
        :param border_path:
        :param track_path:
        :param rewards_path:
        :return:
        """
        self.border_path = border_path
        self.track_path = track_path
        self.rewards_path = rewards_path

    def set_start_pos(self, start_pos=None, x=None, y=None):
        """
        sets the start position of the car
        :return:
        """
        assert start_pos is not None or (x is not None and y is not None)
        if start_pos is not None:
            self.start_pos = start_pos
        else:
            self.start_pos = (x, y)

    def update_and_display_arrow_display(self):
        self.arrow_display.render(
            self.window, (520, 500), current_actions=self.car.current_action
        )

    def init_track(self) -> (str, str, str):
        """
        Should set the images of the track (paths to the images):
        called in the constructor of the simulation class
        - track border
        - track bg
        - track rewards
        :return: the path to the tracks in the order 'border, background (design), rewards'
        """
        return self.border_path, self.track_path, self.rewards_path
