from abc import ABC
from time import time

import numpy as np
import pygame
from gui.components import TimedLabelQueue, TimedLabel, Label
from agent import Agent
from example import Car
import os
from vehicle import SensorBuilder, Sensor
from utils import CollisionSet


class GeneticCar(Car):

    def __init__(self, driver, car_number, debug=False, acceleration_multiplier=.5, normalize=True):
        super(GeneticCar, self).__init__(driver, debug, acceleration_multiplier, normalize)
        self.number = 0
        self.car_number = car_number
        self.parents = []
        self.collision = False

    @staticmethod
    def generate_cars(batch_size, params=None):
        if params is None:
            params = {
                'acceleration_multiplier': .5,
                'normalize': True,
                'debug': False
            }
        return np.array([GeneticCar(
            driver=None, car_number=num, **params
        ) for num in range(batch_size)])

    def reset(self, simulation):
        """
        - Resets the car properties, so it is ready for another episode
        TODO - same implementation of superclass
        :return: None
        """
        self.velocity.reset_velocity(
            x=simulation.start_pos[0],
            y=simulation.start_pos[1],
            angle=180,
            speed=0
        )
        self.collision = False
        self.odometer = 0

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

    def update_sensors(self, window, simulation):
        """
        Calculate the values for the sensors of the vehicle
        :param window: the pygame window
        :param simulation: the simulation that the car is functioning in
        :return: None
        """
        # get (x1,y1,x2,y2) tuples for all sensor positions
        s = [sensor.update(car=self, window=window, simulation=simulation) for sensor in self.sensors]
        if self._debug:
            self.display_sensor(car_pos=GeneticAlgorithmSimulation.get_vehicle_image_position(self), window=window)

    def update(self, simulation):
        """
        update necessary data and send back to simulation
        :param simulation: the simulation the vehicle exists in
        :return: None
        """
        window = simulation.window
        # account for reoccurring events (such as velocity update)
        if not self.collision:
            self.update_pos()
            self.odometer += self.velocity.speed  # update odometer as the distance moved each step
        self.blit_rotate_center(window, (self.velocity.x, self.velocity.y))
        if self.collision:
            pygame.draw.line(window, (255, 0, 0), (self.velocity.x, self.velocity.y),
                             (self.velocity.x + self.current_image.get_width(), self.velocity.y + self.current_image.get_width()), width=10)
            pygame.draw.line(window, (255, 0, 0), (self.velocity.x, self.velocity.y + self.current_image.get_width()),
                             (self.velocity.x + self.current_image.get_width(), self.velocity.y), width=10)

class GeneticCarSet:

    def __init__(self, cars, mini_batch_size):
        """
        Single entity to perform operations concerning all cars
        :param cars: a numpy array of cars
        :param mini_batch_size: the number of cars in the mini-batch
        """
        self.cars = cars
        self.sensor_builder = None
        self.batch_size = len(cars)
        self.car_offset = self.cars[0].image.get_width() / 2, self.cars[0].image.get_height() / 2
        self.mini_batch_size = mini_batch_size
        # batch state
        self.batch_results = np.zeros(self.batch_size)
        # mini-batch state
        self.mini_batch_index = self.mini_batch_size
        self.mini_batch = self.cars[:self.mini_batch_size]
        self.collision_set = CollisionSet(mini_batch_size=mini_batch_size)

        # other
        # simulation must be set prior to running simulation.simulate()
        self.simulation = None

    def initialize(self):
        if self.cars is not None:
            print("Initializing car image conversion...")
            for car in self.cars:
                car.image = car.image.convert()

    def get_external_inputs(self):
        return self.cars[0].get_external_inputs()

    def initialize_sensors(self, sensor_batch):
        for car, sensors in zip(self.cars, sensor_batch):
            car.sensors = sensors

    def initialize_drivers(self, drivers, simulation):
        for car, driver in zip(self.cars, drivers):
            car.driver = driver
            car.reset(simulation=simulation)

    def reset_all(self):
        for car in self.cars:
            car.reset(self.simulation)

    def handle_reset(self, simulation):
        """
        reset the mini_batch if all cars are collided
        :return: None
        """
        if self.collision_set.full_collision().all():
            self.reset_mini_batch(simulation)

    def reset_mini_batch(self, simulation):
        """
        reset the collision_set and queue the next set of cars
        :return:
        """
        self.collision_set.clear()
        for car in self.mini_batch:
            car.reset(self.simulation)

        # If the batch is done
        if self.mini_batch_index >= len(self.cars):
            self.clean_up_iteration(simulation)
        else:
            next_index = self.mini_batch_index + self.mini_batch_size
            # grab the next mini_batch of cars
            self.mini_batch = self.cars[
                              self.mini_batch_index: next_index if next_index < len(self.cars) else len(self.cars) - 1]
            self.mini_batch_index += self.mini_batch_size

    def clean_up_iteration(self, simulation):
        """
        called when the entire batch is complete
        :return:
        """
        simulation.clean_up_iteration()
        s = np.argsort(self.batch_results)
        parent_cars = self.cars[s[-2:]]
        driver1, driver2 = parent_cars[0].driver, parent_cars[1].driver
        # Note: new_drivers includes driver1 and driver2
        new_drivers = driver1.cross_over_mutation(driver2, num_mutations=self.batch_size)
        self.initialize_drivers(new_drivers, simulation)
        return parent_cars[2].odometer

    def update_sensors(self, window, simulation):
        for i, car in enumerate(self.mini_batch):
            if not self.collision_set.collision_at(i):
                car.update_sensors(window, simulation)

    def step_mini_batch(self, keys_pressed):
        for i, car in enumerate(self.mini_batch):
            if not self.collision_set.collision_at(i):
                car.step(
                    reward=False,
                    collision=False,
                    keys_pressed=keys_pressed
                )

    def update_mini_batch(self, simulation):
        for i, car in enumerate(self.mini_batch):
            car.update(simulation)


class GeneticAlgorithmSimulation:

    def __init__(self,
                 debug=True,
                 fps=None,
                 num_episodes=None,
                 cars: GeneticCarSet = None,
                 track_offset=(0, 0),
                 screen_size=(1400, 800),
                 track_size=(1400, 800),
                 batch_size=100,
                 mini_batch_size=3,
                 caption="Genetic Algorithm"
                 ):

        """

                Simulation should hold only the information relevant to the actual simulation and not any information
                about the agent driving (the ai or human)

                Rewards should be initialized within the agent itself and not the simulator

                :param debug: whether or not sensors/etc. will be shown
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
        self.track_border_width = None
        self.track_border_height = None
        self.border_mask = None
        self._track_bg = None
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._clock = pygame.time.Clock()
        self.iteration_num = 1
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
        self.label_manager = TimedLabelQueue(self.window)
        # this value should be overridden by child class
        self.start_pos = (0, 0)

        # handle init
        self.init_display()
        self.init_iteration_count_label()
        self.init_fps_label()
        self.init_car_start_pos()
        self.init_car_label()
        self.convert_images()  # initializes the track
        self.mini_batch_size = mini_batch_size
        self.current_index = mini_batch_size
        # stores the odometer readings from cars that have "competed"
        self.batch_size = batch_size
        self.cars: GeneticCarSet = cars
        self.car_death_count = 0

    def convert_images(self):
        """
        called in constructor for converting images
        :return:
        """
        border, track = self.init_track()
        self._track_border = pygame.image.load(border)
        self._track_border = pygame.transform.smoothscale(self._track_border, self._track_dim).convert_alpha()
        self._track_bg = pygame.image.load(track).convert()
        self._track_bg = pygame.transform.smoothscale(self._track_bg, self._track_dim).convert()

        if self._track_border is not None:
            self.track_border_width = self._track_border.get_width()
            self.track_border_height = self._track_border.get_height()
        if border is not None:
            self.border_mask = pygame.mask.from_surface(self._track_border)

    def get_vehicle_offset(self):
        return None if self.cars is None else self.cars.car_offset

    @staticmethod
    def get_vehicle_image_position(car):
        """
        :return: The absolute position of the image of the vehicle (in relation to the window)
        """
        return np.array((car.velocity.x + (car.image.get_width() / 2) + 12,
                         car.velocity.y + (car.image.get_height() / 2) + 12))

    def init_display(self):
        if self._caption is None:
            self._caption = "Racing Simulation"

    def init_fps_label(self):
        self.fps_label = Label((10, 10), "FPS: 0", size=30, font=None, color=(0, 0, 0), background=None,
                               anti_alias=False)

    def init_car_label(self):
        self.car_label = Label((10, self._screen_dim[1] - 40), "Speed: ", size=30, font=None, color=(0, 0, 0),
                               background=(255, 255, 255), anti_alias=False)

    def init_iteration_count_label(self):
        self.iteration_count_label = Label((1100, 10), "Iteration: ", size=30, font=None, color=(0, 0, 0),
                                           background=(255, 255, 255), anti_alias=False)

    def simulate(self):
        """
        main 'game-loop' for simulation
        :return: None
        """
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        print("Begin simulation init...")
        run = True
        pygame.display.set_caption(self._caption)
        self.cars.initialize()
        print("Done!")
        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        while run:
            if self._fps is not None and run:
                self._clock.tick(self._fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    if self.cars is not None:
                        print("Saving Car...")
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
        self.cars.update_sensors(self.window, self)
        # update the collision set to determine if all cars in the minibatch have 'crashed'
        self.handle_collision()
        # Only call reset when all cars have 'died'
        self.cars.handle_reset(self)
        self.cars.step_mini_batch(keys_pressed=keys_pressed)
        self.cars.update_mini_batch(simulation=self)
        self.update_and_display_labels()
        self.op_display()

    def handle_collision(self) -> None:
        """
        Only works if the track_border is not None
        :return: whether the vehicle hit a wall
        """
        for i, car in enumerate(self.cars.mini_batch):
            if self._track_border is not None \
                    and car is not None \
                    and car.current_image is not None:
                car_mask = pygame.mask.from_surface(car.image)
                # TODO remove manual offset and use function call
                x, y = (car.velocity.x + (.5 * car.image.get_width()),
                        car.velocity.y + (.5 * car.image.get_height()))
                col = self.border_mask.overlap(car_mask, (x, y))
                if col is not None:
                    self.cars.collision_set.set_collision(i)
                    car.collision = True
            else:
                raise Exception("Track has not been properly initialized.")

    def update_and_display_labels(self):
        self.iteration_count_label.append_text(str(self.iteration_num))
        self.iteration_count_label.render(self.window)
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
        resets the minibatch to begin the next one
        Called whenever all cars in mini_batch crash and the next mini_batch begins
        :return:
        """
        # restart the collision set
        self.cars.reset_mini_batch()

    def clean_up_iteration(self):
        """
        - Clean up the current iteration for the next one
        - Find the highest ranking cars, then crossover, then mutate
        :return:
        """
        self.iteration_num += 1


    def init_car_start_pos(self):
        """
        sets the start position of the car
        :return:
        """
        self.start_pos = (875, 100)

    def init_track(self) -> (str, str):
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
            os.path.join("assets", "track.png")


class GeneticAlgorithmDriver(Agent, ABC):

    def __init__(self, num_inputs, num_outputs, driver_id, epsilon):
        super().__init__(num_inputs, num_outputs)
        # weight initialization range of [-1.5, 1.5)
        self.w1 = 3 * np.random.random((num_inputs, 32)) - 1.5
        self.w2 = 3 * np.random.random((32, 16)) - 1.5
        self.w3 = 3 * np.random.random((16, num_outputs)) - 1.5
        self.driver_id = str(driver_id)
        self.epsilon = epsilon

    def forward(self, input_arr: np.array):
        # input shape is (1, num_input)
        pass1 = np.matmul(input_arr, self.w1)  # pass1 has shape of (1, 32)
        pass2 = np.matmul(pass1, self.w2)      # pass2 has shape of (1, 16)
        pass3 = np.matmul(pass2, self.w3)      # pass3 has shape of (1, num_outputs)
        return np.tanh(pass3)                  # shape is still (1, num_outputs)

    def update(self, inputs, reward_collision=False, wall_collision=False, keys_pressed=None) -> list[int]:
        """
        - Given input from the simulation make a decision
        :param wall_collision: whether the car collided with the wall
        :param reward_collision: whether the car collided with a reward
        :param inputs: sensor input as a numpy array
        :param keys_pressed: a map of pressed keys
        :return direction: int [0 - num_outputs)
        """
        return [np.argmax(self.forward(inputs))]

    def cross_over_mutation(self, other, total_batch_size):
        """
        1. cross over self and other
        2. generate {total_batch_size - 2 (self and other)} number of mutations
        3. return the list of drivers as a numpy array, with the first two being the parents
        :param other:
        :return:
        """
        child_driver = self.cross_over(other)
        mutations = child_driver.mutate(total_batch_size - 2)
        return np.array([self, other] + mutations)

    def cross_over(self, other):
        """
        Single point crossover
        :param other:
        :return:
        """
        child_driver = GeneticAlgorithmDriver(self.num_inputs, self.num_outputs, "({} & {})".format(self.driver_id, other.driver_id), self.epsilon)
        cross_over_point = np.random.randint(self.w1.shape[1])
        child_driver.w1 = np.hstack((self.w1[:, :cross_over_point], other.w2[:, cross_over_point:]))
        cross_over_point = np.random.randint(self.w2.shape[1])
        child_driver.w2 = np.hstack((self.w1[:, :cross_over_point], other.w2[:, cross_over_point:]))
        cross_over_point = np.random.randint(self.w3.shape[1])
        child_driver.w3 = np.hstack((self.w1[:, :cross_over_point], other.w2[:, cross_over_point:]))
        return child_driver

    def mutate(self, num_mutations):
        """
        :param num_mutations: the number of mutations to create
        :return: a list of GeneticAlgorithmDriver mutations
        """
        mutations = []
        for i in range(num_mutations):
            mutation = GeneticAlgorithmDriver(self.num_inputs, self.num_outputs, driver_id="{}_{}".format(self.driver_id, i), epsilon=self.epsilon)
            mutation_arr = self.generate_mutation_arr()  # generate mutations and masks
            # read following as add mutations to self.w1 at the positions where the mask is less than epsilon
            mutation.w1 = self.w1 + mutation_arr[0][0] * (mutation_arr[0][1] < self.epsilon)
            mutation.w2 = self.w2 + mutation_arr[1][0] * (mutation_arr[1][1] < self.epsilon)
            mutation.w3 = self.w3 + mutation_arr[2][0] * (mutation_arr[2][1] < self.epsilon)
            mutations.append(mutation)
        return mutations

    def generate_mutation_arr(self):
        """
        :return: list([(mutation array, mask array), ...])
        """
        ret = []
        shapes = [self.w1.shape, self.w2.shape, self.w3.shape]
        for i in shapes:
            # range = number of layers
            ret.append((2 * np.random.random(i) - 1, np.random.random(i)))
        return ret


    @staticmethod
    def generate_drivers(num_drivers, num_inputs, num_outputs, epsilon):
        """
        create a list of random Drivers
        :param num_inputs:
        :param num_outputs:
        :param num_drivers:
        :return:
        """
        return np.array([GeneticAlgorithmDriver(num_inputs, num_outputs, driver_id=identifier, epsilon=epsilon)
                         for identifier in range(num_drivers)])

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


def main():
    # the number of cars within a particular batch
    BATCH_SIZE = 100
    # the number of cars to render on the screen at once
    MINI_BATCH_SIZE = 3
    # step 1: generate cars, and create a new car set
    genetic_cars = GeneticCar.generate_cars(BATCH_SIZE, params=None)
    car_set = GeneticCarSet(genetic_cars, MINI_BATCH_SIZE)
    # step 2: generate genetic simulation
    simulation = GeneticAlgorithmSimulation(
        debug=True,
        fps=40,
        num_episodes=None,
        cars=car_set,
        track_offset=(0, 0),
        batch_size=BATCH_SIZE,
        mini_batch_size=3,
        caption="Genetic Algorithm Simulation"
    )
    # step 3: generate sensors and attach to the cars
    sb = SensorBuilder(
        depth=500,
        sim=simulation,  # ignore warning, simulations are compatible
        default_value=None,
        color="random",
        width=2,
        pointer=True
    )
    sensor_batch = sb.generate_sensors(sensor_range=(-90, 90, 10), num_sets=BATCH_SIZE)
    # Attach the sensors to the cars
    car_set.initialize_sensors(sensor_batch=sensor_batch)
    # calculate inputs and output size
    num_inputs = car_set.get_external_inputs() + sb.num_sensors
    num_outputs = Car.get_num_outputs()
    # step 4: initialize the drivers given the input/output numbers and put them in the cars
    initial_drivers = GeneticAlgorithmDriver.generate_drivers(BATCH_SIZE, num_inputs, num_outputs, epsilon=.25)
    car_set.initialize_drivers(drivers=initial_drivers, simulation=simulation)
    # step 5: simulate!
    simulation.simulate()


if __name__ == "__main__":
    main()
