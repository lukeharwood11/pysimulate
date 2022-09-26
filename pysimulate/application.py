from abc import ABC
from time import time

import pygame
from utils import calculate_fps

class Initializer:

    def __init__(self, func, requires_app=False):
        """
        :param func:
        :param requires_app:
        """
        self.func = func
        self.requires_app = requires_app


class App:

    def __init__(self, screen_dim=None,
                 caption="Pysimulate Simulation",
                 no_frame=False,
                 full_screen=True,
                 scaled=True,
                 resizeable=False,
                 fps=None):
        # Screen attributes
        self.screen_dim = screen_dim
        self.caption = caption
        self.resizable = resizeable
        self.full_screen = full_screen
        self.no_frame = no_frame
        self.scaled = scaled
        self.opengl = False  # not supported at this time
        self.screen_dim = screen_dim

        self.window = self.init_window()

        self.fps = fps
        self.running_fps = 0
        self.clock = pygame.time.Clock()

        self.caption = caption
        self.current_timestamp = time()

        self._components = []
        self._init = []

    def add_component(self, component):
        self._components.append(component)

    def add_initializer(self, initializer):
        self._init.append(initializer)

    def init_window(self):
        flags = pygame.SHOWN
        if self.scaled:
            flags = flags | pygame.SCALED
        if self.resizable:
            flags = flags | pygame.RESIZABLE
        if self.no_frame:
            flags = flags | pygame.NOFRAME
        if self.full_screen:
            flags = flags | pygame.FULLSCREEN
        if self.opengl:
            flags = flags | pygame.OPENGL

        return pygame.display.set_mode(size=self.screen_dim, flags=flags)

    def launch(self):
        pygame.init()
        running = True
        # run all the initializers
        for i in self._init:
            i.func(self) if i.requires_app else i.func()
        while running:
            self.window.fill((0, 0, 0))
            time_now = time()
            self.running_fps = calculate_fps(time_now - self.current_timestamp)
            self.current_timestamp = time_now
            if self.fps is not None:
                self.clock.tick(self.fps)
            for component in self._components:
                component.render(self)
            pygame.display.update()
