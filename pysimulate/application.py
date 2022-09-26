from abc import ABC
from time import time

import pygame

from pysimulate import Element
from utils import calculate_fps


class AppFunction:

    def __init__(self, func, requires_app=False):
        """
        :param func:
        :param requires_app:
        """
        self.func = func
        self.requires_app = requires_app


class Document:

    def __init__(self, window):
        self._elements = {}
        self._style_sheet = {}
        self._window = window
        self._elements["root"] = Element(element_id="root")

    def get_root_element(self):
        return self._elements["root"]

    def get_element_by_id(self, element_id):
        return self._elements.get(id)

    def get_elements_by_type_name(self, type_name):
        elements = []
        for values in self._elements.values():
            if values.type_name == type_name:
                elements.append(elements)
        return elements

    def get_elements_by_class_name(self, class_name):
        elements = []
        for values in self._elements.values():
            if values.class_name == class_name:
                elements.append(elements)
        return elements

    def add_element(self, element):
        assert self._elements.get(element.id) is None, "Attempting to add element with duplicate id: {}".format(element.id)
        self._elements[element.id] = element

    def remove_element(self, element):
        assert element.id != "root", "Illegal Operation: Attempting to remove the root element."
        if self._elements.get(element.id) is not None:
            self._elements.pop(element.id)


class App:

    def __init__(self, screen_dim=None,
                 caption="Pysimulate Simulation",
                 no_frame=False,
                 full_screen=True,
                 scaled=True,
                 resizeable=False,
                 fps=None,
                 bg_color=(0, 0, 0)):
        # Screen attributes
        self.screen_dim = screen_dim
        self.caption = caption
        self.resizable = resizeable
        self.full_screen = full_screen
        self.no_frame = no_frame
        self.scaled = scaled
        self.opengl = False  # not supported at this time
        self.screen_dim = screen_dim
        self.bg_color = bg_color

        self.window = self.init_window()
        self.document = Document(self.window)

        self.fps = fps
        self.running_fps = 0
        self.clock = pygame.time.Clock()

        self.caption = caption
        self.current_timestamp = time()

        self._components = []
        self._init = []
        self._on_close = []

    def add_component(self, component):
        self._components.append(component)

    def add_initializer(self, initializer):
        self._init.append(initializer)

    def add_exit_action(self, exit_action):
        self._on_close.append(exit_action)

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
            if self.bg_color is not None:
                self.window.fill(self.bg_color)
            time_now = time()
            self.running_fps = calculate_fps(time_now - self.current_timestamp)
            self.current_timestamp = time_now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    for save in self._on_close:
                        save.func(self) if save.requires_app else save.func()
            if self.fps is not None:
                self.clock.tick(self.fps)
            for component in self._components:
                component.render(self)
            pygame.display.update()
