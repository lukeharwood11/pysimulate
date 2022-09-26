from abc import ABC
from time import time

import pygame

from pysimulate import Element, ActionSubscriptions
from utils import calculate_fps


class AppFunction:

    def __init__(self, func, requires_app=False):
        """
        :param func:
        :param requires_app:
        """
        self.func = func
        self.requires_app = requires_app


class Event:

    def __init__(self, event_type):
        self.type = event_type

    def consume(self):
        pass


class EventManager:

    def __init__(self, document):
        self.document = document
        self._subscriptions = {}

    def add_subscription(self, event_type, element_id):
        if self._subscriptions.get(event_type) is None:
            self._subscriptions[event_type] = []
        self._subscriptions[event_type].append(element_id)

    def fire(self, event_type):
        if self._subscriptions.get(event_type) is None:
            self._subscriptions[event_type] = []
        for e_id in self._subscriptions[event_type]:
            self.document.get_element_by_id(e_id).fire_event(event_type)


class Document:

    def __init__(self, window):
        self._elements = {}
        self._style_sheet = {}
        self._window = window
        self._elements["root"] = Element(element_id="root")
        self._event_manager = EventManager(self)

    def get_root_element(self):
        """
        :return: the root element
        """
        return self._elements["root"]

    def get_element_by_id(self, element_id):
        """
        :param element_id: the id of the element you are trying to find
        :return: an element, or None if it doesn't exist
        """
        return self._elements.get(element_id)

    def get_elements_by_type_name(self, type_name):
        """
        Search for elements by Type
        :param type_name: ElementType of the elements
        :return: a list of elements, or an empty list if none exist
        """
        elements = []
        for values in self._elements.values():
            if values.type_name == type_name:
                elements.append(elements)
        return elements

    def get_elements_by_class_name(self, class_name):
        """
        :param class_name: the class name of the elements
        :return: a list of elements, or an empty list if none exist
        """
        elements = []
        for values in self._elements.values():
            if values.class_name == class_name:
                elements.append(elements)
        return elements

    def add_element(self, element):
        """
        Add an element to the root element of the DOM
        :param element:
        :return:
        """
        assert self._elements.get(
            element.id) is None, "Illegal Operation: Attempting to add element with duplicate id: {}".format(element.id)
        self._elements[element.id] = element

    def remove_element(self, element):
        assert element.id != "root", "Illegal Operation: Attempting to remove the root element."
        assert element.id is not None, "Element must contain an id"
        if self._elements.get(element.id) is not None:
            self._elements.pop(element.id)
        else:
            self.get_root_element().remove_child(element)

    def remove_element_by_id(self, element_id):
        assert element_id != "root", "Illegal Operation: Attempting to remove the root element."
        assert element_id is not None, "Element must contain an id"
        if self._elements.get(element_id) is not None:
            self._elements.pop(element_id)
        else:
            self.get_root_element().remove_child_by_id(element_id)

    def pack(self):
        """
        Verifies layout of application and ensures no errors were made when assembling DOM
        - Ensures that no element_ids are the same
        :return: None
        """
        pass

    def manage_event(self, event):
        self._event_manager.fire(event)

    def add_event_listener(self, event_type, element_id):
        self._event_manager.add_subscription()

    def render(self, application):
        """
        render the document model to the pygame window
        TODO remove application parameter if unneeded
        :return: None
        """
        pass

    @staticmethod
    def from_file(path):
        pass


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

        self._init = []
        self._on_close = []

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
                self.document.manage_event(event)
            if self.fps is not None:
                self.clock.tick(self.fps)
            self.document.render(self)
            pygame.display.update()
