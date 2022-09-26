from abc import ABC

import numpy as np
import pygame
import time

from typing import List
from enum import Enum


class ElementType(Enum):
    """
    Enum containing all suppported Element Types
    """
    ELEMENT = 0,
    BUTTON = 1


class Action:

    def __init__(self, event_type, func, active=True):
        self.event_type = event_type
        self.func = func
        self.active = active


class ActionSubscriptions:

    def __init__(self):
        self.subscriptions = {}

    def add_subscription(self, action):
        if self.subscriptions[action.event_type] is None:
            self.subscriptions[action.event_type] = []
        self.subscriptions[action.event_type].append(action)

    def fire(self, event_type):
        if self.subscriptions[event_type] is None:
            self.subscriptions[event_type] = []
        for i in self.subscriptions[event_type]:
            if i.active:
                i.func()


class Element:

    def __init__(self, top=0, left=0, width=0, height=0, element_id=None, class_name=None, dom=None):
        """

        """
        self.x = 0
        self.y = 0
        self.type_name = ElementType.ELEMENT
        self.class_name = class_name
        self.parent_element = None
        self.id = element_id
        self.rect = pygame.Rect(left, top, width, height)
        self.surface = pygame.Surface((self.rect.w, self.rect.h))
        self.action_subscriptions = ActionSubscriptions()  # dict holding all subscriptions
        # private attributes
        if self.id == "root":
            assert dom is not None, "DOM cannot be None for root element"
            self.dom = dom
        self._children = []
        self._elements = {}
        self._num_children = 0

        self._child_id = 0

    def set_child_id(self, child_id):
        self._child_id = child_id

    def get_child_id(self):
        return self._child_id

    def is_root(self):
        return self.id == 'root'

    def update(self, app):
        pass

    def render(self, app):
        """
        Render the given element
        :param app:
        :return:
        """
        self.update(app)
        app.window.blit(self.surface, (self.x, self.y))
        for c in self._children:
            c.render(app.window)

    def append_child(self, element):
        """
        Append the child to the element
        :param element: the element to add
        :return: None
        """
        if element.id is not None:
            assert self._elements.get(
                element.id) is None, "Illegal Operation: Attempting to add element with duplicate id: {}".format(
                element.id)
            element.parent_element = self
            self._children.append(element)
            self._child_id = self._num_children
            self._elements[element] = element
        self._num_children += 1

    def remove_child(self, element):
        """
        Remove a child given the object
        :param element: child
        :return: None
        """
        assert element.id is not None, "Element must contain an id"
        if self._elements.get(element.id) is not None:
            element = self._elements.pop(element.id)
            self._children.pop(element.get_child_id())
            self._num_children -= 1
        else:
            for element in self._children:
                element.remove_child(element)

    def remove_child_by_id(self, element_id):
        """
        Remove a given child using its id
        :param element_id: child id
        :return: None
        """
        assert element_id is not None, "Element must contain an id"
        if self._elements.get(element_id) is not None:
            element = self._elements.pop(element_id)
            self._children.pop(element.get_child_id())
            self._num_children -= 1
        else:
            for element in self._children:
                element.remove_child(element)

    def get_element_by_id(self, element_id):
        """
        Return an element matching the given id
        :param element_id:
        :return:
        """
        assert element_id is not None, "get_element_by_id() called with argument: [None]"
        if self._elements.get(element_id) is not None:
            return self._elements.get(element_id)
        else:
            for element in self._children:
                element.remove_child(element)
        return None

    def add_event_listener(self, event_type, action):
        self.action_subscriptions.add_subscription(Action(event_type, action))
        self.parent_element.propagate_event_subscription(event_type, self.id)

    def fire_event(self, event_type):
        """
        Fires an event of type [event_type] for the given element
        :param event_type:
        :return:
        """
        self.action_subscriptions.fire(event_type)

    def propagate_event_subscription(self, event_type, element_id):
        """
        Propagates an event subscription to the root element
        :param element_id: The element that is assigned to the event
        :param event_type: The type of event that is assigned to the element
        :return:
        """
        if self.dom is not None:
            self.dom.add_event_listener(event_type, element_id)
        else:
            self.parent_element.propagate_event_subscription(event_type, element_id)


class ArrowKeysDisplay:
    """
    @deprecated
    """

    def __init__(self, unselected_color, selected_color, border=None, scale=1, position=None):
        self.position = position
        self.scale = scale
        self.unselected_color = unselected_color
        self.selected_color = selected_color
        self.border = border
        self.width = 100  # pixels
        # rectangles
        self._left_key_rect = None
        self._up_key_rect = None
        self._right_key_rect = None
        self._down_key_rect = None
        self.build_model()

    def render(self, window, position, current_actions: List[int]):
        """

        :param window: window to blit model onto
        :param current_actions: list of actions that the agent has choosen
        :param position: position on the screen for the model to be printed on
        :param current_action:
        :return:
        """
        pygame.draw.rect(window, (self.unselected_color if current_actions.count(0) == 0 else self.selected_color),
                         self._left_key_rect.move(*position), border_radius=10)
        pygame.draw.rect(window, (self.unselected_color if current_actions.count(1) == 0 else self.selected_color),
                         self._up_key_rect.move(*position), border_radius=10)
        pygame.draw.rect(window, (self.unselected_color if current_actions.count(2) == 0 else self.selected_color),
                         self._right_key_rect.move(*position), border_radius=10)
        pygame.draw.rect(window, (self.unselected_color if current_actions.count(3) == 0 else self.selected_color),
                         self._down_key_rect.move(*position), border_radius=10)
        if self.border is not None:
            pygame.draw.rect(window, self.border,
                             self._left_key_rect.move(*position), width=10, border_radius=10)
            pygame.draw.rect(window, self.border,
                             self._up_key_rect.move(*position), width=10, border_radius=10)
            pygame.draw.rect(window, self.border,
                             self._right_key_rect.move(*position), width=10, border_radius=10)
            pygame.draw.rect(window, self.border,
                             self._down_key_rect.move(*position), width=10, border_radius=10)

    def build_model(self) -> np.array:
        # format is x1, y1, width, height
        self.width *= self.scale
        self._up_key_rect = pygame.rect.Rect(self.width, 0, self.width, self.width)
        self._left_key_rect = pygame.rect.Rect(0, self.width, self.width, self.width)
        self._down_key_rect = pygame.rect.Rect(self.width, self.width, self.width, self.width)
        self._right_key_rect = pygame.rect.Rect(self.width * 2, self.width, self.width, self.width)


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

    def render(self, window, position=None):
        text = self.font.render(self.text, self.anti_alias, self.color, self.background)
        window.blit(text, self.position if position is None else position)

    def append_text(self, text, refresh_count=None, append_to_front=False):
        if refresh_count is not None:
            self.refresh_count = refresh_count
        if self.refresh_count is None or self.current_count >= self.refresh_count:
            self.text = self.original_text + text if not append_to_front else text + self.original_text
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


class TimedLabel(Label):

    def __init__(self, position, timeout: float, queue, text="", size=12, font=None, color=(0, 0, 0),
                 refresh_count=None, background=None,
                 anti_alias=False):
        """
        :param position: (x, y)
        :param timeout: number of seconds for the label to last
        :param text:
        :param size:
        :param font:
        :param color:
        :param refresh_count:
        :param background:
        :param anti_alias:
        """
        super().__init__(position=position, text=text, size=size, font=font, color=color, refresh_count=refresh_count,
                         background=background, anti_alias=anti_alias)
        self.time_created = time.time()
        self.timeout = timeout
        self.queue = queue

    def render(self, window):
        if (time.time() - self.time_created) < self.timeout:
            text = self.font.render(self.text, self.anti_alias, self.color, self.background)
            window.blit(text, self.position)
        else:
            self._remove_label()

    def _remove_label(self):
        self.queue.remove_label()


class TimedLabelQueue:

    def __init__(self, window):
        self.labels = []
        self.current_label = None
        self._is_label_showing = False
        self._window = window

    def remove_label(self):
        self.current_label = None
        if len(self.labels) != 0:
            self.current_label = self.labels.pop(0)

    def display_label(self, label, force=False):
        if force or len(self.labels) == 0:
            self.current_label = label
        else:
            self.labels.append(label)

    def render(self):
        if self.current_label is not None:
            self.current_label.render(self._window)
