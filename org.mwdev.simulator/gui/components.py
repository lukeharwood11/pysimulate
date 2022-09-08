import pygame
import time


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
        super(TimedLabel).__init__(position, text, size, font, color, refresh_count, background, anti_alias)
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
        if force:
            self.current_label = label

    def render(self):
        if self.current_label is not None:
            self.current_label.render(self._window)
