import time
from typing import *

import numpy as np
import visdom


class Visualizer(object):
    """

    """

    def __init__(self, env='default', **kwargs):
        """

        :param env:
        :param kwargs:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def __getattr__(self, name):
        """

        :param name:
        :return:
        """
        return getattr(self.vis, name)

    def modify(self, env='default', **kwargs):
        """

        :param env:
        :param kwargs:
        :return:
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, dct: Dict[str, float]):
        """

        :param dct:
        :return:
        """
        for name, value in dct.items():
            self.plot_line(name, value)

    def images(self, dct):
        """

        :param dct:
        :return:
        """
        for k, v in dct.items():
            self._image(k, v)

    def plot_line(self, name, y, **kwargs):
        """

        :param name:
        :param y:
        :param kwargs:
        :return:
        """
        x = self.index.get(name, 0)
        self.vis.line(np.array([y]), X=np.array([x]),
                      win=name,
                      update='append',
                      name=name,
                      opts=dict(title=name),
                      **kwargs)
        self.index[name] = x + 1

    def _image(self, name, image, **kwargs):
        """

        :param name:
        :param image:
        :param kwargs:
        :return:
        """
        self.vis.images(image.detach().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs)

    def log(self, info, win='log_text'):
        """

        :param info:
        :param win:
        :return:
        """
        log_time = time.strftime('%m%d_%H%M%S')
        self.log_text += f'[{log_time}] {info} <br>'
        self.vis.text(self.log_text, win)
