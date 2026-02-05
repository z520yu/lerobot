import matplotlib.pyplot as plt
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoDisplay(_subscriber.Subscriber):
    """显示环境图像，优先 cam_high。"""

    def __init__(self) -> None:
        self._ax: plt.Axes | None = None
        self._plt_img: plt.Image | None = None

    @override
    def on_episode_start(self) -> None:
        plt.ion()
        self._ax = plt.subplot()
        self._plt_img = None

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        assert self._ax is not None

        if "images" in observation and isinstance(observation["images"], dict):
            images = observation["images"]
            key = "cam_high" if "cam_high" in images else next(iter(images.keys()))
            im = images[key]
            if im.ndim == 3 and im.shape[0] in (1, 3) and im.shape[-1] not in (1, 3):
                # CHW -> HWC
                im = np.transpose(im, (1, 2, 0))
        else:
            im = observation["image"][0]
            im = np.transpose(im, (1, 2, 0))

        if self._plt_img is None:
            self._plt_img = self._ax.imshow(im)
        else:
            self._plt_img.set_data(im)
        plt.pause(0.001)

    @override
    def on_episode_end(self) -> None:
        plt.ioff()
        plt.close()


