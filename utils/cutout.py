import torch
import numpy as np


class Cutout(object):
    """Randomly mask out a patche from an image.

    Args:
        length (int): The length (in pixels) of each square patch.
        p (float): The probability of cutout being applied.
    """
    def __init__(self, length, p=0.5):
        self.length = length
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        if np.random.rand() < self.p:

            h = img.size(1)
            w = img.size(2)

            mask = np.ones((h, w), np.float32)

            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length, 0, h)
            y2 = np.clip(y + self.length, 0, h)
            x1 = np.clip(x - self.length, 0, w)
            x2 = np.clip(x + self.length, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img
