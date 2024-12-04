import cv2
import numpy as np
from typing import Tuple
from math import ceil
import torch

class PadSequence(object):
    def __init__(self, length: int, padwith: int = 0):
        self.length = length
        self.padwith = padwith

    def __call__(self, sequence: torch.Tensor):
        sequenceLength = sequence.shape[0]
        if sequenceLength == self.length:
            return sequence
        targetLength = self.length - sequenceLength
        return F.pad(sequence, pad=(0, targetLength), mode="constant", value=self.padwith)


class ResizeToHeight(object):

    def __init__(self, size: int):
        self.height = size

    def forward(self, img: np.ndarray):
        oldWidth, oldHeight = img.shape[1], img.shape[0]
        if oldHeight > oldWidth:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            return cv2.resize(img, (intermediateWidth, self.height))
        else:
            return cv2.resize(img, (int(oldWidth * self.height / oldHeight), self.height))


class ResizeAndPad(object):
    """
    Custom transformation that maintains the image's original aspect ratio by scaling it to the given height and padding
    it to achieve the desired width.
    """

    def __init__(self, height: int, width: int, padwith: int = 1):
        self.width = width
        self.height = height
        self.padwith = padwith

    def __call__(self, img: np.ndarray):
        oldHeight, oldWidth = img.shape[:2]
        if oldWidth == self.width and oldHeight == self.height:
            return img
        else:
            scaleFactor = self.height / oldHeight
            intermediateWidth = ceil(oldWidth * scaleFactor)
            if intermediateWidth > self.width:
                intermediateWidth = self.width
            resized = cv2.resize(img, (intermediateWidth, self.height), interpolation=cv2.INTER_CUBIC)

            # Padding if necessary
            padWidth = self.width - resized.shape[1]
            if padWidth > 0:
                padded = np.full((self.height, self.width, 3), self.padwith, dtype=np.uint8)  # Pad with the specified value (e.g., 1 for black)
                padded[:, :resized.shape[1]] = resized
                return padded
            else:
                return resized

    @classmethod
    def invert(cls, image: np.ndarray, targetShape: Tuple[int, int]) -> np.ndarray:
        # resize so that the height matches, then cut off at width ...
        originalHeight, originalWidth = image.shape[:2]
        scaleFactor = targetShape[0] / originalHeight
        resized = cv2.resize(image, (int(originalWidth * scaleFactor), targetShape[0]))
        return resized[:, :targetShape[1]]

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'
