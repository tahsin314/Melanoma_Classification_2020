# Courtesy: https://www.kaggle.com/nroman/melanoma-pytorch-starter-efficientnet
# Albumentation version is also added
import random
import numpy as np
import cv2 
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

class DrawHair:
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        width, height, _ = img.shape
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, width={self.width})'

class Hair(ImageOnlyTransform):
    def __init__(
            self, hairs:int = 4, width:tuple = (1, 2),
            always_apply=False,
            p=0.5,
    ):
        super(Hair, self).__init__(always_apply, p)
        self.hairs = hairs
        self.width = width

    def apply(self, image, **params):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return image
        
        width, height, _ = image.shape
        self.image = image
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(self.image, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return self.image