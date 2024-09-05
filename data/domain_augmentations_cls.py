"""IMPORT PACKAGES"""
import albumentations as A
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random


"""""" """""" """""" """""" ""
"""" DATA AUGMENTATION """
"""""" """""" """""" """""" ""


# Custom Class for Identity Mapping (Torchvision)
class Identity:
    def __init__(self):
        self.identity = None

    def __call__(self, img):
        return img


# Custom Class for Motion Blur (Albumentations)
class MotionBlur:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.blur_limit = [(9, 9), (11, 11), (17, 17), (21, 21), (25, 25)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = A.MotionBlur(blur_limit=self.blur_limit[severity], always_apply=True, allow_shifted=False)(
            image=np.array(image)
        )['image']

        return Image.fromarray(image)


# Custom Class for Median Blur (Albumentations)
class MedianBlur:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.blur_limit = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = A.MedianBlur(blur_limit=self.blur_limit[severity], always_apply=True)(image=np.array(image))['image']

        return Image.fromarray(image)


# Custom Class for Zoom Blur (Albumentations)
class ZoomBlur:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.max_factor = [(1.01, 1.02), (1.03, 1.04), (1.05, 1.06), (1.07, 1.08), (1.09, 1.10)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = A.ZoomBlur(max_factor=self.max_factor[severity], always_apply=True)(image=np.array(image))['image']

        return Image.fromarray(image)


# Custom Class for Lens Blur (Albumentations)
class LensBlur:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.blur_limit = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = A.Blur(blur_limit=self.blur_limit[severity], always_apply=True)(image=np.array(image))['image']

        return Image.fromarray(image)


# Custom Class for Defocus Blur (Albumentations)
class DefocusBlur:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.radius = [(9, 9), (11, 11), (17, 17), (21, 21), (25, 25)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = A.Defocus(radius=self.radius[severity], always_apply=True)(image=np.array(image))['image']

        return Image.fromarray(image)


# Custom Class for Increased Contrast (Torchvision)
class IncreasedContrast:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.limit = [(1.01, 1.10), (1.11, 1.20), (1.21, 1.25), (1.26, 1.30), (1.31, 1.4)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(contrast=(self.limit[severity][0], self.limit[severity][1]))(image)

        return image


# Custom Class for Decreased Contrast (Torchvision)
class DecreasedContrast:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.limit = [(0.95, 0.99), (0.90, 0.94), (0.85, 0.89), (0.75, 0.84), (0.65, 0.74)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(contrast=(self.limit[severity][0], self.limit[severity][1]))(image)

        return image


# Custom Class for Increased Brightness (Torchvision)
class IncreasedBrightness:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.limit = [(1.01, 1.20), (1.21, 1.40), (1.41, 1.60), (1.61, 1.80), (1.81, 2.0)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(brightness=(self.limit[severity][0], self.limit[severity][1]))(image)

        return image


# Custom Class for Decreased Brightness (Torchvision)
class DecreasedBrightness:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.limit = [(0.90, 0.99), (0.85, 0.89), (0.80, 0.84), (0.75, 0.79), (0.70, 0.74)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(brightness=(self.limit[severity][0], self.limit[severity][1]))(image)

        return image


# Custom Class for Increased Saturation (Torchvision)
class IncreasedSaturation:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.sat_shift_limit = [(1.01, 1.05), (1.06, 1.1), (1.11, 1.15), (1.16, 1.20), (1.21, 1.25)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(
            saturation=(self.sat_shift_limit[severity][0], self.sat_shift_limit[severity][1])
        )(image)

        return image


# Custom Class for Decreased Saturation (Torchvision)
class DecreasedSaturation:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.sat_shift_limit = [(0.90, 0.99), (0.80, 0.89), (0.70, 0.79), (0.60, 0.69), (0.50, 0.59)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(
            saturation=(self.sat_shift_limit[severity][0], self.sat_shift_limit[severity][1])
        )(image)

        return image


# Custom Class for Hue in Red/Pink spectrum (Torchvision)
class HueRed:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.hue_shift_limit = [(0.0, 0.01), (0.011, 0.013), (0.014, 0.016), (0.017, 0.018), (0.019, 0.020)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(hue=(self.hue_shift_limit[severity][0], self.hue_shift_limit[severity][1]))(
            image
        )

        return image


# Custom Class for Hue in Yellow/Green spectrum (Torchvision)
class HueGreen:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.hue_shift_limit = [(-0.01, 0.0), (-0.013, -0.011), (-0.016, -0.014), (-0.020, 0.017), (-0.025, -0.021)]

    def __call__(self, image):
        severity = random.randint(0, self.max_severity)
        image = transforms.ColorJitter(hue=(self.hue_shift_limit[severity][0], self.hue_shift_limit[severity][1]))(
            image
        )
        return image


# Custom Class for Increased Sharpness(Albumentations)
class IncreasedSharpness:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.alpha = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
        self.lightness = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]

    def __call__(self, image):
        severity_a = random.randint(0, self.max_severity)
        severity_l = random.randint(0, self.max_severity)
        image = A.Sharpen(alpha=self.alpha[severity_a], lightness=self.lightness[severity_l], always_apply=True)(
            image=np.array(image)
        )['image']

        return Image.fromarray(image)


# Custom Class for Decreased Sharpness (Albumentations)
class DecreasedSharpness:
    def __init__(self, max_severity=4):
        self.max_severity = max_severity
        self.alpha = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
        self.blur_limit = [(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)]

    def __call__(self, image):
        severity_a = random.randint(0, self.max_severity)
        severity_b = random.randint(0, self.max_severity)
        image = A.UnsharpMask(blur_limit=self.blur_limit[severity_b], alpha=self.alpha[severity_a], always_apply=True)(
            image=np.array(image)
        )['image']

        return Image.fromarray(image)
