import pandas as pd
import numpy as np
import cv2
import os
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2


# untouched example pipeline
training_transform = A.Compose([
    A.Rotate(limit=90, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
