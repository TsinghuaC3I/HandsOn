# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image-corruption augmentation for Hy-VLA training (additive noise only)."""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

np.bool = np.bool_
import imgaug.augmenters as iaa
from PIL import Image

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        # Execute one of the following noise augmentations
        iaa.OneOf([
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
            ),
            iaa.AdditiveLaplaceNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
            iaa.AdditivePoissonNoise(lam=(0.0, 0.05 * 255), per_channel=0.5)
        ]),
    ],
    # do all of the above augmentations in random order
    random_order=True
)


def image_corrupt(image: Image):
    image_arr = np.array(image)
    image_arr = image_arr[None, ...]

    image_arr = seq(images=image_arr)

    image = Image.fromarray(image_arr[0])
    return image
