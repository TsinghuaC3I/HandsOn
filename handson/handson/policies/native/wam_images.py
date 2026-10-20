import numpy as np
import torch
from PIL import Image

def letterbox(image, size):
    w, h = size
    factor = min(w / image.width, h / image.height)
    image = image.resize((max(1, round(image.width * factor)), max(1, round(image.height * factor))),
                         Image.Resampling.BILINEAR)
    out = Image.new("RGB", size)
    out.paste(image, ((w - image.width) // 2, (h - image.height) // 2))
    return out

def mosaic(front, left, right):
    out = Image.new("RGB", (320, 384))
    out.paste(letterbox(front, (320, 256)), (0, 0))
    out.paste(letterbox(left, (160, 128)), (0, 256))
    out.paste(letterbox(right, (160, 128)), (160, 256))
    return out

def tensor_images(images, device):
    array = np.stack([np.asarray(im, dtype=np.uint8).transpose(2, 0, 1) for im in images])
    return torch.as_tensor(array, device=device, dtype=torch.float32).div_(255)
