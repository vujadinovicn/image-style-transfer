from PIL import Image
import numpy as np
import torch

def tensor_to_image(tensor):
    tensor = tensor.cpu().detach().numpy()

    if tensor.shape[0] == 1:
        tensor = tensor[0]

    tensor = tensor * 255
    tensor = np.round(tensor).clip(0, 255)
    tensor = tensor.astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))

    return Image.fromarray(tensor)