from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def tensor_to_image(generated):
    tensor = generated.cpu().detach().numpy()

    if tensor.shape[0] == 1:
        tensor = tensor[0]

    tensor = tensor * 255
    tensor = np.round(tensor).clip(0, 255)
    tensor = tensor.astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))

    return Image.fromarray(tensor)