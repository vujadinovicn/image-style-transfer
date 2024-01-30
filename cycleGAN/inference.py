import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from model.gan import Generator
from PIL import Image
import numpy as np

from utils.config import PARAM_CONFIG, flatten
from utils.show_image import tensor_to_image

if __name__ == "__main__":
    config = flatten(PARAM_CONFIG)

    transform = transforms.Compose([
        transforms.Resize(config['load_shape']),
        transforms.ToTensor(),
    ])

    gen_AB = Generator(config['dim_A'], config['dim_B']).to(config['device'])
    pre_dict = torch.load('pretrained/styleCycleGAN_62000.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])

    content_item = transform(Image.open('landscapes/00000496_(3).jpg'))
    real_A = nn.functional.interpolate(content_item, size=config['target_shape'])
    real_A = real_A.to(config['device'])

    with torch.no_grad():
        fake_B = gen_AB(real_A)
    fake_B_save = tensor_to_image(fake_B)
    fake_B_save.show()
    fake_B_save.save(f"outputgan/image_faces_final_1 (1698).jpg")

