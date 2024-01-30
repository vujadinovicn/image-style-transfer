import torch
from torch.optim import Adam
import argparse
from torchvision import transforms

import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from cycleGAN.model.gan import Generator
from PIL import Image
import numpy as np

from cycleGAN.utils.config import PARAM_CONFIG, flatten
from cycleGAN.utils.show_image import tensor_to_image
from nst.dataset import load_and_preprocess_generated_image
from nst.train import train_step

from nst.model import get_vgg_feature_extractor
from nst.utils import tensor_to_image

def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("GPU not available. Using CPU.")
        return "cpu"

def compare(content_image, style_image, style_selection, only_gan):
    device = get_device()

    if not only_gan:
        nst_content_image = content_image.resize((224, 224))
        transform = transforms.ToTensor()
        nst_content_image =  transform(nst_content_image).unsqueeze(0).to(device)
        style_image = style_image.resize((224, 224))
        transform = transforms.ToTensor()
        style_image =  transform(style_image).unsqueeze(0).to(device)
        generated_image = load_and_preprocess_generated_image(nst_content_image, device)

        model, layers = get_vgg_feature_extractor(device)
        optimizer = Adam([generated_image], lr=0.001)
        generated_image.requires_grad = True
        with torch.no_grad():
            model.eval()
            a_C = model(nst_content_image)
            a_S = model(style_image)

        
        nst_image = None
        epoch = 0
        while epoch < 10000:
            train_step(model, generated_image, optimizer, a_C, a_S, layers)
            epoch += 1
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} ")
                nst_image = tensor_to_image(generated_image)

    config = flatten(PARAM_CONFIG)

    transform = transforms.Compose([
        transforms.Resize(config['load_shape']),
        transforms.ToTensor(),
    ])

    gen_AB = Generator(config['dim_A'], config['dim_B']).to(config['device'])
    if style_selection == "Cubism":
        pre_dict = torch.load('cubface/styleCycleGAN_46000.pth')
    else:
        pre_dict = torch.load('novi/styleCycleGAN_62000.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])

    gan_content_image = transform(content_image)
    real_A = nn.functional.interpolate(gan_content_image, size=config['target_shape'])
    real_A = real_A.to(config['device'])

    with torch.no_grad():
        fake_B = gen_AB(real_A)
    fake_B_save = tensor_to_image(fake_B)

    if not only_gan:
        return np.concatenate((nst_image, fake_B_save.resize((224,224))), axis=1)
    else:
        return fake_B_save
            

if __name__ == "__main__":
    import gradio as gr

    iface = gr.Interface(
    fn=compare,
    inputs=[
        gr.Image(type="pil", label="Content image"),
        gr.Image(type="pil", label="NST style image"),
        gr.Radio(["Ukiyo-e", "Cubism"], label="Style selection"),
        gr.Radio(["Generate only using GAN"], label="Technique")
    ],
    outputs=gr.Image(type="numpy", label="Generated images"),
    live=False,
    )
    iface.launch()