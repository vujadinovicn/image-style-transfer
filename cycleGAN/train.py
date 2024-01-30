import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from model.gan import Generator, Discriminator

from dataloader.dataset import StyleTransferDataset
from utils.config import PARAM_CONFIG, flatten
from loss.discriminator_loss import get_disc_loss
from loss.generator_loss import get_gen_loss
from utils.show_image import show_tensor_images

from skimage import color
import numpy as np
plt.rcParams["figure.figsize"] = (10, 10)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    config = flatten(PARAM_CONFIG)

    transform = transforms.Compose([
        transforms.Resize(config['load_shape']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    adv_criterion = nn.MSELoss()
    recon_criterion = nn.L1Loss()
    gen_AB = Generator(config['dim_A'], config['dim_B']).to(config['device'])
    gen_BA = Generator(config['dim_B'], config['dim_A']).to(config['device'])
    gen_opt = torch.optim.Adam(list(gen_AB.parameters()) + list(gen_BA.parameters()), lr=config['lr'], betas=(0.5, 0.999))
    disc_A = Discriminator(config['dim_A']).to(config['device'])
    disc_A_opt = torch.optim.Adam(disc_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    disc_B = Discriminator(config['dim_A']).to(config['device'])
    disc_B_opt = torch.optim.Adam(disc_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

    pretrained = True
    if pretrained:
        pre_dict = torch.load('pretrained/styleCycleGAN_42000.pth')
        gen_AB.load_state_dict(pre_dict['gen_AB'])
        gen_BA.load_state_dict(pre_dict['gen_BA'])
        gen_opt.load_state_dict(pre_dict['gen_opt'])
        disc_A.load_state_dict(pre_dict['disc_A'])
        disc_A_opt.load_state_dict(pre_dict['disc_A_opt'])
        disc_B.load_state_dict(pre_dict['disc_B'])
        disc_B_opt.load_state_dict(pre_dict['disc_B_opt'])
    else:
        gen_AB = gen_AB.apply(weights_init)
        gen_BA = gen_BA.apply(weights_init)
        disc_A = disc_A.apply(weights_init)
        disc_B = disc_B.apply(weights_init)

    dataset = StyleTransferDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 42000
    save_model = True

    for epoch in range(config['n_epochs']):
        for real_A, real_B in tqdm(dataloader):
            real_A = nn.functional.interpolate(real_A, size=config['target_shape'])
            real_B = nn.functional.interpolate(real_B, size=config['target_shape'])
            cur_batch_size = len(real_A)
            real_A = real_A.to(config['device'])
            real_B = real_B.to(config['device'])

            disc_A_opt.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            disc_A_loss = get_disc_loss(real_A, real_B, disc_A, adv_criterion)
            disc_A_loss.backward(retain_graph=True)
            disc_A_opt.step()

            disc_B_opt.zero_grad() 
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            disc_B_loss = get_disc_loss(real_B, fake_B, disc_B, adv_criterion)
            disc_B_loss.backward(retain_graph=True)
            disc_B_opt.step()

            gen_opt.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(
                real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, recon_criterion, recon_criterion
            )

            gen_loss.backward() 
            gen_opt.step()

            mean_discriminator_loss += disc_A_loss.item() / config['display_step']
            mean_generator_loss += gen_loss.item() / config['display_step']

            if cur_step % 200 == 0:
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                show_tensor_images(torch.cat([real_A, real_B]), size=(config['dim_A'], config['target_shape'], config['target_shape']))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(config['dim_B'], config['target_shape'], config['target_shape']))
                
                if save_model and cur_step % 2000 == 0 and cur_step != 42000:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_opt.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': disc_A_opt.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': disc_B_opt.state_dict()
                    }, f"cubceb/styleCycleGAN_{cur_step}.pth")
            
            cur_step += 1