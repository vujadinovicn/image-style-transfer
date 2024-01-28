from PIL import Image
import torch
from torchvision import transforms

def load_and_preprocess_image(image_path, device):
    image = Image.open(image_path).resize((224, 224))
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0).to(device)

def load_and_preprocess_generated_image(content_image, device):
    generated_image = torch.tensor(content_image, dtype=torch.float32)

    # noise = torch.rand_like(generated_image) * 0.3 - 0.2
    # generated_image = generated_image + noise

    return generated_image.clamp(0.0, 1.0).to(device)
