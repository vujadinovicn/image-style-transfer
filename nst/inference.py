import torch
from torch.optim import Adam
import argparse

from model import get_vgg_feature_extractor
from dataset import load_and_preprocess_image, load_and_preprocess_generated_image
from train import train_step
from utils import tensor_to_image


def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return "cuda"
    else:
        print("GPU not available. Using CPU.")
        return "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description="NST script")
    parser.add_argument("--content_image_path", required=True, help="Path to the content image")
    parser.add_argument("--style_image_path", required=True, help="Path to the style image")
    parser.add_argument("--output_path", required=True, help="Path to folder for saving images")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    device = get_device()

    content_image = load_and_preprocess_image(args.content_image_path, device)
    style_image = load_and_preprocess_image(args.style_image_path, device)
    generated_image = load_and_preprocess_generated_image(content_image, device)
    
    model, layers = get_vgg_feature_extractor(device)
    optimizer = Adam([generated_image], lr=0.001)
    generated_image.requires_grad = True

    with torch.no_grad():
        model.eval()
        a_C = model(content_image)
        a_S = model(style_image)
    
    epochs = 20000
    for i in range(epochs):
        train_step(model, generated_image, optimizer, a_C, a_S, layers)
        if i % 250 == 0:
            print(f"Epoch {i} ")
            image = tensor_to_image(generated_image)
            #imshow(image)
            image.save(f"{args.output_path}/image_{i}.jpg")
            #plt.show()