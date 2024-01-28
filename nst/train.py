import torch
import torch.nn.functional as F
from loss import compute_style_cost, compute_content_cost, total_cost

def train_step(model, generated_image, optimizer, a_C, a_S, layers):
    optimizer.zero_grad()

    a_G = model(generated_image)

    J_style = compute_style_cost(a_S, a_G, layers)
    J_content = compute_content_cost(a_C, a_G, layers[-1])
    J = total_cost(J_content, J_style)
    J.backward()

    optimizer.step()
    generated_image.data.clamp_(0, 1)

    return J.item()
